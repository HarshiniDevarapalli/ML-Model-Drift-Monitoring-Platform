"""Observational production-data quality checks, independent of drift monitoring."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.data_loader import PROJECT_ROOT
from src.preprocessing import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from src.reference import REFERENCE_DATA_PATH

QUALITY_RESULTS_DIR = PROJECT_ROOT / "results" / "data_quality"
STATUS_RANK = {"PASS": 0, "WARNING": 1, "FAIL": 2}


@dataclass(frozen=True)
class QualityThresholds:
    """Example thresholds; real production thresholds require domain calibration."""

    max_missing_percentage: float = 5.0
    max_duplicate_percentage: float = 1.0
    max_unexpected_category_percentage: float = 0.0
    max_invalid_numeric_percentage: float = 0.0


# These are physical/domain constraints, not reference-distribution thresholds.
MINIMUM_NUMERIC_VALUES = {
    "tenure": 0.0,
    "MonthlyCharges": 0.0,
    "TotalCharges": 0.0,
}


def _status_for_rate(rate: float, baseline_rate: float, threshold: float) -> str:
    if rate > threshold:
        return "FAIL"
    if rate > baseline_rate:
        return "WARNING"
    return "PASS"


def _worst_status(*statuses: str) -> str:
    return max(statuses, key=lambda status: STATUS_RANK[status], default="PASS")


def check_schema(production: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    """Report missing and unexpected columns without changing the input frame."""
    expected = list(reference.columns)
    actual = list(production.columns)
    missing = [column for column in expected if column not in production.columns]
    unexpected = [column for column in actual if column not in reference.columns]
    return {
        "status": "FAIL" if missing or unexpected else "PASS",
        "expected_columns": expected,
        "missing_columns": missing,
        "unexpected_columns": unexpected,
        "column_order_matches_reference": actual == expected,
    }


def check_missing_values(
    production: pd.DataFrame,
    reference: pd.DataFrame,
    thresholds: QualityThresholds,
) -> dict[str, Any]:
    """Compare per-feature production missingness with baseline and threshold."""
    columns: dict[str, Any] = {}
    statuses: list[str] = []
    for column in reference.columns:
        if column not in production.columns:
            continue
        baseline_rate = float(reference[column].isna().mean() * 100)
        production_count = int(production[column].isna().sum())
        production_rate = float(production[column].isna().mean() * 100)
        status = _status_for_rate(
            production_rate, baseline_rate, thresholds.max_missing_percentage
        )
        columns[column] = {
            "reference_missing_count": int(reference[column].isna().sum()),
            "reference_missing_percentage": baseline_rate,
            "production_missing_count": production_count,
            "production_missing_percentage": production_rate,
            "status": status,
        }
        statuses.append(status)
    return {"status": _worst_status(*statuses), "columns": columns}


def check_duplicates(production: pd.DataFrame, thresholds: QualityThresholds) -> dict[str, Any]:
    """Report exact duplicate production rows and threshold status."""
    duplicate_count = int(production.duplicated().sum())
    duplicate_percentage = float(duplicate_count / len(production) * 100) if len(production) else 0.0
    return {
        "status": _status_for_rate(
            duplicate_percentage, 0.0, thresholds.max_duplicate_percentage
        ),
        "duplicate_count": duplicate_count,
        "duplicate_percentage": duplicate_percentage,
    }


def check_categorical_values(
    production: pd.DataFrame,
    reference: pd.DataFrame,
    thresholds: QualityThresholds,
) -> dict[str, Any]:
    """Identify non-null category values absent from the reference feature data."""
    columns: dict[str, Any] = {}
    statuses: list[str] = []
    for column in CATEGORICAL_FEATURES:
        if column not in production.columns or column not in reference.columns:
            continue
        expected = set(reference[column].dropna().astype(str))
        observed = production[column].dropna().astype(str)
        unexpected = observed[~observed.isin(expected)]
        counts = unexpected.value_counts().sort_index()
        percentage = float(len(unexpected) / len(production) * 100) if len(production) else 0.0
        status = _status_for_rate(
            percentage, 0.0, thresholds.max_unexpected_category_percentage
        )
        columns[column] = {
            "expected_category_count": len(expected),
            "unexpected_counts": {str(value): int(count) for value, count in counts.items()},
            "unexpected_percentage": percentage,
            "status": status,
        }
        statuses.append(status)
    return {"status": _worst_status(*statuses), "columns": columns}


def check_numeric_validity(
    production: pd.DataFrame,
    thresholds: QualityThresholds,
) -> dict[str, Any]:
    """Detect non-numeric and non-finite numeric entries, excluding ordinary nulls."""
    columns: dict[str, Any] = {}
    statuses: list[str] = []
    for column in NUMERICAL_FEATURES:
        if column not in production.columns:
            continue
        raw = production[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        non_numeric = raw.notna() & numeric.isna()
        non_finite = numeric.notna() & ~np.isfinite(numeric)
        invalid_count = int(non_numeric.sum() + non_finite.sum())
        invalid_percentage = float(invalid_count / len(production) * 100) if len(production) else 0.0
        status = _status_for_rate(
            invalid_percentage, 0.0, thresholds.max_invalid_numeric_percentage
        )
        columns[column] = {
            "non_numeric_count": int(non_numeric.sum()),
            "non_finite_count": int(non_finite.sum()),
            "invalid_percentage": invalid_percentage,
            "status": status,
        }
        statuses.append(status)
    return {"status": _worst_status(*statuses), "columns": columns}


def check_range_constraints(production: pd.DataFrame) -> dict[str, Any]:
    """Apply only explicit non-negative constraints justified by feature meaning."""
    columns: dict[str, Any] = {}
    statuses: list[str] = []
    for column, minimum in MINIMUM_NUMERIC_VALUES.items():
        if column not in production.columns:
            continue
        numeric = pd.to_numeric(production[column], errors="coerce")
        violations = numeric.notna() & (numeric < minimum)
        status = "FAIL" if violations.any() else "PASS"
        columns[column] = {
            "minimum_allowed": minimum,
            "violation_count": int(violations.sum()),
            "violation_percentage": float(violations.mean() * 100),
            "status": status,
        }
        statuses.append(status)
    return {"status": _worst_status(*statuses), "columns": columns}


def check_data_types(production: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    """Check cleaned semantic types; numeric features must remain numeric."""
    columns: dict[str, Any] = {}
    statuses: list[str] = []
    for column in reference.columns:
        if column not in production.columns:
            continue
        expected_kind = "numeric" if column in NUMERICAL_FEATURES else "categorical"
        compatible = (
            is_numeric_dtype(production[column])
            if expected_kind == "numeric"
            else not is_numeric_dtype(production[column]) or column == "SeniorCitizen"
        )
        status = "PASS" if compatible else "FAIL"
        columns[column] = {
            "expected_semantic_type": expected_kind,
            "reference_dtype": str(reference[column].dtype),
            "production_dtype": str(production[column].dtype),
            "status": status,
        }
        statuses.append(status)
    return {"status": _worst_status(*statuses), "columns": columns}


def run_data_quality_checks(
    production: pd.DataFrame,
    reference: pd.DataFrame,
    thresholds: QualityThresholds = QualityThresholds(),
) -> dict[str, Any]:
    """Return a structured quality report; neither input dataframe is changed."""
    checks = {
        "schema": check_schema(production, reference),
        "missing_values": check_missing_values(production, reference, thresholds),
        "duplicates": check_duplicates(production, thresholds),
        "categorical_values": check_categorical_values(production, reference, thresholds),
        "numeric_validity": check_numeric_validity(production, thresholds),
        "range_constraints": check_range_constraints(production),
        "data_types": check_data_types(production, reference),
    }
    return {
        "status": _worst_status(*(check["status"] for check in checks.values())),
        "row_count": int(len(production)),
        "thresholds": asdict(thresholds),
        "checks": checks,
    }


def save_quality_report(report: dict[str, Any], path: Path) -> Path:
    """Save a JSON quality report without modifying its observed batch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run observational production data-quality checks.")
    parser.add_argument("--batch", type=Path, required=True, help="Production batch CSV path")
    parser.add_argument("--reference", type=Path, default=REFERENCE_DATA_PATH)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    production_batch = pd.read_csv(args.batch)
    reference_data = pd.read_csv(args.reference)
    report = run_data_quality_checks(production_batch, reference_data)
    output = args.output or QUALITY_RESULTS_DIR / f"{args.batch.stem}_quality_report.json"
    save_quality_report(report, output)
    print(f"Quality status: {report['status']}")
    print(f"Saved report: {output}")
