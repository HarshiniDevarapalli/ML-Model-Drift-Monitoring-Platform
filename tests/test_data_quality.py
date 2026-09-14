"""Tests for independent production data-quality monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_quality import QualityThresholds, run_data_quality_checks
from src.preprocessing import FEATURE_COLUMNS
from src.simulation import (
    generate_healthy_batch,
    generate_invalid_value_batch,
    generate_missing_value_batch,
)


def _reference_features() -> pd.DataFrame:
    rows = []
    for index in range(40):
        rows.append({
            "tenure": index + 1, "MonthlyCharges": 30.0 + index,
            "TotalCharges": 30.0 * (index + 1), "gender": "Female" if index % 2 else "Male",
            "SeniorCitizen": str(index % 2), "Partner": "Yes" if index % 3 else "No",
            "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "DSL" if index % 2 else "Fiber optic",
            "OnlineSecurity": "Yes" if index % 2 else "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "Month-to-month" if index % 3 else "One year",
            "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        })
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def test_healthy_batch_passes_and_reference_is_unchanged() -> None:
    reference = _reference_features()
    original = reference.copy(deep=True)
    healthy = generate_healthy_batch(reference, batch_size=30, rng=np.random.default_rng(1))
    report = run_data_quality_checks(healthy, reference)
    assert report["status"] == "PASS"
    assert reference.equals(original)


def test_missing_values_are_detected() -> None:
    reference = _reference_features()
    batch = generate_missing_value_batch(
        reference, batch_size=30, rng=np.random.default_rng(2), missing_fraction=0.20
    )
    report = run_data_quality_checks(batch, reference)
    assert report["status"] == "FAIL"
    assert report["checks"]["missing_values"]["columns"]["MonthlyCharges"]["status"] == "FAIL"


def test_invalid_categories_and_negative_values_are_detected() -> None:
    reference = _reference_features()
    batch = generate_invalid_value_batch(
        reference, batch_size=30, rng=np.random.default_rng(3), invalid_fraction=0.10
    )
    report = run_data_quality_checks(batch, reference)
    assert report["checks"]["categorical_values"]["columns"]["InternetService"]["unexpected_counts"] == {"Satellite": 3}
    assert report["checks"]["range_constraints"]["columns"]["MonthlyCharges"]["violation_count"] == 3
    assert report["status"] == "FAIL"


def test_duplicates_missing_and_unexpected_columns_are_detected() -> None:
    reference = _reference_features()
    duplicate_batch = pd.concat([reference.iloc[:10], reference.iloc[:1]], ignore_index=True)
    duplicate_report = run_data_quality_checks(
        duplicate_batch, reference, QualityThresholds(max_duplicate_percentage=10.0)
    )
    assert duplicate_report["checks"]["duplicates"]["duplicate_count"] == 1
    assert duplicate_report["checks"]["duplicates"]["status"] == "WARNING"

    broken = reference.drop(columns=["Contract"]).copy()
    broken["unexpected"] = "value"
    schema_report = run_data_quality_checks(broken, reference)
    assert schema_report["checks"]["schema"]["missing_columns"] == ["Contract"]
    assert schema_report["checks"]["schema"]["unexpected_columns"] == ["unexpected"]
    assert schema_report["status"] == "FAIL"


def test_valid_categories_are_not_flagged_and_thresholds_change_status() -> None:
    reference = _reference_features()
    batch = generate_healthy_batch(reference, batch_size=20, rng=np.random.default_rng(4))
    batch.loc[0, "Contract"] = np.nan
    report = run_data_quality_checks(
        batch, reference, QualityThresholds(max_missing_percentage=10.0)
    )
    assert report["checks"]["categorical_values"]["status"] == "PASS"
    assert report["checks"]["missing_values"]["columns"]["Contract"]["status"] == "WARNING"
    assert report["status"] == "WARNING"
