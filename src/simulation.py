"""Generate reproducible, controlled production-like data batches.

This module simulates inputs and, for the performance-degradation scenario,
separate ground-truth labels. It intentionally does not calculate drift,
quality scores, alerts, or monitoring decisions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import PROJECT_ROOT
from src.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN
from src.reference import REFERENCE_DATA_PATH, REFERENCE_TARGETS_PATH

PRODUCTION_DIR = PROJECT_ROOT / "results" / "production"
SCENARIOS = (
    "healthy",
    "numerical_drift",
    "categorical_drift",
    "missing_values",
    "invalid_values",
    "prediction_shift",
    "performance_degradation",
)


def load_reference_features(path: Path = REFERENCE_DATA_PATH) -> pd.DataFrame:
    """Load the semantic Milestone 4 reference features without modifying them."""
    return pd.read_csv(path)


def load_reference_targets(path: Path = REFERENCE_TARGETS_PATH) -> pd.Series:
    """Load separately stored reference labels for performance simulation only."""
    return pd.read_csv(path)[TARGET_COLUMN]


def _sample_indices(row_count: int, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return rng.choice(row_count, size=batch_size, replace=batch_size > row_count)


def _sample_batch(
    reference_features: pd.DataFrame,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray]:
    indices = _sample_indices(len(reference_features), batch_size, rng)
    return reference_features.iloc[indices].reset_index(drop=True).copy(), indices


def validate_batch(batch: pd.DataFrame, reference_features: pd.DataFrame, batch_size: int) -> None:
    """Check schema and size immediately after simulation, without scoring drift."""
    if list(batch.columns) != list(reference_features.columns):
        raise ValueError("Production batch columns do not match the reference schema")
    if list(batch.columns) != FEATURE_COLUMNS:
        raise ValueError("Production batch does not match the established feature schema")
    if len(batch) != batch_size:
        raise ValueError(f"Expected {batch_size} rows, received {len(batch)}")


def generate_healthy_batch(
    reference_features: pd.DataFrame, *, batch_size: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Sample normal production-like rows from the reference distribution."""
    batch, _ = _sample_batch(reference_features, batch_size, rng)
    return batch


def generate_numerical_drift_batch(
    reference_features: pd.DataFrame,
    *,
    batch_size: int,
    rng: np.random.Generator,
    monthly_charge_shift: float = 25.0,
) -> pd.DataFrame:
    """Shift MonthlyCharges upward by a controlled monetary amount."""
    batch = generate_healthy_batch(reference_features, batch_size=batch_size, rng=rng)
    batch["MonthlyCharges"] = batch["MonthlyCharges"] + monthly_charge_shift
    return batch


def generate_categorical_drift_batch(
    reference_features: pd.DataFrame,
    *,
    batch_size: int,
    rng: np.random.Generator,
    month_to_month_proportion: float = 0.80,
) -> pd.DataFrame:
    """Increase the valid Month-to-month contract share to a target proportion."""
    if not 0.0 <= month_to_month_proportion <= 1.0:
        raise ValueError("month_to_month_proportion must be between 0 and 1")
    batch = generate_healthy_batch(reference_features, batch_size=batch_size, rng=rng)
    target_rows = int(round(batch_size * month_to_month_proportion))
    selected = rng.choice(batch.index.to_numpy(), size=target_rows, replace=False)
    remaining = batch.index.difference(selected)
    batch.loc[selected, "Contract"] = "Month-to-month"
    # Keep the requested share exact by assigning other valid contract types to the remainder.
    if len(remaining):
        valid_alternatives = reference_features.loc[
            reference_features["Contract"] != "Month-to-month", "Contract"
        ].dropna().to_numpy()
        batch.loc[remaining, "Contract"] = rng.choice(valid_alternatives, size=len(remaining))
    return batch


def generate_missing_value_batch(
    reference_features: pd.DataFrame,
    *,
    batch_size: int,
    rng: np.random.Generator,
    missing_fraction: float = 0.20,
    columns: tuple[str, ...] = ("MonthlyCharges", "Contract"),
) -> pd.DataFrame:
    """Inject a controlled rate of missing values into selected valid columns."""
    if not 0.0 <= missing_fraction <= 1.0:
        raise ValueError("missing_fraction must be between 0 and 1")
    unknown = set(columns).difference(reference_features.columns)
    if unknown:
        raise ValueError(f"Unknown missing-value columns: {sorted(unknown)}")
    batch = generate_healthy_batch(reference_features, batch_size=batch_size, rng=rng)
    missing_rows = int(round(batch_size * missing_fraction))
    for column in columns:
        selected = rng.choice(batch.index.to_numpy(), size=missing_rows, replace=False)
        batch.loc[selected, column] = np.nan
    return batch


def generate_invalid_value_batch(
    reference_features: pd.DataFrame,
    *,
    batch_size: int,
    rng: np.random.Generator,
    invalid_fraction: float = 0.05,
) -> pd.DataFrame:
    """Introduce a small number of explicit categorical and numeric violations."""
    if not 0.0 <= invalid_fraction <= 1.0:
        raise ValueError("invalid_fraction must be between 0 and 1")
    batch = generate_healthy_batch(reference_features, batch_size=batch_size, rng=rng)
    invalid_rows = int(round(batch_size * invalid_fraction))
    selected = rng.choice(batch.index.to_numpy(), size=invalid_rows, replace=False)
    batch.loc[selected, "InternetService"] = "Satellite"  # unseen category
    batch.loc[selected, "MonthlyCharges"] = -1.0  # invalid negative charge
    return batch


def generate_prediction_shift_batch(
    reference_features: pd.DataFrame,
    *,
    batch_size: int,
    rng: np.random.Generator,
    monthly_charge_shift: float = 20.0,
) -> pd.DataFrame:
    """Create valid, churn-risk-oriented inputs expected to shift model outputs."""
    batch = generate_healthy_batch(reference_features, batch_size=batch_size, rng=rng)
    batch["Contract"] = "Month-to-month"
    batch["InternetService"] = "Fiber optic"
    batch["OnlineSecurity"] = "No"
    batch["TechSupport"] = "No"
    batch["PaymentMethod"] = "Electronic check"
    batch["MonthlyCharges"] = batch["MonthlyCharges"] + monthly_charge_shift
    return batch


def generate_performance_degradation_batch(
    reference_features: pd.DataFrame,
    reference_targets: pd.Series,
    *,
    batch_size: int,
    rng: np.random.Generator,
    target_flip_fraction: float = 0.65,
) -> tuple[pd.DataFrame, pd.Series]:
    """Sample normal-looking inputs but flip labels to change the X-to-y relationship."""
    if not 0.0 <= target_flip_fraction <= 1.0:
        raise ValueError("target_flip_fraction must be between 0 and 1")
    if len(reference_features) != len(reference_targets):
        raise ValueError("Reference features and targets must have aligned row counts")
    batch, indices = _sample_batch(reference_features, batch_size, rng)
    production_targets = reference_targets.iloc[indices].reset_index(drop=True).copy()
    flip_rows = int(round(batch_size * target_flip_fraction))
    selected = rng.choice(production_targets.index.to_numpy(), size=flip_rows, replace=False)
    production_targets.loc[selected] = 1 - production_targets.loc[selected].astype(int)
    production_targets.name = TARGET_COLUMN
    return batch, production_targets


def _scenario_summary(scenario: str, reference: pd.DataFrame, batch: pd.DataFrame) -> str:
    if scenario == "numerical_drift":
        return (
            "MonthlyCharges mean: "
            f"reference={reference['MonthlyCharges'].mean():.2f}, "
            f"production={batch['MonthlyCharges'].mean():.2f}"
        )
    if scenario == "categorical_drift":
        return (
            "Month-to-month share: "
            f"reference={(reference['Contract'] == 'Month-to-month').mean():.2%}, "
            f"production={(batch['Contract'] == 'Month-to-month').mean():.2%}"
        )
    if scenario == "missing_values":
        return f"MonthlyCharges missing in production: {batch['MonthlyCharges'].isna().mean():.2%}"
    if scenario == "invalid_values":
        return f"Invalid rows: {(batch['InternetService'] == 'Satellite').sum()} Satellite / {(batch['MonthlyCharges'] < 0).sum()} negative charges"
    if scenario == "prediction_shift":
        return "Applied valid churn-risk-oriented service, contract, payment, and charge changes."
    if scenario == "performance_degradation":
        return "Feature rows are sampled normally; separately saved labels have a controlled relationship shift."
    return "Sampled from the reference distribution without intentional changes."


def generate_scenario(
    scenario: str,
    *,
    batch_size: int = 1000,
    seed: int = 42,
    output_dir: Path = PRODUCTION_DIR,
    monthly_charge_shift: float = 25.0,
    month_to_month_proportion: float = 0.80,
    missing_fraction: float = 0.20,
    invalid_fraction: float = 0.05,
    target_flip_fraction: float = 0.65,
) -> dict[str, Path]:
    """Generate, validate, and save one named production scenario."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {SCENARIOS}")
    reference = load_reference_features()
    reference_before = reference.copy(deep=True)
    rng = np.random.default_rng(seed)
    targets: pd.Series | None = None

    if scenario == "healthy":
        batch = generate_healthy_batch(reference, batch_size=batch_size, rng=rng)
    elif scenario == "numerical_drift":
        batch = generate_numerical_drift_batch(
            reference, batch_size=batch_size, rng=rng, monthly_charge_shift=monthly_charge_shift
        )
    elif scenario == "categorical_drift":
        batch = generate_categorical_drift_batch(
            reference, batch_size=batch_size, rng=rng,
            month_to_month_proportion=month_to_month_proportion,
        )
    elif scenario == "missing_values":
        batch = generate_missing_value_batch(
            reference, batch_size=batch_size, rng=rng, missing_fraction=missing_fraction
        )
    elif scenario == "invalid_values":
        batch = generate_invalid_value_batch(
            reference, batch_size=batch_size, rng=rng, invalid_fraction=invalid_fraction
        )
    elif scenario == "prediction_shift":
        batch = generate_prediction_shift_batch(
            reference, batch_size=batch_size, rng=rng, monthly_charge_shift=monthly_charge_shift
        )
    else:
        batch, targets = generate_performance_degradation_batch(
            reference, load_reference_targets(), batch_size=batch_size, rng=rng,
            target_flip_fraction=target_flip_fraction,
        )

    validate_batch(batch, reference, batch_size)
    if not reference.equals(reference_before):
        raise RuntimeError("Simulation unexpectedly modified the reference data")

    output_dir.mkdir(parents=True, exist_ok=True)
    batch_path = output_dir / f"{scenario}_batch.csv"
    batch.to_csv(batch_path, index=False)
    paths = {"batch": batch_path}
    if targets is not None:
        target_path = output_dir / f"{scenario}_targets.csv"
        targets.to_frame(name=TARGET_COLUMN).to_csv(target_path, index=False)
        paths["targets"] = target_path
    print(f"Created {scenario} batch: {batch_path} ({len(batch)} rows)")
    print(_scenario_summary(scenario, reference, batch))
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a controlled production-data scenario.")
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--monthly-charge-shift", type=float, default=25.0)
    parser.add_argument("--month-to-month-proportion", type=float, default=0.80)
    parser.add_argument("--missing-fraction", type=float, default=0.20)
    parser.add_argument("--invalid-fraction", type=float, default=0.05)
    parser.add_argument("--target-flip-fraction", type=float, default=0.65)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_scenario(
        args.scenario,
        batch_size=args.batch_size,
        seed=args.seed,
        monthly_charge_shift=args.monthly_charge_shift,
        month_to_month_proportion=args.month_to_month_proportion,
        missing_fraction=args.missing_fraction,
        invalid_fraction=args.invalid_fraction,
        target_flip_fraction=args.target_flip_fraction,
    )
