"""Tests for reproducible production-data scenario generation."""

from __future__ import annotations

import pandas as pd
import pytest
import numpy as np

from src.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN
from src.simulation import (
    generate_categorical_drift_batch,
    generate_healthy_batch,
    generate_invalid_value_batch,
    generate_missing_value_batch,
    generate_numerical_drift_batch,
    generate_performance_degradation_batch,
    generate_prediction_shift_batch,
    validate_batch,
)


def _reference_features() -> pd.DataFrame:
    rows = []
    for index in range(60):
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


def test_healthy_batch_shape_schema_and_reproducibility() -> None:
    reference = _reference_features()
    first = generate_healthy_batch(reference, batch_size=25, rng=np.random.default_rng(7))
    second = generate_healthy_batch(reference, batch_size=25, rng=np.random.default_rng(7))
    different = generate_healthy_batch(reference, batch_size=25, rng=np.random.default_rng(8))
    validate_batch(first, reference, 25)
    assert first.equals(second)
    assert not first.equals(different)


def test_scenarios_do_not_mutate_reference_and_keep_schema() -> None:
    reference = _reference_features()
    original = reference.copy(deep=True)
    batches = [
        generate_numerical_drift_batch(reference, batch_size=30, rng=np.random.default_rng(1)),
        generate_categorical_drift_batch(reference, batch_size=30, rng=np.random.default_rng(2)),
        generate_missing_value_batch(reference, batch_size=30, rng=np.random.default_rng(3)),
        generate_invalid_value_batch(reference, batch_size=30, rng=np.random.default_rng(4)),
        generate_prediction_shift_batch(reference, batch_size=30, rng=np.random.default_rng(5)),
    ]
    assert reference.equals(original)
    for batch in batches:
        validate_batch(batch, reference, 30)


def test_numerical_and_categorical_drift_change_expected_distributions() -> None:
    reference = _reference_features()
    numerical = generate_numerical_drift_batch(
        reference, batch_size=60, rng=np.random.default_rng(3), monthly_charge_shift=40
    )
    categorical = generate_categorical_drift_batch(
        reference, batch_size=60, rng=np.random.default_rng(4), month_to_month_proportion=0.85
    )
    assert numerical["MonthlyCharges"].mean() > reference["MonthlyCharges"].mean() + 20
    assert (categorical["Contract"] == "Month-to-month").mean() == pytest.approx(0.85)


def test_missing_and_invalid_value_scenarios_create_intended_values() -> None:
    reference = _reference_features()
    missing = generate_missing_value_batch(
        reference, batch_size=50, rng=np.random.default_rng(1), missing_fraction=0.20
    )
    invalid = generate_invalid_value_batch(
        reference, batch_size=50, rng=np.random.default_rng(2), invalid_fraction=0.10
    )
    assert missing["MonthlyCharges"].isna().mean() == pytest.approx(0.20)
    assert missing["Contract"].isna().mean() == pytest.approx(0.20)
    assert (invalid["InternetService"] == "Satellite").sum() == 5
    assert (invalid["MonthlyCharges"] < 0).sum() == 5


def test_performance_degradation_changes_separate_targets() -> None:
    reference = _reference_features()
    targets = pd.Series([0] * len(reference), name=TARGET_COLUMN)
    healthy = generate_healthy_batch(reference, batch_size=40, rng=np.random.default_rng(10))
    batch, production_targets = generate_performance_degradation_batch(
        reference, targets, batch_size=40, rng=np.random.default_rng(10), target_flip_fraction=0.60
    )
    assert batch.equals(healthy)
    assert len(production_targets) == len(batch)
    assert production_targets.sum() == 24
