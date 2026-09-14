"""Tests for semantic reference-data artifacts."""

from __future__ import annotations

import json

import joblib
import pandas as pd
import pytest

from src.preprocessing import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERICAL_FEATURES, clean_raw_dataframe
from src.reference import (
    categorical_reference_distributions,
    create_reference_artifacts,
    load_reference_artifacts,
    numerical_reference_statistics,
    reference_prediction_statistics,
    select_reference_data,
)
from src.train import build_model_pipeline


def _raw_row(index: int, churn: str) -> dict[str, object]:
    return {
        "customerID": f"customer-{index}", "gender": "Female" if index % 2 else "Male",
        "SeniorCitizen": index % 2, "Partner": "Yes" if index % 3 else "No",
        "Dependents": "No", "tenure": index + 1, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "DSL" if index % 2 else "Fiber optic",
        "OnlineSecurity": "Yes" if index % 2 else "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "One year" if index % 2 else "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 30.0 + index, "TotalCharges": str(30.0 * (index + 1)),
        "Churn": churn,
    }


def _raw_fixture() -> pd.DataFrame:
    return pd.DataFrame([_raw_row(i, "Yes" if i % 3 == 0 else "No") for i in range(30)])


def _cleaned_fixture() -> pd.DataFrame:
    return clean_raw_dataframe(_raw_fixture())


def test_reference_selection_excludes_target_and_identifier() -> None:
    X_reference, y_reference, X_test, y_test = select_reference_data(_cleaned_fixture())
    assert list(X_reference.columns) == FEATURE_COLUMNS
    assert "Churn" not in X_reference
    assert "customerID" not in X_reference
    assert len(X_reference) + len(X_test) == 30
    assert len(y_reference) + len(y_test) == 30


def test_reference_statistics_preserve_semantic_features() -> None:
    X_reference, _, _, _ = select_reference_data(_cleaned_fixture())
    numerical = numerical_reference_statistics(X_reference)
    categorical = categorical_reference_distributions(X_reference)
    assert set(numerical) == set(NUMERICAL_FEATURES)
    assert numerical["tenure"]["count"] == len(X_reference)
    assert numerical["tenure"]["q25"] <= numerical["tenure"]["q75"]
    assert set(categorical) == set(CATEGORICAL_FEATURES)
    assert categorical["gender"]["unique_categories"] == 2
    assert sum(categorical["gender"]["frequencies"].values()) == len(X_reference)
    assert sum(categorical["gender"]["proportions"].values()) == pytest.approx(1.0)


def test_reference_prediction_statistics() -> None:
    X_reference, y_reference, _, _ = select_reference_data(_cleaned_fixture())
    pipeline = build_model_pipeline(n_estimators=5).fit(X_reference, y_reference)
    prediction_statistics = reference_prediction_statistics(pipeline, X_reference)
    assert prediction_statistics["row_count"] == len(X_reference)
    assert sum(prediction_statistics["predicted_class_counts"].values()) == len(X_reference)
    assert 0.0 <= prediction_statistics["positive_class_prediction_rate"] <= 1.0
    assert 0.0 <= prediction_statistics["positive_class_probability"]["mean"] <= 1.0


def test_reference_artifacts_are_reproducible_and_loadable(tmp_path, monkeypatch) -> None:
    cleaned = _cleaned_fixture()
    X_reference, y_reference, _, _ = select_reference_data(cleaned)
    model_path = tmp_path / "model.joblib"
    joblib.dump(build_model_pipeline(n_estimators=5).fit(X_reference, y_reference), model_path)
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"accuracy": 0.7, "roc_auc": 0.8}))
    monkeypatch.setattr("src.reference.load_raw_dataframe", _raw_fixture)

    output_dir = tmp_path / "reference"
    paths = create_reference_artifacts(
        output_dir=output_dir, model_path=model_path, metrics_path=metrics_path
    )
    assert all(path.exists() for path in paths.values())
    loaded = load_reference_artifacts(output_dir)
    assert list(loaded["reference_data"].columns) == FEATURE_COLUMNS
    assert list(loaded["reference_targets"].columns) == ["Churn"]
    assert "customerID" not in loaded["reference_data"]
    assert "Churn" not in loaded["reference_data"]
    assert loaded["baseline_metadata"]["training_row_count"] == len(X_reference)
    assert loaded["baseline_metadata"]["baseline_evaluation_metrics"]["roc_auc"] == 0.8
