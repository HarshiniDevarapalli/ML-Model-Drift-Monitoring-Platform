"""Tests for the Random Forest training pipeline."""

from __future__ import annotations

import joblib
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.preprocessing import FEATURE_COLUMNS, clean_raw_dataframe, split_features_target
from src.train import (
    build_model_pipeline,
    evaluate_model,
    feature_importances,
    make_train_test_split,
)


def _raw_row(index: int, churn: str) -> dict[str, object]:
    return {
        "customerID": f"customer-{index}",
        "gender": "Female" if index % 2 else "Male",
        "SeniorCitizen": index % 2,
        "Partner": "Yes" if index % 3 else "No",
        "Dependents": "No",
        "tenure": index + 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL" if index % 2 else "Fiber optic",
        "OnlineSecurity": "Yes" if index % 2 else "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year" if index % 2 else "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 30.0 + index,
        "TotalCharges": str(30.0 * (index + 1)),
        "Churn": churn,
    }


def _fixture_xy() -> tuple[pd.DataFrame, pd.Series]:
    raw = pd.DataFrame([_raw_row(i, "Yes" if i % 3 == 0 else "No") for i in range(30)])
    return split_features_target(clean_raw_dataframe(raw))


def test_model_pipeline_is_unfitted_and_contains_preprocessing() -> None:
    pipeline = build_model_pipeline(n_estimators=5)
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["preprocessor", "model"]
    assert not hasattr(pipeline.named_steps["preprocessor"], "transformers_")


def test_fit_predict_evaluate_and_reload_model(tmp_path) -> None:
    X, y = _fixture_xy()
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)
    pipeline = build_model_pipeline(n_estimators=10)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    assert predictions.shape == (len(X_test),)
    assert set(predictions).issubset({0, 1})

    metrics, evaluated_predictions, probabilities = evaluate_model(pipeline, X_test, y_test)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())
    assert evaluated_predictions.shape == probabilities.shape == (len(X_test),)

    model_path = tmp_path / "churn_random_forest.joblib"
    joblib.dump(pipeline, model_path)
    reloaded = joblib.load(model_path)
    assert reloaded.predict(X_test).shape == (len(X_test),)


def test_split_happens_before_preprocessor_fitting() -> None:
    X, y = _fixture_xy()
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)
    assert len(X_train) + len(X_test) == len(X)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_train.columns) == set(FEATURE_COLUMNS)

    pipeline = build_model_pipeline(n_estimators=5)
    assert not hasattr(pipeline.named_steps["preprocessor"], "transformers_")
    pipeline.fit(X_train, y_train)
    assert hasattr(pipeline.named_steps["preprocessor"], "transformers_")
    pipeline.predict(X_test)


def test_feature_importance_matches_transformed_features() -> None:
    X, y = _fixture_xy()
    X_train, _, y_train, _ = make_train_test_split(X, y)
    pipeline = build_model_pipeline(n_estimators=5).fit(X_train, y_train)
    importances = feature_importances(pipeline)
    assert len(importances) == len(pipeline.named_steps["preprocessor"].get_feature_names_out())
    assert importances["importance"].is_monotonic_decreasing
    assert importances["importance"].sum() == pytest.approx(1.0)
