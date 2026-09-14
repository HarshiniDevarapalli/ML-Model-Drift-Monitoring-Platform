"""Create semantic training-data reference artifacts for later monitoring."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data_loader import PROJECT_ROOT, load_raw_dataframe
from src.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    clean_raw_dataframe,
    split_features_target,
)
from src.train import METRICS_PATH, MODEL_PATH, make_train_test_split

REFERENCE_DIR = PROJECT_ROOT / "results" / "reference"
REFERENCE_DATA_PATH = REFERENCE_DIR / "reference_data.csv"
REFERENCE_TARGETS_PATH = REFERENCE_DIR / "reference_targets.csv"
REFERENCE_STATISTICS_PATH = REFERENCE_DIR / "reference_statistics.json"
REFERENCE_PREDICTIONS_PATH = REFERENCE_DIR / "reference_prediction_statistics.json"
BASELINE_METADATA_PATH = REFERENCE_DIR / "baseline_metadata.json"


def select_reference_data(
    cleaned: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split cleaned data and return training/reference rows before test rows.

    The first pair is the reference distribution because it is exactly the
    portion on which the persisted model's preprocessing and forest were fit.
    The held-out test pair remains solely for model evaluation.
    """
    X, y = split_features_target(cleaned)
    X_reference, X_test, y_reference, y_test = make_train_test_split(X, y)
    return X_reference, y_reference, X_test, y_test


def numerical_reference_statistics(reference_features: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """Calculate a concise set of distribution statistics for numeric features."""
    statistics: dict[str, dict[str, float | int]] = {}
    for column in NUMERICAL_FEATURES:
        values = reference_features[column]
        non_missing = values.dropna()
        statistics[column] = {
            "count": int(non_missing.count()),
            "missing_count": int(values.isna().sum()),
            "missing_percentage": float(values.isna().mean() * 100),
            "mean": float(non_missing.mean()),
            "std": float(non_missing.std()),
            "median": float(non_missing.median()),
            "min": float(non_missing.min()),
            "max": float(non_missing.max()),
            "q05": float(non_missing.quantile(0.05)),
            "q25": float(non_missing.quantile(0.25)),
            "q75": float(non_missing.quantile(0.75)),
            "q95": float(non_missing.quantile(0.95)),
        }
    return statistics


def categorical_reference_distributions(reference_features: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Calculate category counts and proportions without altering category values."""
    distributions: dict[str, dict[str, Any]] = {}
    for column in CATEGORICAL_FEATURES:
        values = reference_features[column]
        observed = values.dropna()
        counts = observed.astype(str).value_counts().sort_index()
        distributions[column] = {
            "unique_categories": int(observed.nunique()),
            "missing_count": int(values.isna().sum()),
            "missing_percentage": float(values.isna().mean() * 100),
            "frequencies": {str(category): int(count) for category, count in counts.items()},
            "proportions": {
                str(category): float(count / len(values)) for category, count in counts.items()
            },
        }
    return distributions


def reference_prediction_statistics(
    model_pipeline: Pipeline,
    reference_features: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize the model's normal classes and positive-class probabilities."""
    predicted_classes = pd.Series(model_pipeline.predict(reference_features))
    positive_probabilities = pd.Series(model_pipeline.predict_proba(reference_features)[:, 1])
    class_counts = predicted_classes.value_counts().sort_index()
    return {
        "row_count": int(len(reference_features)),
        "predicted_class_counts": {str(label): int(count) for label, count in class_counts.items()},
        "predicted_class_proportions": {
            str(label): float(count / len(reference_features)) for label, count in class_counts.items()
        },
        "positive_class_prediction_rate": float((predicted_classes == 1).mean()),
        "positive_class_probability": {
            "mean": float(positive_probabilities.mean()),
            "std": float(positive_probabilities.std()),
            "median": float(positive_probabilities.median()),
            "min": float(positive_probabilities.min()),
            "max": float(positive_probabilities.max()),
            "q05": float(positive_probabilities.quantile(0.05)),
            "q25": float(positive_probabilities.quantile(0.25)),
            "q75": float(positive_probabilities.quantile(0.75)),
            "q95": float(positive_probabilities.quantile(0.95)),
        },
    }


def _read_metrics(path: Path) -> dict[str, float]:
    """Load the held-out Milestone 3 metrics required by baseline metadata."""
    with path.open() as file:
        return {key: float(value) for key, value in json.load(file).items()}


def _write_json(data: dict[str, Any], path: Path) -> Path:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return path


def baseline_metadata(
    model_pipeline: Pipeline,
    *,
    training_row_count: int,
    evaluation_metrics: dict[str, float],
) -> dict[str, Any]:
    """Record the minimum model context needed to interpret reference artifacts."""
    model = model_pipeline.named_steps["model"]
    return {
        "model_name": "churn_random_forest",
        "model_type": type(model).__name__,
        "model_random_seed": model.random_state,
        "training_row_count": training_row_count,
        "reference_row_count": training_row_count,
        "feature_list": FEATURE_COLUMNS,
        "numerical_feature_list": NUMERICAL_FEATURES,
        "categorical_feature_list": CATEGORICAL_FEATURES,
        "target_name": TARGET_COLUMN,
        "baseline_evaluation_metrics": evaluation_metrics,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }


def create_reference_artifacts(
    *,
    output_dir: Path = REFERENCE_DIR,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> dict[str, Path]:
    """Generate all reproducible Milestone 4 reference artifacts.

    The exported feature CSV contains only semantic model inputs. Target labels
    are stored separately and `customerID` was already excluded by
    ``split_features_target``. No one-hot encoded matrix is exported.
    """
    cleaned = clean_raw_dataframe(load_raw_dataframe())
    X_reference, y_reference, _, _ = select_reference_data(cleaned)
    model_pipeline = joblib.load(model_path)
    metrics = _read_metrics(metrics_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "reference_data": output_dir / REFERENCE_DATA_PATH.name,
        "reference_targets": output_dir / REFERENCE_TARGETS_PATH.name,
        "reference_statistics": output_dir / REFERENCE_STATISTICS_PATH.name,
        "reference_predictions": output_dir / REFERENCE_PREDICTIONS_PATH.name,
        "baseline_metadata": output_dir / BASELINE_METADATA_PATH.name,
    }
    X_reference.to_csv(paths["reference_data"], index=False)
    y_reference.to_frame(name=TARGET_COLUMN).to_csv(paths["reference_targets"], index=False)
    _write_json(
        {
            "row_count": int(len(X_reference)),
            "numerical": numerical_reference_statistics(X_reference),
            "categorical": categorical_reference_distributions(X_reference),
        },
        paths["reference_statistics"],
    )
    _write_json(
        reference_prediction_statistics(model_pipeline, X_reference),
        paths["reference_predictions"],
    )
    _write_json(
        baseline_metadata(
            model_pipeline,
            training_row_count=len(X_reference),
            evaluation_metrics=metrics,
        ),
        paths["baseline_metadata"],
    )
    return paths


def load_reference_artifacts(output_dir: Path = REFERENCE_DIR) -> dict[str, Any]:
    """Load generated reference files for later monitoring modules."""
    return {
        "reference_data": pd.read_csv(output_dir / REFERENCE_DATA_PATH.name),
        "reference_targets": pd.read_csv(output_dir / REFERENCE_TARGETS_PATH.name),
        "reference_statistics": json.loads(
            (output_dir / REFERENCE_STATISTICS_PATH.name).read_text()
        ),
        "reference_predictions": json.loads(
            (output_dir / REFERENCE_PREDICTIONS_PATH.name).read_text()
        ),
        "baseline_metadata": json.loads(
            (output_dir / BASELINE_METADATA_PATH.name).read_text()
        ),
    }


if __name__ == "__main__":
    created = create_reference_artifacts()
    print("Reference artifacts created:")
    for name, path in created.items():
        print(f"- {name}: {path}")
