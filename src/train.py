"""Train and evaluate the leak-safe Random Forest churn baseline."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import joblib

# Avoid a user-home cache dependency when this module is imported by local scripts.
_MATPLOTLIB_CONFIG_DIR = Path(tempfile.gettempdir()) / "ml_drift_monitoring_matplotlib"
_MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CONFIG_DIR))
import matplotlib

# Training runs non-interactively (including CI), so use a file-rendering backend.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_loader import PROJECT_ROOT, load_raw_dataframe
from src.preprocessing import RANDOM_STATE, build_preprocessor, clean_raw_dataframe, split_features_target

MODEL_PATH = PROJECT_ROOT / "models" / "churn_random_forest.joblib"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_PATH = RESULTS_DIR / "model_metrics.json"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_PATH = RESULTS_DIR / "feature_importances.csv"
FEATURE_IMPORTANCE_PLOT_PATH = RESULTS_DIR / "feature_importances.png"
TEST_SIZE = 0.20
N_ESTIMATORS = 300


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return a reproducible, stratified train/test split before any fitting."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_model_pipeline(*, n_estimators: int = N_ESTIMATORS) -> Pipeline:
    """Build an unfitted preprocessing-plus-Random-Forest pipeline.

    The forest is a compact, reproducible baseline: 300 trees reduce variance
    without a hyperparameter search. No class weighting or resampling is used.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float], pd.Series, pd.Series]:
    """Evaluate a fitted pipeline on held-out data."""
    predictions = pd.Series(pipeline.predict(X_test), index=X_test.index, name="prediction")
    probabilities = pd.Series(
        pipeline.predict_proba(X_test)[:, 1], index=X_test.index, name="churn_probability"
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    return metrics, predictions, probabilities


def feature_importances(pipeline: Pipeline) -> pd.DataFrame:
    """Return transformed-feature importances from a fitted forest pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    table = pd.DataFrame(
        {
            "feature": preprocessor.get_feature_names_out(),
            "importance": model.feature_importances_,
        }
    )
    return table.sort_values("importance", ascending=False, ignore_index=True)


def save_metrics(metrics: dict[str, float], path: Path = METRICS_PATH) -> Path:
    """Save serializable baseline metrics with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return path


def save_confusion_matrix(
    y_test: pd.Series,
    predictions: pd.Series,
    path: Path = CONFUSION_MATRIX_PATH,
) -> Path:
    """Save a labeled held-out-test confusion matrix plot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No churn (0)", "Churn (1)"],
        yticklabels=["No churn (0)", "Churn (1)"],
        ax=ax,
    )
    ax.set(xlabel="Predicted", ylabel="Actual", title="Random Forest confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_feature_importance_outputs(
    importances: pd.DataFrame,
    *,
    table_path: Path = FEATURE_IMPORTANCE_PATH,
    plot_path: Path = FEATURE_IMPORTANCE_PLOT_PATH,
    top_n: int = 15,
) -> tuple[Path, Path]:
    """Save full importance table and a compact top-feature plot."""
    table_path.parent.mkdir(parents=True, exist_ok=True)
    importances.to_csv(table_path, index=False)
    top_features = importances.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_features["feature"], top_features["importance"], color="#2878b5")
    ax.set(xlabel="Random Forest importance", title=f"Top {len(top_features)} transformed features")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return table_path, plot_path


def train_baseline() -> dict[str, object]:
    """Train, evaluate, persist, and report the Milestone 3 baseline."""
    cleaned = clean_raw_dataframe(load_raw_dataframe())
    X, y = split_features_target(cleaned)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)

    pipeline = build_model_pipeline()
    # Fitting the pipeline here fits its transformer on X_train only.
    pipeline.fit(X_train, y_train)
    metrics, predictions, _ = evaluate_model(pipeline, X_test, y_test)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    save_metrics(metrics)
    save_confusion_matrix(y_test, predictions)
    importances = feature_importances(pipeline)
    save_feature_importance_outputs(importances)

    return {
        "metrics": metrics,
        "model_path": MODEL_PATH,
        "test_rows": len(X_test),
        "train_rows": len(X_train),
        "top_features": importances.head(10),
    }


if __name__ == "__main__":
    result = train_baseline()
    print(f"Train/test rows: {result['train_rows']}/{result['test_rows']}")
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    print(f"Saved model: {result['model_path']}")
    print("Top transformed features:")
    print(result["top_features"].to_string(index=False))
