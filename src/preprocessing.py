"""Deterministic cleaning and an unfitted scikit-learn preprocessing pipeline.

Cleaning (TotalCharges conversion, target encoding) is row-wise and does not
learn from other rows. The ColumnTransformer returned by build_preprocessor()
must be fitted only on training/reference rows after a split. Fitting it on
the full cleaned table would leak validation/test information into imputers
and one-hot categories.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data_loader import PROJECT_ROOT, load_raw_dataframe

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "telco_cleaned.csv"

TARGET_COLUMN = "Churn"
IDENTIFIER_COLUMNS = ["customerID"]
NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

TARGET_MAP = {"No": 0, "Yes": 1}
RANDOM_STATE = 42


def convert_total_charges(series: pd.Series) -> pd.Series:
    """Parse TotalCharges to float. Blank / non-numeric strings become NaN."""
    stripped = series.astype(str).str.strip()
    stripped = stripped.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return pd.to_numeric(stripped, errors="coerce")


def encode_target(series: pd.Series) -> pd.Series:
    """Map Churn labels No→0 and Yes→1. Unexpected labels raise ValueError."""
    values = series.astype("string").str.strip()
    valid = values.isna() | values.isin(TARGET_MAP)
    unknown = sorted(values[~valid].dropna().unique().tolist())
    if unknown:
        raise ValueError(f"Unexpected Churn labels: {unknown}")
    return values.map(TARGET_MAP).astype("Int64")


def clean_raw_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply justified quality cleaning. Does not drop rows or select features.

    TotalCharges is stored as text. Eleven customers have tenure 0 and a blank
    TotalCharges value; they have not accrued billed total yet, so those blanks
    are set to 0. Any other non-numeric TotalCharges stays missing for the
    imputer. SeniorCitizen is stored as a categorical flag, not a numeric
    quantity. Exact duplicate rows are not dropped here because the raw file
    contains none; if they appeared later they would be handled explicitly.
    """
    df = raw.copy()
    if "TotalCharges" in df.columns:
        total_charges = convert_total_charges(df["TotalCharges"])
        if "tenure" in df.columns:
            new_customer = df["tenure"].eq(0) & total_charges.isna()
            total_charges = total_charges.mask(new_customer, 0.0)
        df["TotalCharges"] = total_charges

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype("string")

    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = encode_target(df[TARGET_COLUMN])

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features and target. Identifier columns are excluded."""
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Cleaned frame is missing columns: {missing}")
    X = df.loc[:, FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified 60/20/20 split. Preprocessing must be fit on X_train only."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val,
        random_state=random_state,
        stratify=y_train_val,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor() -> ColumnTransformer:
    """Return an *unfitted* ColumnTransformer for later training/reference data.

    Random Forest does not require feature scaling, so numerical columns are
    median-imputed only. Categorical columns are imputed then one-hot encoded
    with handle_unknown='ignore' so production batches with new levels do not
    crash scoring.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def save_cleaned_dataframe(
    df: pd.DataFrame,
    path: Path = CLEANED_DATA_PATH,
) -> Path:
    """Write quality-cleaned (not one-hot encoded) rows. Raw CSV is left untouched."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    raw = load_raw_dataframe()
    cleaned = clean_raw_dataframe(raw)
    out = save_cleaned_dataframe(cleaned)
    counts = {int(k): int(v) for k, v in cleaned[TARGET_COLUMN].value_counts().sort_index().items()}
    print(f"Cleaned dataset written to: {out}")
    print(f"Rows: {len(cleaned)} (raw rows: {len(raw)}; none dropped)")
    print(f"Target 0/1 counts: {counts}")
    print(
        "Preprocessor is unfitted. Fit build_preprocessor() on training "
        "features only after split_train_val_test()."
    )
