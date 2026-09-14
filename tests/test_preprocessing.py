"""Unit tests for Telco churn cleaning and the unfitted sklearn preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
    clean_raw_dataframe,
    convert_total_charges,
    encode_target,
    split_features_target,
    split_train_val_test,
)


def _raw_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "customerID": "0000-AAAAA",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": "59.70",
        "Churn": "No",
    }
    row.update(overrides)
    return row


def test_encode_target_maps_yes_no() -> None:
    encoded = encode_target(pd.Series(["No", "Yes", "No"]))
    assert encoded.tolist() == [0, 1, 0]


def test_encode_target_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError, match="Unexpected Churn labels"):
        encode_target(pd.Series(["No", "maybe"]))


def test_convert_total_charges_parses_numbers_and_blanks() -> None:
    converted = convert_total_charges(pd.Series(["29.85", " ", "", "108.15"]))
    assert converted.iloc[0] == pytest.approx(29.85)
    assert pd.isna(converted.iloc[1])
    assert pd.isna(converted.iloc[2])
    assert converted.iloc[3] == pytest.approx(108.15)


def test_clean_sets_blank_total_charges_to_zero_for_new_customers() -> None:
    raw = pd.DataFrame(
        [
            _raw_row(customerID="new", tenure=0, TotalCharges=" ", Churn="No"),
            _raw_row(customerID="old", tenure=12, TotalCharges="240.5", Churn="Yes"),
        ]
    )
    cleaned = clean_raw_dataframe(raw)
    assert len(cleaned) == 2
    assert cleaned.loc[0, "TotalCharges"] == pytest.approx(0.0)
    assert cleaned.loc[1, "TotalCharges"] == pytest.approx(240.5)
    assert cleaned[TARGET_COLUMN].tolist() == [0, 1]


def test_clean_does_not_invent_total_charges_when_tenure_positive() -> None:
    raw = pd.DataFrame([_raw_row(tenure=5, TotalCharges="   ")])
    cleaned = clean_raw_dataframe(raw)
    assert pd.isna(cleaned.loc[0, "TotalCharges"])


def test_clean_keeps_senior_citizen_as_categorical_string() -> None:
    cleaned = clean_raw_dataframe(pd.DataFrame([_raw_row(SeniorCitizen=1)]))
    assert str(cleaned.loc[0, "SeniorCitizen"]) == "1"


def test_preprocessor_imputes_numeric_missing_with_median() -> None:
    train = pd.DataFrame(
        [
            _raw_row(tenure=10, MonthlyCharges=20.0, TotalCharges="100"),
            _raw_row(tenure=20, MonthlyCharges=40.0, TotalCharges="200"),
        ]
    )
    cleaned_train = clean_raw_dataframe(train)
    X_train, _ = split_features_target(cleaned_train)
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    X_prod = pd.DataFrame(
        [
            {
                **{c: cleaned_train.iloc[0][c] for c in FEATURE_COLUMNS},
                "tenure": np.nan,
                "MonthlyCharges": np.nan,
                "TotalCharges": np.nan,
            }
        ]
    )
    transformed = preprocessor.transform(X_prod)
    feature_names = list(preprocessor.get_feature_names_out())
    tenure_idx = feature_names.index("num__tenure")
    monthly_idx = feature_names.index("num__MonthlyCharges")
    total_idx = feature_names.index("num__TotalCharges")
    assert transformed[0, tenure_idx] == pytest.approx(15.0)
    assert transformed[0, monthly_idx] == pytest.approx(30.0)
    assert transformed[0, total_idx] == pytest.approx(150.0)


def test_preprocessor_one_hot_encodes_known_categories() -> None:
    train = pd.DataFrame(
        [
            _raw_row(gender="Female", Contract="Month-to-month"),
            _raw_row(gender="Male", Contract="Two year"),
        ]
    )
    X_train, _ = split_features_target(clean_raw_dataframe(train))
    preprocessor = build_preprocessor()
    Xt = preprocessor.fit_transform(X_train)
    names = list(preprocessor.get_feature_names_out())
    assert "cat__gender_Female" in names
    assert "cat__gender_Male" in names
    assert "cat__Contract_Month-to-month" in names
    assert Xt.shape[0] == 2
    assert Xt.dtype == np.float64 or np.issubdtype(Xt.dtype, np.floating)


def test_preprocessor_ignores_unseen_categorical_values() -> None:
    train = pd.DataFrame(
        [
            _raw_row(InternetService="DSL"),
            _raw_row(InternetService="Fiber optic"),
        ]
    )
    X_train, _ = split_features_target(clean_raw_dataframe(train))
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    unseen = clean_raw_dataframe(
        pd.DataFrame([_raw_row(InternetService="satellite")])
    )
    X_unseen, _ = split_features_target(unseen)
    Xt = preprocessor.transform(X_unseen)
    names = list(preprocessor.get_feature_names_out())
    internet_cols = [i for i, n in enumerate(names) if n.startswith("cat__InternetService_")]
    assert internet_cols
    assert Xt[0, internet_cols].sum() == pytest.approx(0.0)


def test_preprocessor_fit_transform_output_shape() -> None:
    rows = [_raw_row(customerID=f"id-{i}", tenure=i + 1) for i in range(6)]
    X, y = split_features_target(clean_raw_dataframe(pd.DataFrame(rows)))
    preprocessor = build_preprocessor()
    Xt = preprocessor.fit_transform(X)
    assert Xt.shape[0] == 6
    assert Xt.shape[1] == len(preprocessor.get_feature_names_out())
    assert Xt.shape[1] > len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES) - 5
    assert y.dtype == int or np.issubdtype(y.dtype, np.integer)


def test_split_does_not_fit_preprocessor_and_preserves_row_count() -> None:
    rows = [
        _raw_row(customerID=f"id-{i}", Churn="Yes" if i % 3 == 0 else "No")
        for i in range(20)
    ]
    X, y = split_features_target(clean_raw_dataframe(pd.DataFrame(rows)))
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y)
    assert len(X_train) + len(X_val) + len(X_test) == len(X)
    assert set(X_train.columns) == set(FEATURE_COLUMNS)
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    preprocessor.transform(X_val)
    preprocessor.transform(X_test)
