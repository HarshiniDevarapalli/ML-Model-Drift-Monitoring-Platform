# ML Model Drift Monitoring Platform

Local Python project for monitoring a customer-churn classifier after a simulated deployment. The focus is **model monitoring**—data quality, statistical drift, prediction shift, and performance degradation—not winning a leaderboard.

This repository currently includes **Milestone 1** (structure and dataset download) and **Milestone 2** (exploration and a reusable preprocessing pipeline). The Random Forest, drift detectors, simulation, alerts, and monitoring plots are not implemented yet.

## Problem statement

A model that looks accurate at training time can fail in production when incoming data changes. This project will eventually train a Random Forest churn model, freeze a reference (baseline) dataset, simulate production batches, and detect quality issues, feature drift, prediction drift, and metric degradation.

## Why model monitoring matters

Production traffic rarely matches the training sample forever. Tenure mix, contract types, billing amounts, and missing fields can all shift. Without monitoring, those changes are invisible until business metrics drop. A local monitoring loop makes those failure modes measurable and reproducible.

## Architecture (current)

This is a single local Python package. There is no API, database, Docker stack, or cloud deployment.

```text
ML-Model-Drift-Monitoring-Platform/
├── data/
│   ├── raw/                 # downloaded Telco churn CSV (gitignored)
│   └── processed/           # quality-cleaned CSV from python -m src.preprocessing
├── models/                  # later: saved Random Forest
├── notebooks/
│   └── 01_data_exploration.ipynb
├── results/                 # later: figures and monitoring reports
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # download + load raw CSV
│   └── preprocessing.py     # cleaning, feature groups, unfitted sklearn pipeline
├── tests/
│   └── test_preprocessing.py
├── pytest.ini
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset overview

| Field | Value |
| --- | --- |
| Dataset name | IBM Telco Customer Churn |
| File | `data/raw/Telco-Customer-Churn.csv` |
| Rows × columns | 7,043 × 21 |
| Task | Binary classification: will the customer churn? |
| Target | `Churn` (`No` → 0, `Yes` → 1 after cleaning) |
| Acquisition | `python -m src.data_loader` |

### Source

https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

Distributed with IBM's archived code pattern [IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d). Originally IBM Cognos / Watson Analytics sample data ([IBM Community write-up](https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)).

### License / usage

- The IBM code-pattern **repository** is Apache License 2.0.
- The CSV is IBM sample / demo data (fictional customers). A separate Creative Commons-style license is not stated for the CSV, so **the raw file is not committed**. Download it locally instead.

### Target variable

Raw labels: `Churn` is `Yes` or `No`.

| Churn | Count | Share |
| --- | ---: | ---: |
| No | 5,174 | 73.46% |
| Yes | 1,869 | 26.54% |

After cleaning, those become `0` and `1` with the same counts. No oversampling or class weighting is applied in this milestone.

### Feature categories

Defined in `src/preprocessing.py` and used by the sklearn pipeline:

| Group | Columns |
| --- | --- |
| Identifier (not a model feature) | `customerID` |
| Target | `Churn` |
| Numerical | `tenure`, `MonthlyCharges`, `TotalCharges` |
| Categorical | `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod` |

`customerID` is a unique row key (7,043 distinct values). It is dropped from `X` in `split_features_target()`. `SeniorCitizen` is a 0/1 flag and is treated as categorical so one-hot encoding does not pretend it is a continuous measurement.

### Why this dataset fits monitoring

It mixes numerical billing/tenure fields with many service and contract categoricals, so later milestones can simulate numerical drift, categorical drift, missing values, invalid categories, prediction-rate shift, and label-conditioned performance drop on a realistic churn schema.

### Reference vs production data

**Reference data** is the distribution used while developing and evaluating the model (training / held-out baseline). **Production data** is new incoming observations. Monitoring will compare production batches to the reference. Those production batches are not simulated yet.

## Dataset findings (Milestone 2)

pandas reports **no `NaN` values** on the raw CSV. The important defect is hidden in `TotalCharges`.

### TotalCharges

- Stored as **text**, not float.
- **11** values are blank strings (0.16% of rows).
- All 11 have `tenure == 0` and `Churn == No`.
- Interpretation: brand-new customers with no accrued billed total.

Cleaning converts `TotalCharges` with `pd.to_numeric(..., errors="coerce")`. Blanks that coincide with `tenure == 0` are set to **0**. Any other non-numeric value would stay missing for the median imputer (none in this file). **Rows are not dropped.**

### Duplicates

Exact duplicate rows: **0**. Duplicate `customerID`: **0**. No de-duplication is applied.

### Other quality notes

- No negative `tenure` or `MonthlyCharges`.
- Service fields such as `No internet service` and `No phone service` are valid category levels, not missingness. They are left unchanged.
- No outlier clipping and no feature selection.

## Preprocessing approach

1. **Quality cleaning** (`clean_raw_dataframe`): type fixes and target encoding only. Deterministic per row. Writes `data/processed/telco_cleaned.csv` when you run `python -m src.preprocessing`. That file is still *not* one-hot encoded.
2. **Sklearn pipeline** (`build_preprocessor`): an **unfitted** `ColumnTransformer`.
   - Numerical: median imputation only. **No scaling** — the planned model is Random Forest, which does not need standardized inputs.
   - Categorical: constant imputer (`missing`) then `OneHotEncoder(handle_unknown="ignore")` so new category levels in production do not crash transform.
3. **Splits** (`split_train_val_test`): stratified 60/20/20 (`random_state=42`). Training code in a later milestone must call `preprocessor.fit(X_train)` only.

Ordinal integer encodings are not used for unordered categoricals (`Contract` looks ordered, but one-hot avoids imposing a linear scale the forest does not need).

The reusable pipeline lives in Python (`src/preprocessing.py`), not in a one-off encoded CSV. A fully dummy-encoded table is not exported, because that would freeze categories learned from the full dataset.

### Data leakage prevention

Imputers and one-hot category sets are **statistics learned from data**. If they were fit on all 7,043 rows, validation and test information would leak into the transform. This project therefore:

- keeps `build_preprocessor()` unfitted;
- splits features **before** any `fit`;
- documents that `fit` belongs on the training/reference portion only.

Row-wise cleaning (parsing `TotalCharges`, mapping `Churn`) does not use other rows' statistics, so it may run before the split.

## Machine learning model (planned)

Random Forest classifier (scikit-learn) predicting `Churn`. Not trained in this milestone.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.data_loader
python -m src.preprocessing
```

Exploration notebook:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

## How to run tests

```bash
python -m pytest
```

## Future improvements (not started)

Random Forest training, reference-set creation, production simulation, data-quality checks, numerical/categorical/prediction drift, performance monitoring, alerts, visualizations, and end-to-end experiments.
