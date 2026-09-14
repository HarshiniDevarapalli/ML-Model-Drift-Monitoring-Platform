# ML Model Drift Monitoring Platform

Local Python project for monitoring a customer-churn classifier after a simulated deployment. The focus is **model monitoring**—data quality, statistical drift, prediction shift, and performance degradation—not winning a leaderboard.

This repository currently includes **Milestone 1** (structure and dataset download), **Milestone 2** (exploration and preprocessing), **Milestone 3** (a baseline Random Forest), **Milestone 4** (reference-data artifacts), **Milestone 5** (controlled production-data simulation), and **Milestone 6** (data-quality monitoring). Feature/prediction drift, performance monitoring, alerts, and monitoring plots are not implemented yet.

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
├── models/                  # local saved pipeline: churn_random_forest.joblib
├── notebooks/
│   └── 01_data_exploration.ipynb
├── results/                 # local metrics, confusion matrix, importances
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # download + load raw CSV
│   ├── preprocessing.py     # cleaning, feature groups, unfitted sklearn pipeline
│   ├── train.py             # baseline training, evaluation, persistence
│   ├── reference.py         # training-data reference artifacts for monitoring
│   ├── simulation.py        # reproducible production-data scenarios
│   └── data_quality.py      # independent production-data quality reports
├── tests/
│   ├── test_preprocessing.py
│   ├── test_train.py
│   ├── test_reference.py
│   ├── test_simulation.py
│   └── test_data_quality.py
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
3. **Splits** (`split_train_val_test`): offers a stratified 60/20/20 helper (`random_state=42`). The Milestone 3 baseline uses its own stratified 80/20 train/test split and fits preprocessing on its training portion only.

Ordinal integer encodings are not used for unordered categoricals (`Contract` looks ordered, but one-hot avoids imposing a linear scale the forest does not need).

The reusable pipeline lives in Python (`src/preprocessing.py`), not in a one-off encoded CSV. A fully dummy-encoded table is not exported, because that would freeze categories learned from the full dataset.

### Data leakage prevention

Imputers and one-hot category sets are **statistics learned from data**. If they were fit on all 7,043 rows, validation and test information would leak into the transform. This project therefore:

- keeps `build_preprocessor()` unfitted;
- splits features **before** any `fit`;
- documents that `fit` belongs on the training/reference portion only.

Row-wise cleaning (parsing `TotalCharges`, mapping `Churn`) does not use other rows' statistics, so it may run before the split.

## Baseline model (Milestone 3)

The baseline is a scikit-learn `RandomForestClassifier` with 300 trees, `random_state=42`, and no class weighting or resampling. It is a sensible, low-maintenance baseline for a mixed numerical/categorical tabular problem and does not require feature scaling.

The evaluation uses a reproducible, stratified 80/20 train/test split (`random_state=42`): 5,634 training rows and 1,409 held-out test rows. The saved object is one `Pipeline` containing the Milestone 2 `ColumnTransformer` followed by the Random Forest. Calling `pipeline.fit(X_train, y_train)` ensures imputation statistics and one-hot category levels are learned only from training data.

### Held-out baseline performance

| Metric | Value |
| --- | ---: |
| Accuracy | 0.7821 |
| Precision | 0.6151 |
| Recall | 0.4786 |
| F1 | 0.5383 |
| ROC-AUC | 0.8200 |

Metrics are calculated on the held-out test set, not on training data. ROC-AUC is solid for this simple baseline, while the 0.4786 recall shows that it misses a meaningful share of churners; this is an appropriate baseline to monitor rather than an optimized final classifier. See `results/model_metrics.json`, `results/confusion_matrix.png`, `results/feature_importances.csv`, and `results/feature_importances.png` after running training. The importance table refers to transformed (including one-hot encoded) features; it is useful for a simple baseline inspection, not a causal explanation.

Run the baseline:

```bash
python -m src.train
```

This saves `models/churn_random_forest.joblib`. Reload it with joblib and pass it the cleaned model-feature frame produced by `clean_raw_dataframe()` and `split_features_target()`; this retains the exact preprocessing learned from training for future inference.

## Reference baseline (Milestone 4)

In this project, **reference data** means the cleaned, semantic feature representation from the same 5,634 rows used to train the baseline model. It contains the 19 model features before one-hot encoding: `TotalCharges` remains numeric and categorical values remain meaningful labels such as contract and payment-method names. It excludes `customerID` and `Churn`, so future monitoring can compare feature distributions without identifiers or outcome leakage. The aligned `Churn` values are stored separately for a future performance-monitoring step.

The 1,409 held-out test rows are deliberately not part of the reference dataset. They measured baseline model quality in Milestone 3; the reference describes the normal environment on which the preprocessing and model were developed. Future simulated production batches will be compared to this reference, not to the test set.

Run reference generation after training:

```bash
python -m src.reference
```

It creates local, reproducible artifacts under `results/reference/` (these derived files are gitignored):

- `reference_data.csv` — 5,634 semantic feature rows.
- `reference_targets.csv` — separately aligned churn labels.
- `reference_statistics.json` — numeric count/missingness, mean, standard deviation, median, range, and selected quantiles; categorical unique counts, frequencies, proportions, and missingness.
- `reference_prediction_statistics.json` — normal predicted-class distribution and positive-class probability summary. The baseline predicts churn for 1,492 / 5,634 reference rows (26.48%); mean churn probability is 26.75%.
- `baseline_metadata.json` — model identity and seed, feature lists, reference/training row counts, and the Milestone 3 held-out metrics.

These files establish data and prediction baselines only. No drift or data-quality detection is implemented yet.

## Production-data simulation (Milestone 5)

Future monitoring needs controlled inputs to evaluate. The simulator samples only from the 5,634-row training/reference artifact—not the held-out test set—and never changes that artifact. Every generated batch has the same 19 semantic feature columns as the reference data and is saved locally under `results/production/` (gitignored). A fixed seed makes each experiment reproducible.

Generate a scenario with:

```bash
python -m src.simulation --scenario healthy --batch-size 1000 --seed 42
```

Available scenarios are:

- `healthy` — an unmodified sample from the reference distribution.
- `numerical_drift` — shifts `MonthlyCharges` upward; configure `--monthly-charge-shift`.
- `categorical_drift` — changes the valid `Month-to-month` contract share; configure `--month-to-month-proportion`.
- `missing_values` — injects missing values into `MonthlyCharges` and `Contract`; configure `--missing-fraction`.
- `invalid_values` — inserts a small number of `Satellite` internet-service values and negative monthly charges; configure `--invalid-fraction`.
- `prediction_shift` — applies valid, churn-risk-oriented contract, service, payment, and charge changes. It changes inputs only; it does not calculate prediction drift.
- `performance_degradation` — samples normal-looking feature rows but saves separately altered churn labels in `performance_degradation_targets.csv`; configure `--target-flip-fraction`.

These scenarios represent distinct concepts. **Data drift** changes input distributions. A **data-quality failure** makes inputs incomplete or invalid. **Prediction drift** is a future measurement of changed model outputs; the `prediction_shift` scenario merely supplies inputs expected to produce it. **Performance degradation** changes the relationship between inputs and newly observed labels, so it can reduce future model performance even when feature distributions look normal.

The CLI validates schema and row count after generation and prints the intended change. For example, with 200 rows and seed 42, the numerical-drift scenario changes mean `MonthlyCharges` from $64.93 in the reference to $95.86; categorical drift produces an 80% Month-to-month share; and missing-values produces 20% missingness in each selected column. These are simulation validations, not drift-detection results.

## Data-quality monitoring (Milestone 6)

Data quality is the validation gate before any future drift analysis: it asks whether an incoming batch is structurally valid, complete, and meaningful enough to interpret. It does **not** ask whether valid values have changed distribution relative to the reference—that is feature drift and is not implemented yet. The quality monitor is independent of the future drift monitor:

```text
Production batch → data-quality report
Production batch → future drift report
```

Run a report against a generated batch:

```bash
python -m src.data_quality --batch results/production/healthy_batch.csv
```

Reports are saved locally under `results/data_quality/` (gitignored). They include overall `PASS`, `WARNING`, or `FAIL` status plus separate schema, missingness, duplicate, categorical-value, numeric-validity, range, and semantic-data-type checks. The monitor observes its input and does not reorder, drop, encode, or otherwise change the batch.

The project-example thresholds are centralized in `QualityThresholds`: 5% maximum per-feature missingness, 1% maximum exact duplicate rows, and 0% tolerance for unexpected categories or non-numeric/non-finite numeric values. A rate above the reference baseline but within threshold is a warning; exceeding threshold is a failure. These values are deliberately conservative examples and require calibration for a real system.

Range constraints are limited to clear physical/domain rules: `tenure`, `MonthlyCharges`, and `TotalCharges` may not be negative. No arbitrary upper bounds are applied. Expected categorical levels come directly from the cleaned reference artifact, so valid service values remain valid while the simulator's `Satellite` category is reported as unexpected.

With the generated 200-row examples (seed 42), the healthy batch returns `PASS`; the missing-values batch returns `FAIL` because `MonthlyCharges` and `Contract` each have 20% missingness; and the invalid-values batch returns `FAIL` with 10 unexpected `Satellite` values and 10 negative monthly charges. These are quality-check results, not drift-detection results.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.data_loader
python -m src.preprocessing
python -m src.train
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

Numerical/categorical/prediction drift, performance monitoring, alerts, visualizations, and end-to-end experiments.
