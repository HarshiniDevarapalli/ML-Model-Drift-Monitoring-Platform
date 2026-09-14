# ML Model Drift Monitoring Platform

Local Python project for monitoring a customer-churn classifier after a simulated deployment. The focus is **model monitoring**—data quality, statistical drift, prediction shift, and performance degradation—not winning a leaderboard.

This repository currently contains **Milestone 1**: project structure and reproducible dataset acquisition. Training, monitoring, alerts, and experiments are not implemented yet.

## Problem statement

A model that looks accurate at training time can fail in production when incoming data changes. This project will eventually train a Random Forest churn model, freeze a reference (baseline) dataset, simulate production batches, and detect quality issues, feature drift, prediction drift, and metric degradation.

## Why model monitoring matters

Production traffic rarely matches the training sample forever. Tenure mix, contract types, billing amounts, and missing fields can all shift. Without monitoring, those changes are invisible until business metrics drop. A local monitoring loop makes those failure modes measurable and reproducible.

## Architecture (current)

This is a single local Python package. There is no API, database, Docker stack, or cloud deployment.

```text
ML-Model-Drift-Monitoring-Platform/
├── data/
│   ├── raw/              # downloaded Telco churn CSV (gitignored)
│   └── processed/        # later: cleaned / reference / production splits
├── models/               # later: saved Random Forest
├── notebooks/            # later: exploration notebooks
├── results/              # later: figures and monitoring reports
├── src/
│   ├── __init__.py
│   └── data_loader.py    # download raw dataset
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

Planned modules (`preprocessing.py`, `train.py`, drift/quality/alerts, simulation) will be added in later milestones.

## Dataset

| Field | Value |
| --- | --- |
| Dataset name | IBM Telco Customer Churn |
| File | `data/raw/Telco-Customer-Churn.csv` |
| Rows | 7,043 customers (plus header) |
| Task | Binary classification: will the customer churn? |
| Target | `Churn` (`Yes` / `No`) |
| Acquisition | `python -m src.data_loader` writes the public CSV to `data/raw/` (urllib, with a curl fallback if the local Python CA bundle is incomplete) |

### Source

Primary public file used by this project:

https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

That CSV is distributed with IBM's archived code pattern [IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d). IBM originally published the Telco Customer Churn sample (fictional California telco, Q3, 7,043 customers) as Cognos Analytics / Watson Analytics sample data. See IBM's sample description: [Telco customer churn (IBM Community)](https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113).

Verified accessible (HTTP 200) at the GitHub raw URL above before download.

### License / usage

- The IBM code-pattern **repository** is licensed under **Apache License 2.0**.
- The CSV is **IBM sample / demo data** (fictional customers). IBM's code-pattern README notes that third-party objects invoked by the pattern may have their own licenses; a separate open-data license (for example Creative Commons) is **not** stated for the CSV itself.
- Use is appropriate for education, tutorials, and portfolio demonstrations of churn modeling and monitoring.
- Because the dataset license is sample-data rather than an explicit redistribution grant, **the raw CSV is not committed**. Reproduce it locally with the download command below.

### Target and feature types

- **Target:** `Churn`
- **Identifier:** `customerID` (not a model feature)
- **Numerical features:** `tenure`, `MonthlyCharges`, `TotalCharges` (`TotalCharges` is stored as text and can contain blanks for tenure 0)
- **Binary / integer flag:** `SeniorCitizen` (`0` / `1`; often treated as categorical)
- **Categorical features:** `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`

### Why this dataset fits monitoring

It mixes numerical billing/tenure fields with many service and contract categoricals, so later milestones can simulate numerical drift, categorical drift, missing values, invalid categories, prediction-rate shift, and label-conditioned performance drop on a realistic churn schema.

### Reference vs production data

**Reference data** is the distribution used while developing and evaluating the model (training / held-out baseline). **Production data** is new incoming observations. Monitoring will compare production batches to the reference. Those splits are not created in Milestone 1.

## Machine learning model (planned)

Random Forest classifier (scikit-learn) predicting `Churn`. Not trained in this milestone.

## How to run (Milestone 1)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.data_loader
```

The CSV is written to `data/raw/Telco-Customer-Churn.csv`. Re-running the command skips the download if the file already exists.

## How to run tests

No tests yet. `pytest` is listed in `requirements.txt` for later milestones.

## Future improvements (not started)

Dataset exploration, preprocessing, Random Forest training, reference-set creation, production simulation, data-quality checks, numerical/categorical/prediction drift, performance monitoring, alerts, visualizations, and end-to-end experiments.
