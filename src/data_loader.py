"""Load the IBM Telco Customer Churn dataset from data/raw/."""

from __future__ import annotations

import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_FILENAME = "Telco-Customer-Churn.csv"
RAW_DATA_PATH = RAW_DATA_DIR / RAW_DATA_FILENAME

# Public CSV distributed with IBM's Telco Customer Churn code pattern.
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def download_raw_dataset(
    destination: Path = RAW_DATA_PATH,
    url: str = DATASET_URL,
) -> Path:
    """Download the raw Telco Customer Churn CSV if it is not already present."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    try:
        urllib.request.urlretrieve(url, destination)
    except urllib.error.URLError:
        # Some local Python installs lack a complete CA bundle; curl uses the OS store.
        subprocess.run(
            ["curl", "-fsSL", "-o", str(destination), url],
            check=True,
        )
    return destination


def load_raw_dataframe(path: Path | None = None) -> pd.DataFrame:
    """Load the raw CSV without cleaning or type coercion beyond pandas defaults."""
    csv_path = path or download_raw_dataset()
    return pd.read_csv(csv_path)


if __name__ == "__main__":
    path = download_raw_dataset()
    print(f"Raw dataset available at: {path}")
