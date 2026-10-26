"""Step 1: load and inspect the dataset (pandas)."""
from __future__ import annotations
from pathlib import Path
from typing import Tuple

import pandas as pd

from .config import DATA_CSV, FEATURES, LABEL


class DatasetError(RuntimeError):
    pass


def load_dataset(path: Path | str | None = None) -> pd.DataFrame:
    path = Path(path or DATA_CSV)
    if not path.exists():
        raise DatasetError(
            f"Dataset not found: {path}\n"
            "Download Crop_recommendation.csv from\n"
            "  https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset\n"
            "and place it in the data/ folder (or run: python scripts/make_sample_data.py for a "
            "synthetic stand-in used only for smoke-testing)."
        )
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [LABEL] if c not in df.columns]
    if missing:
        raise DatasetError(f"{path.name} is missing expected columns: {missing}. Found: {list(df.columns)}")
    return df


def inspect_dataset(df: pd.DataFrame) -> str:
    """Columns, dtypes, missing values, duplicates, summary stats, label distribution."""
    out = []
    out.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    out.append("\nColumns and dtypes:\n" + df.dtypes.to_string())
    out.append("\nMissing values per column:\n" + df.isna().sum().to_string())
    out.append(f"\nDuplicate rows: {int(df.duplicated().sum())}")
    out.append("\nSummary statistics:\n" + df[FEATURES].describe().round(2).T.to_string())
    dist = df[LABEL].value_counts()
    out.append(f"\nLabel distribution ({dist.shape[0]} classes):\n" + dist.to_string())
    return "\n".join(out)


class DatasetStats:
    """Descriptive statistics from the CSV used by the Soil Analysis Agent (CSV / pandas tool)."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def crops(self) -> list[str]:
        return sorted(self.df[LABEL].unique().tolist())

    def overall_range(self, feature: str) -> Tuple[float, float]:
        return float(self.df[feature].min()), float(self.df[feature].max())

    def percentile(self, feature: str, value: float) -> float:
        """Percent of dataset rows whose value is <= the supplied value."""
        return float((self.df[feature] <= value).mean() * 100)

    def level(self, feature: str, value: float) -> str:
        p = self.percentile(feature, value)
        return "low" if p < 25 else "high" if p > 75 else "medium"

    def crop_range(self, crop: str, feature: str) -> Tuple[float, float]:
        """5th-95th percentile of `feature` among rows labelled `crop`."""
        s = self.df.loc[self.df[LABEL] == crop, feature]
        return float(s.quantile(0.05)), float(s.quantile(0.95))
