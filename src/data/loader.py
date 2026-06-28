import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

# These columns MUST exist in train.csv — if missing, we stop immediately
REQUIRED_COLUMNS = ["Store", "Dept", "Date", "Weekly_Sales", "IsHoliday"]


def load_train(path="data/raw/train.csv"):
    """
    Loads the main training data from train.csv.
    - Parses the Date column properly
    - Checks required columns exist
    - Sorts by Store, Dept, Date (important for time series!)
    """
    logger.info(f"Loading training data from: {path}")

    df = pd.read_csv(path, parse_dates=["Date"])

    # Check all required columns are present
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in train.csv: {missing}")

    # Sort by Store -> Dept -> Date (time series must be in order!)
    df = df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

    logger.info(
        f"Loaded {len(df):,} rows | "
        f"{df['Store'].nunique()} stores | "
        f"{df['Dept'].nunique()} departments | "
        f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}"
    )
    return df


def load_stores(path="data/raw/stores.csv"):
    """
    Loads store metadata — store Type (A, B, C) and Size.
    Returns None if file doesn't exist (it's optional).
    """
    if not Path(path).exists():
        logger.warning(f"stores.csv not found at {path} — skipping.")
        return None

    df = pd.read_csv(path)
    logger.info(f"Loaded stores data: {df.shape[0]} stores")
    return df


def load_features(path="data/raw/features.csv"):
    """
    Loads external features — Temperature, Fuel Price, CPI,
    Unemployment, and MarkDown promotions.
    Returns None if file doesn't exist.
    """
    if not Path(path).exists():
        logger.warning(f"features.csv not found at {path} — skipping.")
        return None

    df = pd.read_csv(path, parse_dates=["Date"])
    logger.info(f"Loaded features data: {df.shape[0]} rows")
    return df


def merge_all(train, stores=None, features=None):
    """
    Merges train.csv with stores.csv and features.csv.
    Why? The main train.csv doesn't have Temperature, CPI, etc.
    Those are in features.csv and we need to join them together.
    """
    df = train.copy()

    if stores is not None:
        df = df.merge(stores, on="Store", how="left")
        logger.info("Merged store metadata (Type, Size).")

    if features is not None:
       
        feat_cols = [c for c in features.columns if c != "IsHoliday"]
        df = df.merge(features[feat_cols], on=["Store", "Date"], how="left")
        logger.info("Merged external features (Temperature, CPI, etc.).")

    return df