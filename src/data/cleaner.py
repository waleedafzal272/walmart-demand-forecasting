import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


def clean(df):
    """
    Master cleaning function. Runs all cleaning steps in order.
    Returns: (cleaned_dataframe, report_dictionary)

    The report tells you exactly how many rows were fixed in each step.
    """
    report = {}
    original_rows = len(df)

    df = df.copy()  # never modify the original dataframe

    df, report["duplicates_removed"]    = _remove_duplicates(df)
    df, report["negative_sales_fixed"]  = _fix_negative_sales(df)
    df, report["outliers_capped"]       = _cap_outliers(df)
    df, report["missing_values_filled"] = _fill_missing(df)

    logger.info(
        f"Cleaning done: {original_rows} rows → {len(df)} rows | "
        + " | ".join(f"{k}: {v}" for k, v in report.items())
    )
    return df, report


def _remove_duplicates(df):
    """
    Remove rows where Store + Dept + Date combination is repeated.
    Why? Duplicate rows would make our model train on the same data twice.
    """
    before = len(df)
    df = df.drop_duplicates(subset=["Store", "Dept", "Date"], keep="last")
    removed = before - len(df)
    if removed > 0:
        logger.warning(f"Removed {removed} duplicate rows.")
    return df, removed


def _fix_negative_sales(df):
    """
    Set negative Weekly_Sales values to 0.
    Why? Negative sales don't make real-world sense — they're data errors.
    Clipping to 0 is standard practice.
    """
    mask = df["Weekly_Sales"] < 0
    count = int(mask.sum())
    if count > 0:
        logger.warning(f"Fixed {count} negative sales values → set to 0.")
        df.loc[mask, "Weekly_Sales"] = 0.0
    return df, count


def _cap_outliers(df, target="Weekly_Sales", z_thresh=5.0):
    """
    Cap extreme outliers using z-score per (Store, Dept) group.
    Why? A single Department might have an extreme spike (e.g. $1 billion)
    that is clearly a data error. z=5 is very conservative — only the most
    extreme spikes get capped.

    Uses transform() to preserve all columns (important for pandas 2.x).
    """
    original = df[target].copy()

    def cap_group(series):
        mu = series.mean()
        sigma = series.std()
        if sigma == 0:
            return series
        upper_limit = mu + z_thresh * sigma
        return series.clip(upper=upper_limit)

    df[target] = df.groupby(["Store", "Dept"])[target].transform(cap_group)
    capped = int((df[target] < original).sum())

    if capped > 0:
        logger.info(f"Capped {capped} outlier values (z-score > {z_thresh}).")
    return df, capped


def _fill_missing(df):
    """
    Fill missing values — different strategy for different columns:

    - MarkDown columns → fill with 0
      Why? Missing MarkDown means there was NO promotion that week. 0 is correct.

    - Temperature, Fuel_Price, CPI, Unemployment → forward fill then median
      Why? These change slowly over time. The previous week's value is a
      good estimate for a missing week.
    """
    total_filled = 0

   
    markdown_cols = [c for c in df.columns if "MarkDown" in c]
    for col in markdown_cols:
        n = df[col].isna().sum()
        df[col] = df[col].fillna(0.0)
        total_filled += n

    
    external_cols = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]
    for col in external_cols:
        if col not in df.columns:
            continue
        n = df[col].isna().sum()
        if n > 0:
            df[col] = df[col].ffill().fillna(df[col].median())
            total_filled += n

    if total_filled > 0:
        logger.info(f"Filled {total_filled} missing values.")
    return df, total_filled


def summary_stats(df):
    """
    Returns a summary table of the cleaned dataset.
    Shows mean, std, min, max, and how many missing values each column has.
    """
    numeric_cols = df.select_dtypes(include=np.number)
    stats = numeric_cols.describe().T
    stats["missing"] = df[numeric_cols.columns].isna().sum()
    stats["missing_%"] = (stats["missing"] / len(df) * 100).round(2)
    return stats