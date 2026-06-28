import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from src.evaluation.metrics import evaluate_all


# -------------------------------------------------------
# STEP 1 — Prepare data for ML models
# -------------------------------------------------------

def prepare_ml_data(df):
    """
    ML models need only numbers — no dates, no text.
    This function:
        1. Removes columns we cannot use (Date, raw sales)
        2. Converts text columns (Type = A/B/C) to numbers
        3. Returns X (features) and y (target = Weekly_Sales)
    
    Why remove Date?
        We already extracted month, week_of_year etc from Date in Phase 3.
        The raw date string is useless to ML models.
    
    Why convert Type?
        LightGBM cannot handle "A", "B", "C" — needs 0, 1, 2.
    """
    df = df.copy()
    
    # Convert store Type (A, B, C) to numbers if it exists
    if "Type" in df.columns:
        encoder = LabelEncoder()
        df["Type"] = encoder.fit_transform(df["Type"].astype(str))
    
    # Convert IsHoliday True/False to 1/0
    if "IsHoliday" in df.columns:
        df["IsHoliday"] = df["IsHoliday"].astype(int)
    
    # These columns are NOT features — remove them
    columns_to_drop = [
        "Date",          # replaced by month, week_of_year etc
        "Weekly_Sales",  # this is what we are predicting (target)
    ]
    
    # Drop only columns that actually exist
    columns_to_drop = [c for c in columns_to_drop if c in df.columns]
    
    X = df.drop(columns=columns_to_drop)
    y = df["Weekly_Sales"]
    
    # Fill any remaining NaN values with 0
    X = X.fillna(0)
    
    return X, y


def time_series_split(df, test_weeks=12):
    """
    Simple time-based split.
    
    Train = everything BEFORE the last test_weeks
    Test  = last test_weeks weeks
    
    Why not random split?
    In time series, future data cannot be used to train a model
    that predicts the past. Always split by time.
    """
    all_dates  = sorted(df["Date"].unique())
    cutoff_date = all_dates[-test_weeks]
    
    train_df = df[df["Date"] < cutoff_date].copy()
    test_df  = df[df["Date"] >= cutoff_date].copy()
    
    print(f"Train: {train_df['Date'].min().date()} → {train_df['Date'].max().date()} ({len(train_df)} rows)")
    print(f"Test:  {test_df['Date'].min().date()} → {test_df['Date'].max().date()} ({len(test_df)} rows)")
    
    return train_df, test_df


# -------------------------------------------------------
# MODEL 1 — LightGBM
# -------------------------------------------------------

def train_lightgbm(X_train, y_train, X_test, y_test):
    """
    LightGBM — Light Gradient Boosting Machine.
    
    What is gradient boosting?
    It builds many small decision trees one after another.
    Each new tree tries to fix the mistakes of the previous tree.
    Together they make a very accurate model.
    
    Why LightGBM?
    - Very fast (much faster than XGBoost on large data)
    - Handles missing values automatically
    - Usually the best performer on tabular/retail data
    - Used by winners of many Kaggle competitions
    
    Parameters explained:
        n_estimators    = how many trees to build (200 is a good start)
        learning_rate   = how much each tree contributes (smaller = more careful)
        max_depth       = how deep each tree grows (deeper = more complex)
        num_leaves      = max leaves per tree (controls complexity)
        random_state    = so results are reproducible
    """
    print("  Training LightGBM...")
    
    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        verbose=-1        # hide training output
    )
    
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    predictions = np.maximum(predictions, 0)  # no negative sales
    
    metrics = evaluate_all(y_test, predictions)
    print(f"  LightGBM — RMSE: {metrics['RMSE']:.2f} | MAPE: {metrics['MAPE']:.2f}% | R2: {metrics['R2']:.4f}")
    
    return model, predictions, metrics


# -------------------------------------------------------
# MODEL 2 — XGBoost
# -------------------------------------------------------

def train_xgboost(X_train, y_train, X_test, y_test):
    """
    XGBoost — Extreme Gradient Boosting.
    
    Very similar to LightGBM but works differently internally.
    XGBoost builds trees level by level.
    LightGBM builds trees leaf by leaf (that's why it's faster).
    
    Why include both?
    Sometimes XGBoost beats LightGBM on certain datasets.
    Good practice to try both and pick the winner.
    
    Parameters:
        n_estimators  = number of trees
        learning_rate = step size (same concept as LightGBM)
        max_depth     = tree depth
        subsample     = use 80% of data per tree (reduces overfitting)
        colsample_bytree = use 80% of features per tree (reduces overfitting)
    """
    print("  Training XGBoost...")
    
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0       # hide training output
    )
    
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    predictions = np.maximum(predictions, 0)
    
    metrics = evaluate_all(y_test, predictions)
    print(f"  XGBoost    — RMSE: {metrics['RMSE']:.2f} | MAPE: {metrics['MAPE']:.2f}% | R2: {metrics['R2']:.4f}")
    
    return model, predictions, metrics


# -------------------------------------------------------
# MODEL 3 — CatBoost
# -------------------------------------------------------

def train_catboost(X_train, y_train, X_test, y_test):
    """
    CatBoost — Gradient boosting by Yandex.
    
    Special advantage: handles categorical features automatically.
    No need to manually encode "A", "B", "C" — CatBoost does it internally.
    
    Also very good at avoiding overfitting (ordered boosting technique).
    
    Parameters:
        iterations    = number of trees (same as n_estimators)
        learning_rate = step size
        depth         = tree depth
        verbose       = 0 means silent (no output during training)
    """
    print("  Training CatBoost...")
    
    model = CatBoostRegressor(
        iterations=200,
        learning_rate=0.05,
        depth=6,
        random_state=42,
        verbose=0         # hide training output
    )
    
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    predictions = np.maximum(predictions, 0)
    
    metrics = evaluate_all(y_test, predictions)
    print(f"  CatBoost   — RMSE: {metrics['RMSE']:.2f} | MAPE: {metrics['MAPE']:.2f}% | R2: {metrics['R2']:.4f}")
    
    return model, predictions, metrics


# -------------------------------------------------------
# MODEL 4 — Random Forest
# -------------------------------------------------------

def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Random Forest — many decision trees, each trained on random
    subset of data and features. Final prediction = average of all trees.
    
    Why include Random Forest?
    - Oldest and most understood of the 4 models
    - Very stable — rarely gives terrible results
    - Good baseline to compare gradient boosting against
    - Easy to explain in interviews and reports
    
    Difference from gradient boosting:
        Random Forest  = trees built INDEPENDENTLY, then averaged
        Gradient Boost = trees built SEQUENTIALLY, each fixing previous errors
    
    Parameters:
        n_estimators = number of trees (100 is enough for RF)
        max_depth    = tree depth (None = grow fully, then we use min_samples_leaf)
        min_samples_leaf = each leaf needs at least 10 samples (prevents overfitting)
        n_jobs       = use all CPU cores (-1) to train faster
    """
    print("  Training Random Forest...")
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1         # use all CPU cores
    )
    
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    predictions = np.maximum(predictions, 0)
    
    metrics = evaluate_all(y_test, predictions)
    print(f"  Random Forest — RMSE: {metrics['RMSE']:.2f} | MAPE: {metrics['MAPE']:.2f}% | R2: {metrics['R2']:.4f}")
    
    return model, predictions, metrics


# -------------------------------------------------------
# ENSEMBLE — Combine all models
# -------------------------------------------------------

def ensemble_predictions(all_predictions, weights=None):
    """
    Ensemble = combine predictions from multiple models.
    
    Why ensemble?
    No single model is always best. Combining them often beats
    any individual model — this is called the "wisdom of crowds".
    
    Two ways to combine:
        1. Simple average   — all models count equally
        2. Weighted average — better models count more
    
    Input:
        all_predictions = dictionary of {model_name: predictions_array}
        weights         = dictionary of {model_name: weight} or None for equal weights
    
    Example:
        If LightGBM weight=0.4, XGBoost weight=0.3, CatBoost weight=0.2, RF weight=0.1
        Final = 0.4*lgb + 0.3*xgb + 0.2*cat + 0.1*rf
    """
    model_names = list(all_predictions.keys())
    
    if weights is None:
        # Equal weights — simple average
        weight_per_model = 1.0 / len(model_names)
        weights = {name: weight_per_model for name in model_names}
    
    # Weighted sum
    ensemble = np.zeros(len(list(all_predictions.values())[0]))
    
    for name, preds in all_predictions.items():
        w = weights.get(name, 0)
        ensemble += w * np.array(preds)
    
    return ensemble