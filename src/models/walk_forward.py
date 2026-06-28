import pandas as pd
import numpy as np
from src.evaluation.metrics import evaluate_all


def walk_forward_split(df, n_folds=5, test_weeks=12):
    """
    Splits data into multiple train/test folds for walk-forward validation.
    
    Why? Instead of testing on just one period, we test on 5 different
    periods and average the scores. This gives a much more reliable
    estimate of how good our model really is.
    
    How it works (example with n_folds=3, test_weeks=4):
    
    Total data: 60 weeks
    
    Fold 1: Train = weeks 1-36,  Test = weeks 37-40
    Fold 2: Train = weeks 1-44,  Test = weeks 45-48  
    Fold 3: Train = weeks 1-52,  Test = weeks 53-56
    
    Each fold uses MORE training data than the previous one.
    This is called "expanding window" validation.
    
    Returns:
        List of (train_df, test_df) tuples — one per fold
    """
    splits = []
    
    # Get all unique dates in order
    all_dates = sorted(df["Date"].unique())
    total_weeks = len(all_dates)
    
    # Minimum weeks needed for training (at least 1 year)
    min_train_weeks = 52
    
    # How much to move forward each fold
    step = test_weeks
    
    for fold in range(n_folds):
        # Test set ends at different points going backwards from the end
        # Fold 0 = most recent, Fold n-1 = oldest
        test_end_idx   = total_weeks - fold * step
        test_start_idx = test_end_idx - test_weeks
        
        # Skip if not enough test data
        if test_start_idx <= min_train_weeks:
            print(f"  Fold {fold+1}: not enough data, skipping.")
            continue
        
        # Get the actual dates for train and test
        train_dates = all_dates[:test_start_idx]
        test_dates  = all_dates[test_start_idx:test_end_idx]
        
        train_df = df[df["Date"].isin(train_dates)].copy()
        test_df  = df[df["Date"].isin(test_dates)].copy()
        
        splits.append((fold + 1, train_df, test_df))
        
        print(f"  Fold {fold+1}: "
              f"Train {train_df['Date'].min().date()} → {train_df['Date'].max().date()} "
              f"({len(train_dates)} weeks) | "
              f"Test {test_df['Date'].min().date()} → {test_df['Date'].max().date()} "
              f"({len(test_dates)} weeks)")
    
    return splits


def evaluate_model_cv(model_fn, df, store, dept, n_folds=3, test_weeks=12):
    """
    Run walk-forward cross validation for ONE store-department combination.
    
    Why one store-dept at a time?
    Classical models (ARIMA, Holt-Winters) work on a single time series.
    We pick one store+dept, get its full history, then test the model.
    
    Input:
        model_fn — a function that takes (train_df, horizon) and returns predictions
        df       — full dataset
        store    — store number to use
        dept     — department number to use
        n_folds  — how many folds to run
        test_weeks — how many weeks in each test fold
    
    Output:
        Average metrics across all folds
    """
    # Filter to just this store and department
    series_df = df[
        (df["Store"] == store) &
        (df["Dept"] == dept)
    ].sort_values("Date").reset_index(drop=True)
    
    if len(series_df) < 60:
        print(f"  Not enough data for Store {store}, Dept {dept}. Skipping.")
        return None
    
    print(f"\nWalk-Forward CV — Store {store}, Dept {dept} ({len(series_df)} weeks total)")
    
    # Get the folds
    splits = walk_forward_split(series_df, n_folds=n_folds, test_weeks=test_weeks)
    
    if len(splits) == 0:
        print("  No valid folds found.")
        return None
    
    all_metrics = []
    
    for fold_num, train_df, test_df in splits:
        try:
            # Run the model
            predictions = model_fn(train_df, horizon=test_weeks)
            
            if predictions is None:
                continue
            
            # Get actual values
            actual = test_df["Weekly_Sales"].values[:len(predictions)]
            pred   = np.array(predictions)[:len(actual)]
            
            # Calculate metrics for this fold
            metrics = evaluate_all(actual, pred)
            metrics["fold"] = fold_num
            all_metrics.append(metrics)
            
            print(f"  Fold {fold_num} — RMSE: {metrics['RMSE']:.2f} | "
                  f"MAPE: {metrics['MAPE']:.2f}% | MAE: {metrics['MAE']:.2f}")
            
        except Exception as e:
            print(f"  Fold {fold_num} failed: {e}")
            continue
    
    if len(all_metrics) == 0:
        return None
    
    # Average across all folds
    avg_metrics = {
        "MAE":   round(np.mean([m["MAE"]   for m in all_metrics]), 4),
        "RMSE":  round(np.mean([m["RMSE"]  for m in all_metrics]), 4),
        "MAPE":  round(np.mean([m["MAPE"]  for m in all_metrics]), 4),
        "SMAPE": round(np.mean([m["SMAPE"] for m in all_metrics]), 4),
        "R2":    round(np.mean([m["R2"]    for m in all_metrics]), 4),
        "folds": len(all_metrics)
    }
    
    print(f"\n  Average — RMSE: {avg_metrics['RMSE']:.2f} | "
          f"MAPE: {avg_metrics['MAPE']:.2f}% | R2: {avg_metrics['R2']:.4f}")
    
    return avg_metrics