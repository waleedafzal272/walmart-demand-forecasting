import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import optuna
import lightgbm as lgb
from src.evaluation.metrics import evaluate_all

# Turn off optuna's own log messages
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_lightgbm(X_train, y_train, X_test, y_test, n_trials=50):
    """
    Automatically find the best LightGBM settings using Optuna.
    
    What is Optuna?
    Optuna is a library that tries many different combinations
    of settings and finds which one gives the lowest error.
    
    What is a "trial"?
    One trial = train the model with one set of settings, 
                measure the error, report back.
    
    n_trials=50 means: try 50 different combinations of settings.
    More trials = better settings found, but takes longer.
    
    What settings are we tuning?
    
        n_estimators   — how many trees (100 to 500)
        learning_rate  — how fast the model learns (0.01 to 0.3)
        max_depth      — how deep each tree grows (3 to 10)
        num_leaves     — how many leaves per tree (20 to 100)
        min_child_samples — minimum samples in each leaf (10 to 100)
        subsample      — fraction of rows used per tree (0.6 to 1.0)
        colsample_bytree — fraction of columns used per tree (0.6 to 1.0)
        reg_alpha      — L1 regularization (reduces overfitting)
        reg_lambda     — L2 regularization (reduces overfitting)
    
    Returns:
        best_model   — the trained model with best settings
        best_params  — dictionary of best settings found
        best_metrics — MAE, RMSE, MAPE etc of the best model
    """
    
    print(f"Starting Optuna tuning — {n_trials} trials...")
    print("Each trial trains one model. This may take 5-10 minutes.\n")
    
    # This function runs once per trial
    def objective(trial):
        
        # Optuna suggests values for each setting
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 100),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "verbose": -1,
        }
        
        # Train model with these settings
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        
        # Predict and measure error
        preds = model.predict(X_test)
        preds = np.maximum(preds, 0)
        
        metrics = evaluate_all(y_test, preds)
        
        # Optuna minimizes this value — we want lowest RMSE
        return metrics["RMSE"]
    
    # Create study and run optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Get best settings found
    best_params = study.best_params
    best_params["random_state"] = 42
    best_params["verbose"] = -1
    
    print(f"\nBest settings found:")
    for param, value in best_params.items():
        if param not in ["random_state", "verbose"]:
            print(f"  {param:25s}: {value}")
    
    # Train final model with best settings
    print("\nTraining final model with best settings...")
    best_model = lgb.LGBMRegressor(**best_params)
    best_model.fit(X_train, y_train)
    
    # Final evaluation
    final_preds   = best_model.predict(X_test)
    final_preds   = np.maximum(final_preds, 0)
    best_metrics  = evaluate_all(y_test, final_preds)
    
    print(f"\nTuned LightGBM Results:")
    print(f"  RMSE  : {best_metrics['RMSE']:.2f}")
    print(f"  MAPE  : {best_metrics['MAPE']:.2f}%")
    print(f"  R2    : {best_metrics['R2']:.4f}")
    
    return best_model, best_params, best_metrics, final_preds


def compare_before_after(metrics_before, metrics_after):
    """
    Show clearly how much improvement tuning gave us.
    
    Prints a simple table:
        Metric | Before Tuning | After Tuning | Improvement
    """
    print("\n" + "=" * 55)
    print("IMPROVEMENT FROM TUNING")
    print("=" * 55)
    print(f"{'Metric':<10} {'Before':>12} {'After':>12} {'Change':>12}")
    print("-" * 55)
    
    for metric in ["RMSE", "MAE", "MAPE", "SMAPE", "R2"]:
        before = metrics_before[metric]
        after  = metrics_after[metric]
        
        # For R2, higher is better. For errors, lower is better.
        if metric == "R2":
            change = after - before
            symbol = "+" if change > 0 else ""
        else:
            change = before - after  # positive = improvement
            symbol = "+" if change > 0 else ""
        
        print(f"{metric:<10} {before:>12.4f} {after:>12.4f} {symbol}{change:>11.4f}")
    
    print("=" * 55)