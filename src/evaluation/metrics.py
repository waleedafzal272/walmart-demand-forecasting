import numpy as np
import pandas as pd


def mae(y_true, y_pred):
    """Mean Absolute Error — average of absolute differences."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    """Root Mean Squared Error — penalizes big mistakes more than MAE."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, eps=1e-8):
    """
    Mean Absolute Percentage Error.
    Returns a percentage, e.g. 5.2 means 5.2% average error.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = np.abs(y_true) > eps  # avoid dividing by zero
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true, y_pred, eps=1e-8):
    """
    Symmetric MAPE — handles zero actual values better than regular MAPE.
    Bounded between 0% and 200%.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + eps
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def r2_score(y_true, y_pred):
    """
    R-squared — how much of the variance our model explains.
    1.0 = perfect, 0.0 = as good as just predicting the mean, negative = bad.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-8))


def evaluate_all(y_true, y_pred):
    """
    Run all 5 metrics at once.
    Returns a dictionary like:
        {'MAE': 120.5, 'RMSE': 180.2, 'MAPE': 5.3, 'SMAPE': 5.1, 'R2': 0.87}
    """
    return {
        "MAE":   round(mae(y_true, y_pred), 4),
        "RMSE":  round(rmse(y_true, y_pred), 4),
        "MAPE":  round(mape(y_true, y_pred), 4),
        "SMAPE": round(smape(y_true, y_pred), 4),
        "R2":    round(r2_score(y_true, y_pred), 4),
    }


def metrics_dataframe(results_dict):
    """
    Converts model results to a nicely sorted table.

    Input:
        {
            'LightGBM': {'MAE': 1.2, 'RMSE': 1.8, ...},
            'ARIMA':    {'MAE': 2.1, 'RMSE': 2.9, ...},
        }
    Output:
        A pandas DataFrame sorted by RMSE (best model on top)
    """
    df = pd.DataFrame(results_dict).T
    df.index.name = "Model"
    if "RMSE" in df.columns:
        df = df.sort_values("RMSE")
    return df