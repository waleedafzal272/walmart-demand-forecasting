import pandas as pd
import numpy as np
from src.evaluation.metrics import evaluate_all


# --------------------------------------------------
# MODEL 1 — Naive Forecast
# --------------------------------------------------

def naive_forecast(train, horizon=12):
    """
    Naive model — just repeat the last known value.
    
    Why use this? It sounds too simple, but for weekly retail data
    it is surprisingly hard to beat. If your model can't beat this,
    your model has a problem.
    
    Example:
        Last week sales = 25000
        Prediction for next 12 weeks = [25000, 25000, 25000, ...]
    """
    last_value = train["Weekly_Sales"].iloc[-1]
    predictions = [last_value] * horizon
    return predictions


# --------------------------------------------------
# MODEL 2 — Simple Moving Average
# --------------------------------------------------

def moving_average_forecast(train, horizon=12, window=4):
    """
    Moving average — predict the average of the last 'window' weeks.
    
    Why use this? Smooths out random spikes. Better than naive
    when sales fluctuate week to week.
    
    Example with window=4:
        Last 4 weeks = [24000, 26000, 23000, 27000]
        Average = 25000
        Prediction for next 12 weeks = [25000, 25000, 25000, ...]
    """
    last_values = train["Weekly_Sales"].tail(window)
    average = last_values.mean()
    predictions = [average] * horizon
    return predictions


# --------------------------------------------------
# MODEL 3 — Seasonal Naive
# --------------------------------------------------

def seasonal_naive_forecast(train, horizon=12, season_length=52):
    """
    Seasonal naive — predict using the same week from last year.
    
    Why use this? Retail sales have strong yearly patterns.
    Week 48 (Thanksgiving) is always high, Week 1 is always low.
    Using last year's same week is often better than moving average.
    
    Example:
        Same week last year = 35000
        Prediction = 35000
    """
    predictions = []
    sales = train["Weekly_Sales"].values
    
    for i in range(horizon):
        # Go back exactly one season (52 weeks) + how far ahead we're predicting
        index = len(sales) - season_length + i
        if index >= 0:
            predictions.append(sales[index])
        else:
            # Not enough history — fall back to last value
            predictions.append(sales[-1])
    
    return predictions


# --------------------------------------------------
# EVALUATE ALL BASELINES TOGETHER
# --------------------------------------------------

def evaluate_baselines(train, test, horizon=12):
    """
    Run all 3 baseline models and return their scores in a table.
    
    Input:
        train — training dataframe (past data)
        test  — test dataframe (actual future values to compare against)
        horizon — how many weeks to forecast
    
    Output:
        Dictionary of {model_name: {MAE, RMSE, MAPE, SMAPE, R2}}
    """
    # Actual sales values from the test period
    actual = test["Weekly_Sales"].values[:horizon]
    
    results = {}
    
    # Run each model
    models = {
        "Naive":           naive_forecast(train, horizon),
        "Moving Average":  moving_average_forecast(train, horizon, window=4),
        "Seasonal Naive":  seasonal_naive_forecast(train, horizon),
    }
    
    for model_name, predictions in models.items():
        # Make sure prediction length matches actual length
        pred = np.array(predictions[:len(actual)])
        act  = actual[:len(pred)]
        
        metrics = evaluate_all(act, pred)
        results[model_name] = metrics
        print(f"{model_name:20s} | RMSE: {metrics['RMSE']:10.2f} | MAPE: {metrics['MAPE']:.2f}%")
    
    return results