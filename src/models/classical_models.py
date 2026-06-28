import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.evaluation.metrics import evaluate_all


# --------------------------------------------------
# MODEL 4 — Holt-Winters Exponential Smoothing
# --------------------------------------------------

def holt_winters_forecast(train_series, horizon=12, seasonal_periods=52):
    """
    Holt-Winters — handles TREND and SEASONALITY together.
    
    Why use this? Unlike naive/moving average, Holt-Winters actually
    understands that sales have:
        - A direction (going up or down over time) = TREND
        - A repeating pattern every year = SEASONALITY
    
    Parameters:
        train_series    — a pandas Series with DatetimeIndex
        horizon         — how many weeks to forecast
        seasonal_periods — 52 for weekly data (52 weeks in a year)
    
    The model has 3 components:
        trend='add'      — trend goes up/down in a straight line (additive)
        seasonal='add'   — seasonal pattern is added on top (additive)
        seasonal_periods — one full season = 52 weeks
    """
    print("  Fitting Holt-Winters model...")
    
    model = ExponentialSmoothing(
        train_series,
        trend="add",            # additive trend
        seasonal="add",         # additive seasonality
        seasonal_periods=seasonal_periods,
        initialization_method="estimated"
    )
    
    fitted_model = model.fit(optimized=True)  # auto-tune the smoothing parameters
    predictions = fitted_model.forecast(horizon)
    
    # Make sure no negative predictions (sales can't be negative)
    predictions = np.maximum(predictions, 0)
    
    print(f"  Done. First 3 predictions: {predictions[:3].round(0).tolist()}")
    return predictions


# --------------------------------------------------
# MODEL 5 — ARIMA
# --------------------------------------------------

def arima_forecast(train_series, horizon=12, order=(1, 1, 1)):
    """
    ARIMA — AutoRegressive Integrated Moving Average.
    
    What does (p, d, q) mean?
        p = AR order — how many past values to use (from PACF plot in Phase 2)
        d = differencing — how many times to difference to make it stationary
            d=1 means: instead of sales, model the CHANGE in sales
        q = MA order — how many past errors to use (from ACF plot in Phase 2)
    
    Default (1,1,1) is a safe starting point for most datasets.
    
    Why ARIMA? It's the most well-known time series model.
    It's interpretable — you can explain exactly how it works.
    Required knowledge for any data science job interview!
    
    Input:
        train_series — pandas Series with DatetimeIndex
        horizon      — weeks to forecast
        order        — (p, d, q) tuple
    """
    print(f"  Fitting ARIMA{order} model...")
    
    model = ARIMA(train_series, order=order)
    fitted_model = model.fit()
    
    # Get forecast
    forecast_result = fitted_model.forecast(steps=horizon)
    predictions = np.maximum(forecast_result.values, 0)
    
    print(f"  Done. First 3 predictions: {predictions[:3].round(0).tolist()}")
    return predictions


# --------------------------------------------------
# MODEL 6 — SARIMA
# --------------------------------------------------

def sarima_forecast(train_series, horizon=12,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 52)):
    """
    SARIMA — Seasonal ARIMA.
    
    Why SARIMA over ARIMA? ARIMA doesn't know about yearly seasonality.
    SARIMA adds a second set of (P, D, Q) parameters specifically for
    the seasonal pattern.
    
    Parameters:
        order          = (p, d, q)    — same as ARIMA
        seasonal_order = (P, D, Q, S) — seasonal version
            P = seasonal AR order
            D = seasonal differencing (usually 1)
            Q = seasonal MA order
            S = season length (52 for weekly data)
    
    Note: SARIMA with S=52 is SLOW to fit. We use a smaller dataset
    or simpler orders to keep it fast.
    """
    print(f"  Fitting SARIMA{order}x{seasonal_order} model...")
    print("  (This may take 1-2 minutes for weekly data...)")
    
    model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    fitted_model = model.fit(disp=False)  # disp=False hides fitting output
    forecast_result = fitted_model.forecast(steps=horizon)
    predictions = np.maximum(forecast_result.values, 0)
    
    print(f"  Done. First 3 predictions: {predictions[:3].round(0).tolist()}")
    return predictions


# --------------------------------------------------
# MODEL 7 — Prophet
# --------------------------------------------------

def prophet_forecast(train_df, horizon=12):
    """
    Prophet — developed by Facebook (Meta) for business forecasting.
    
    Why Prophet? It's great for data with:
        - Strong seasonality (weekly, yearly patterns)
        - Holiday effects
        - Missing data or outliers
        - No need to manually set parameters — it's automatic!
    
    Prophet needs the dataframe in a specific format:
        Column 'ds' = dates
        Column 'y'  = values to forecast
    That's it — very simple to use!
    
    Input:
        train_df — dataframe with 'Date' and 'Weekly_Sales' columns
        horizon  — weeks to forecast
    """
    try:
        from prophet import Prophet
    except ImportError:
        print("  Prophet not installed. Run: pip install prophet")
        return None
    
    print("  Fitting Prophet model...")
    
    # Prophet needs columns named 'ds' and 'y'
    prophet_df = pd.DataFrame({
        "ds": train_df["Date"].values,
        "y":  train_df["Weekly_Sales"].values
    })
    
    # Create and fit the model
    model = Prophet(
        yearly_seasonality=True,   # learn yearly pattern
        weekly_seasonality=False,  # weekly data — no weekly pattern needed
        daily_seasonality=False,   # no daily pattern in weekly data
        seasonality_mode="additive"
    )
    
    model.fit(prophet_df)
    
    # Create future dates to forecast
    future = model.make_future_dataframe(periods=horizon, freq="W")
    
    # Get predictions
    forecast = model.predict(future)
    
    # Take only the future predictions (last 'horizon' rows)
    predictions = forecast["yhat"].tail(horizon).values
    predictions = np.maximum(predictions, 0)
    
    print(f"  Done. First 3 predictions: {predictions[:3].round(0).tolist()}")
    return predictions