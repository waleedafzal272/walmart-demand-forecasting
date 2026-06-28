import pandas as pd
import numpy as np



def add_all_features(df):

    print("Adding calendar features...")
    df = add_calendar_features(df)
    
    print("Adding lag features...")
    df = add_lag_features(df)
    
    print("Adding rolling average features...")
    df = add_rolling_features(df)
    
    print("Adding holiday features...")
    df = add_holiday_features(df)
    
    # Remove rows where lag features are NaN
    # (first few weeks won't have lag values yet)
    before = len(df)
    df = df.dropna(subset=["lag_1", "lag_4"])
    after = len(df)
    print(f"Removed {before - after} rows with missing lag values (expected)")
    
    print(f"Done! Total features: {len(df.columns)} columns")
    return df



def add_calendar_features(df):
  
    df = df.copy()
    
    
    df["month"]        = df["Date"].dt.month          
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)  
    df["quarter"]      = df["Date"].dt.quarter        
    df["year"]         = df["Date"].dt.year           
    df["day_of_year"]  = df["Date"].dt.dayofyear    
    
   
    df["is_december"]  = (df["Date"].dt.month == 12).astype(int)  
    df["is_q4"]        = (df["Date"].dt.quarter == 4).astype(int) 
    
    
    df["year_progress"] = df["day_of_year"] / 365.0
    
    return df


def add_lag_features(df):
    
    df = df.copy()
    
    # Sort properly first — very important for lag to work correctly!
    df = df.sort_values(["Store", "Dept", "Date"])
    
    # List of lag periods to create (in weeks)
    lag_weeks = [1, 2, 4, 8, 12, 26, 52]
    
    for lag in lag_weeks:
        col_name = f"lag_{lag}"
        
        # .shift(lag) moves data down by 'lag' rows within each group
        df[col_name] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(lag)
        
        # Example: if lag=4, then today's lag_4 = sales from 4 weeks ago
    
    return df





def add_rolling_features(df):
   
    df = df.copy()
    df = df.sort_values(["Store", "Dept", "Date"])
    
    windows = [4, 8, 12, 26]
    
    for window in windows:
        # shift(1) = look at data starting from 1 week ago (not including today)
        past_sales = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1)
        
        # Average sales over last 'window' weeks
        df[f"rolling_mean_{window}"] = (
            past_sales
            .groupby([df["Store"], df["Dept"]])
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        
        # Standard deviation (how much sales bounce around)
        df[f"rolling_std_{window}"] = (
            past_sales
            .groupby([df["Store"], df["Dept"]])
            .transform(lambda x: x.rolling(window, min_periods=1).std())
        )
    
    # Max and min over 4 weeks (useful for capturing seasonal peaks)
    past_sales = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1)
    
    df["rolling_max_4"] = (
        past_sales
        .groupby([df["Store"], df["Dept"]])
        .transform(lambda x: x.rolling(4, min_periods=1).max())
    )
    
    df["rolling_min_4"] = (
        past_sales
        .groupby([df["Store"], df["Dept"]])
        .transform(lambda x: x.rolling(4, min_periods=1).min())
    )
    
    return df



def add_holiday_features(df):
   
    df = df.copy()
    df = df.sort_values(["Store", "Dept", "Date"])
    
    # Convert IsHoliday to integer (True=1, False=0)
    df["is_holiday"] = df["IsHoliday"].astype(int)
    
    # Get list of all holiday dates in the dataset
    holiday_dates = df[df["IsHoliday"] == True]["Date"].unique()
    holiday_dates = sorted(holiday_dates)
    
    def weeks_until_next_holiday(date):
        """How many weeks from this date until the next holiday?"""
        future_holidays = [h for h in holiday_dates if h > date]
        if len(future_holidays) == 0:
            return 52  # no upcoming holiday, return 52 as default
        next_holiday = future_holidays[0]
        days_diff = (next_holiday - date).days
        return days_diff // 7  # convert days to weeks
    
    def weeks_since_last_holiday(date):
        """How many weeks since the last holiday?"""
        past_holidays = [h for h in holiday_dates if h <= date]
        if len(past_holidays) == 0:
            return 52  # no past holiday, return 52 as default
        last_holiday = past_holidays[-1]
        days_diff = (date - last_holiday).days
        return days_diff // 7
    
    # Apply to each unique date (faster than row by row)
    unique_dates = df["Date"].unique()
    
    date_to_next = {d: weeks_until_next_holiday(pd.Timestamp(d)) for d in unique_dates}
    date_to_last = {d: weeks_since_last_holiday(pd.Timestamp(d)) for d in unique_dates}
    
    df["weeks_to_next_holiday"]   = df["Date"].map(date_to_next)
    df["weeks_from_last_holiday"] = df["Date"].map(date_to_last)
    
    return df



def show_features(df):
  
    
    # Columns that are NOT features (original data columns)
    original_cols = ["Store", "Dept", "Date", "Weekly_Sales", "IsHoliday",
                     "Temperature", "Fuel_Price", "CPI", "Unemployment",
                     "Type", "Size", "MarkDown1", "MarkDown2",
                     "MarkDown3", "MarkDown4", "MarkDown5"]
    
    feature_cols = [c for c in df.columns if c not in original_cols]
    
    print(f"Total features created: {len(feature_cols)}")
    print()
    
    # Group by type
    lag_features      = [c for c in feature_cols if c.startswith("lag_")]
    rolling_features  = [c for c in feature_cols if c.startswith("rolling_")]
    calendar_features = [c for c in feature_cols if c not in lag_features + rolling_features]
    
    print(f"Lag features ({len(lag_features)}):")
    print(" ", lag_features)
    
    print(f"\nRolling features ({len(rolling_features)}):")
    print(" ", rolling_features)
    
    print(f"\nCalendar + Holiday features ({len(calendar_features)}):")
    print(" ", calendar_features)