# 📈 Walmart Demand Forecasting

An end-to-end professional demand forecasting project built on the
Walmart Store Sales dataset from Kaggle.

## 🎯 Project Overview

This project covers the complete data science lifecycle:
- Data loading, cleaning, and exploratory analysis
- Feature engineering (lag, rolling, calendar, holiday features)
- Classical time series models (ARIMA, SARIMA, Prophet, Holt-Winters)
- Machine learning models (LightGBM, XGBoost, CatBoost, Random Forest)
- Hyperparameter tuning with Optuna
- Model explainability with SHAP
- Interactive Streamlit dashboard

## 📊 Dataset

**Walmart Store Sales Forecasting** from Kaggle
- 421,570 weekly sales records
- 45 stores, 99 departments
- Date range: 2010 to 2012
- External features: Temperature, Fuel Price, CPI, Unemployment, Promotions

## 🗂️ Project Structure

demand_forecasting/

├── configs/          # Project settings

├── data/             # Raw and processed data

├── src/

│   ├── data/         # Loading and cleaning

│   ├── features/     # Feature engineering

│   ├── models/       # All models

│   └── evaluation/   # Metrics

├── notebooks/        # Jupyter notebooks per phase

├── reports/          # Plots and metric results

├── models/saved/     # Trained model files

└── dashboard.py      # Streamlit dashboard
## 🤖 Models Used

| Type | Models |
|------|--------|
| Baseline | Naive, Moving Average, Seasonal Naive |
| Classical | ARIMA, SARIMA, Holt-Winters, Prophet |
| ML | LightGBM, XGBoost, CatBoost, Random Forest |
| Ensemble | Weighted average of all ML models |

## 📈 Results

All models evaluated using walk-forward cross validation.
ML models significantly outperform classical models.
LightGBM (tuned) achieves the best results overall.

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/YourUsername/walmart-demand-forecasting.git
cd walmart-demand-forecasting

# 2. Create conda environment
conda create -n forecasting python=3.11 -y
conda activate forecasting

# 3. Install libraries
pip install -r requirements.txt

# 4. Download dataset from Kaggle
# Place train.csv, stores.csv, features.csv in data/raw/

# 5. Run notebooks in order
# Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6

# 6. Launch dashboard
streamlit run dashboard.py
```

## 🛠️ Tech Stack

Python · Pandas · Statsmodels · Prophet · LightGBM · XGBoost · 
CatBoost · Scikit-learn · Optuna · SHAP · Streamlit · Matplotlib

## 👨‍💻 Author

**Waleed** — Software Engineering Student, University of Gujrat
