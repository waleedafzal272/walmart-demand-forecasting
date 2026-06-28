import sys
sys.path.insert(0, ".")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore")

from src.models.ml_models import prepare_ml_data
from src.evaluation.metrics import evaluate_all, metrics_dataframe

# -------------------------------------------------------
# PAGE SETTINGS
# -------------------------------------------------------

st.set_page_config(
    page_title="Walmart Demand Forecasting",
    page_icon="📈",
    layout="wide"
)

# -------------------------------------------------------
# LOAD DATA AND MODELS
# -------------------------------------------------------

@st.cache_data
def load_data():
    """
    Load data once and cache it.
    
    Why cache?
    Without caching, the data reloads every time you click anything.
    With @st.cache_data, it loads once and stays in memory.
    Much faster!
    """
    df = pd.read_parquet("data/processed/train_features.parquet")
    return df


@st.cache_resource
def load_models():
    """
    Load trained models once and cache them.
    Returns a dictionary of available models.
    """
    models = {}
    model_files = {
        "LightGBM (Tuned)": "models/saved/lightgbm_tuned.pkl",
        "LightGBM":         "models/saved/lightgbm.pkl",
        "XGBoost":          "models/saved/xgboost.pkl",
        "CatBoost":         "models/saved/catboost.pkl",
        "Random Forest":    "models/saved/random_forest.pkl",
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    
    return models


def load_phase4_results():
    """Load classical model results from Phase 4."""
    path = "reports/metrics/phase4_results.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def load_phase5_results():
    """Load ML model results from Phase 5."""
    path = "reports/metrics/phase5_results.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


# -------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------

def get_store_dept_data(df, store, dept):
    """Filter dataframe to one store and department."""
    return df[
        (df["Store"] == store) &
        (df["Dept"]  == dept)
    ].sort_values("Date").reset_index(drop=True)


def split_data(df, test_weeks=12):
    """Split into train and test by time."""
    all_dates   = sorted(df["Date"].unique())
    cutoff      = all_dates[-test_weeks]
    train       = df[df["Date"] <  cutoff].copy()
    test        = df[df["Date"] >= cutoff].copy()
    return train, test


def make_predictions(model, test_df):
    """Get predictions from a model for the test period."""
    X_test, y_test = prepare_ml_data(test_df)
    preds = model.predict(X_test)
    preds = np.maximum(preds, 0)
    return preds, y_test.values, X_test


# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------

st.sidebar.title("📊 Controls")
st.sidebar.markdown("---")

# Load data
df     = load_data()
models = load_models()

# Store selector
all_stores = sorted(df["Store"].unique())
selected_store = st.sidebar.selectbox(
    "Select Store",
    all_stores,
    index=0
)

# Department selector
all_depts = sorted(
    df[df["Store"] == selected_store]["Dept"].unique()
)
selected_dept = st.sidebar.selectbox(
    "Select Department",
    all_depts,
    index=0
)

# Model selector
if models:
    selected_model_name = st.sidebar.selectbox(
        "Select Model",
        list(models.keys()),
        index=0
    )
else:
    selected_model_name = None
    st.sidebar.warning("No trained models found. Run Phase 5 first.")

# Test weeks selector
test_weeks = st.sidebar.slider(
    "Test Period (weeks)",
    min_value=4,
    max_value=24,
    value=12,
    step=4
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project:** Walmart Demand Forecasting")
st.sidebar.markdown("**Author:** Waleed")
st.sidebar.markdown("**Dataset:** Walmart Store Sales")

# -------------------------------------------------------
# MAIN PAGE
# -------------------------------------------------------

st.title("📈 Walmart Demand Forecasting Dashboard")
st.markdown("End-to-end time series forecasting — Classical & ML Models")
st.markdown("---")

# -------------------------------------------------------
# TAB LAYOUT
# -------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Sales Overview",
    "🤖 Model Predictions",
    "🏆 Model Comparison",
    "📋 Data Explorer"
])


# -------------------------------------------------------
# TAB 1 — SALES OVERVIEW
# -------------------------------------------------------

with tab1:
    st.header(f"Sales Overview — Store {selected_store}, Dept {selected_dept}")

    # Get data for selected store and dept
    store_dept_df = get_store_dept_data(df, selected_store, selected_dept)

    if len(store_dept_df) == 0:
        st.warning("No data found for this store and department.")
    else:
        # Key numbers at the top
        col1, col2, col3, col4 = st.columns(4)

        total_sales  = store_dept_df["Weekly_Sales"].sum()
        avg_sales    = store_dept_df["Weekly_Sales"].mean()
        max_sales    = store_dept_df["Weekly_Sales"].max()
        total_weeks  = len(store_dept_df)

        col1.metric("Total Sales",    f"${total_sales:,.0f}")
        col2.metric("Average / Week", f"${avg_sales:,.0f}")
        col3.metric("Best Week",      f"${max_sales:,.0f}")
        col4.metric("Total Weeks",    f"{total_weeks}")

        st.markdown("---")

        # Sales over time chart
        st.subheader("Weekly Sales Over Time")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(
            store_dept_df["Date"],
            store_dept_df["Weekly_Sales"],
            color="#4C72B0", linewidth=1.5
        )
        ax.fill_between(
            store_dept_df["Date"],
            store_dept_df["Weekly_Sales"],
            alpha=0.1, color="#4C72B0"
        )

        # Highlight holiday weeks in red
        holiday_weeks = store_dept_df[store_dept_df["IsHoliday"] == True]
        ax.scatter(
            holiday_weeks["Date"],
            holiday_weeks["Weekly_Sales"],
            color="red", s=30, zorder=5, label="Holiday Week"
        )

        ax.set_ylabel("Weekly Sales ($)")
        ax.set_title(f"Store {selected_store}, Dept {selected_dept}")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")

        # Monthly and seasonal patterns side by side
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Average Sales by Month")
            store_dept_df["Month"] = store_dept_df["Date"].dt.month
            monthly = store_dept_df.groupby("Month")["Weekly_Sales"].mean()
            month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"]

            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.bar(
                monthly.index,
                monthly.values / 1e3,
                color="#55A868", alpha=0.85
            )
            ax2.set_xticks(range(1, 13))
            ax2.set_xticklabels(month_names, rotation=45)
            ax2.set_ylabel("Avg Sales ($ Thousands)")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        with col_right:
            st.subheader("Holiday vs Normal Weeks")

            holiday_avg    = store_dept_df.groupby("IsHoliday")["Weekly_Sales"].mean()
            holiday_labels = ["Normal", "Holiday"]
            holiday_colors = ["#4C72B0", "#C44E52"]

            fig3, ax3 = plt.subplots(figsize=(5, 4))
            bars = ax3.bar(
                holiday_labels,
                holiday_avg.values / 1e3,
                color=holiday_colors,
                alpha=0.85, width=0.4
            )
            for bar, val in zip(bars, holiday_avg.values / 1e3):
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.3,
                    f"${val:.1f}K",
                    ha="center", fontsize=11, fontweight="bold"
                )
            ax3.set_ylabel("Avg Weekly Sales ($ Thousands)")
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()


# -------------------------------------------------------
# TAB 2 — MODEL PREDICTIONS
# -------------------------------------------------------

with tab2:
    st.header("Model Predictions vs Actual Sales")

    if selected_model_name is None:
        st.error("No models found. Please run Phase 5 first.")
    else:
        store_dept_df = get_store_dept_data(df, selected_store, selected_dept)

        if len(store_dept_df) < test_weeks + 20:
            st.warning("Not enough data for this store/department. Try another one.")
        else:
            # Split into train and test
            train_df, test_df = split_data(store_dept_df, test_weeks=test_weeks)

            # Get predictions
            model = models[selected_model_name]

            try:
                preds, actuals, X_test = make_predictions(model, test_df)

                # Key metrics
                metrics = evaluate_all(actuals, preds)

                st.subheader(f"Model: {selected_model_name}")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("RMSE",  f"{metrics['RMSE']:,.2f}")
                col2.metric("MAE",   f"{metrics['MAE']:,.2f}")
                col3.metric("MAPE",  f"{metrics['MAPE']:.2f}%")
                col4.metric("R²",    f"{metrics['R2']:.4f}")

                st.markdown("---")

                # Predictions chart
                st.subheader("Predictions vs Actual")

                fig4, ax4 = plt.subplots(figsize=(13, 5))

                # Plot training history in gray
                ax4.plot(
                    train_df["Date"],
                    train_df["Weekly_Sales"],
                    color="#CCCCCC", linewidth=1, label="Training Data"
                )

                # Plot actual test values
                ax4.plot(
                    test_df["Date"],
                    actuals,
                    color="black", linewidth=2, label="Actual Sales"
                )

                # Plot predictions
                ax4.plot(
                    test_df["Date"],
                    preds,
                    color="#C44E52", linewidth=2,
                    linestyle="--", label=f"{selected_model_name} Predictions"
                )

                # Shade the test region
                ax4.axvspan(
                    test_df["Date"].min(),
                    test_df["Date"].max(),
                    alpha=0.05, color="red", label="Test Period"
                )

                ax4.set_ylabel("Weekly Sales ($)")
                ax4.set_title(f"Store {selected_store}, Dept {selected_dept} — {selected_model_name}")
                ax4.legend()
                plt.tight_layout()
                st.pyplot(fig4)
                plt.close()

                st.markdown("---")

                # Error chart
                st.subheader("Prediction Error per Week")

                errors = actuals - preds

                fig5, ax5 = plt.subplots(figsize=(13, 3))
                colors = ["#C44E52" if e < 0 else "#55A868" for e in errors]
                ax5.bar(test_df["Date"], errors, color=colors, alpha=0.8)
                ax5.axhline(0, color="black", linewidth=1)
                ax5.set_ylabel("Error (Actual - Predicted)")
                ax5.set_title("Green = Underestimated | Red = Overestimated")
                plt.tight_layout()
                st.pyplot(fig5)
                plt.close()

                # Show prediction table
                st.markdown("---")
                st.subheader("Prediction Table")

                pred_table = pd.DataFrame({
                    "Date":      test_df["Date"].values,
                    "Actual":    actuals.round(2),
                    "Predicted": preds.round(2),
                    "Error":     (actuals - preds).round(2),
                    "Error %":   (np.abs(actuals - preds) /
                                  (np.abs(actuals) + 1e-8) * 100).round(2)
                })

                st.dataframe(pred_table, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info("Make sure Phase 3 feature engineering was completed first.")


# -------------------------------------------------------
# TAB 3 — MODEL COMPARISON
# -------------------------------------------------------

with tab3:
    st.header("All Models — Performance Comparison")

    # Load saved results
    phase4 = load_phase4_results()
    phase5 = load_phase5_results()

    # Combine all results
    all_results = {}
    all_results.update(phase4)
    all_results.update(phase5)

    if len(all_results) == 0:
        st.warning("No saved results found. Run Phase 4 and Phase 5 first.")
    else:
        # Build comparison table
        results_df = metrics_dataframe(all_results)

        st.subheader("Metrics Table (sorted by RMSE — lower is better)")
        st.dataframe(
            results_df.style.highlight_min(
                subset=["RMSE", "MAE", "MAPE", "SMAPE"],
                color="lightgreen"
            ).highlight_max(
                subset=["R2"],
                color="lightgreen"
            ),
            use_container_width=True
        )

        st.markdown("---")

        # Bar chart comparison
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("RMSE Comparison (lower = better)")
            fig6, ax6 = plt.subplots(figsize=(7, 5))
            rmse_vals = results_df["RMSE"].sort_values()
            colors = ["#4C72B0" if i == 0 else "#AAAAAA"
                      for i in range(len(rmse_vals))]
            ax6.barh(rmse_vals.index, rmse_vals.values, color=colors)
            ax6.set_xlabel("RMSE")
            ax6.set_title("Lower is Better")
            ax6.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig6)
            plt.close()

        with col_right:
            st.subheader("R² Comparison (higher = better)")
            fig7, ax7 = plt.subplots(figsize=(7, 5))
            r2_vals = results_df["R2"].sort_values(ascending=False)
            colors = ["#55A868" if i == 0 else "#AAAAAA"
                      for i in range(len(r2_vals))]
            ax7.barh(r2_vals.index, r2_vals.values, color=colors)
            ax7.set_xlabel("R² Score")
            ax7.set_title("Higher is Better")
            ax7.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig7)
            plt.close()

        st.markdown("---")

        # Winner announcement
        best_model = results_df.index[0]
        best_rmse  = results_df["RMSE"].iloc[0]
        best_mape  = results_df["MAPE"].iloc[0]
        best_r2    = results_df["R2"].iloc[0]

        st.success(
            f"🏆 Best Model: **{best_model}** "
            f"— RMSE: {best_rmse:.2f} | "
            f"MAPE: {best_mape:.2f}% | "
            f"R²: {best_r2:.4f}"
        )


# -------------------------------------------------------
# TAB 4 — DATA EXPLORER
# -------------------------------------------------------

with tab4:
    st.header("Data Explorer")

    store_dept_df = get_store_dept_data(df, selected_store, selected_dept)

    st.subheader(f"Store {selected_store}, Dept {selected_dept} — Raw Data")
    st.dataframe(
        store_dept_df[[
            "Date", "Weekly_Sales", "IsHoliday",
            "Temperature", "Fuel_Price", "CPI", "Unemployment"
        ]].round(2),
        use_container_width=True
    )

    st.markdown("---")

    # Dataset-wide stats
    st.subheader("Overall Dataset Statistics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Stores",      df["Store"].nunique())
    col2.metric("Total Departments", df["Dept"].nunique())
    col3.metric("Total Rows",        f"{len(df):,}")

    st.markdown("---")

    # Top 10 stores by sales
    st.subheader("Top 10 Stores by Total Sales")

    top_stores = (
        df.groupby("Store")["Weekly_Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    top_stores.columns = ["Store", "Total Sales ($)"]
    top_stores["Total Sales ($)"] = top_stores["Total Sales ($)"].round(0)

    st.dataframe(top_stores, use_container_width=True)