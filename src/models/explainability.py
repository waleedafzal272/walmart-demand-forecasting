import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import shap


def explain_model(model, X_train, X_test, model_name="LightGBM"):
    """
    Use SHAP to explain WHY the model makes each prediction.
    
    What is SHAP?
    SHAP = SHapley Additive exPlanations.
    
    For every single prediction, SHAP tells you:
        "The model predicted $25,000 sales because:
            + lag_52 added $8,000  (same week last year was high)
            + is_december added $5,000  (December = peak month)
            + rolling_mean_4 added $2,000  (recent trend is up)
            - unemployment subtracted $500  (economy is slow)
            ... and so on for every feature"
    
    SHAP values = contribution of each feature to each prediction.
    Positive SHAP = feature pushed prediction UP
    Negative SHAP = feature pushed prediction DOWN
    
    We create 4 plots:
        1. Summary plot   — which features matter most overall
        2. Bar plot       — average importance of each feature
        3. Waterfall plot — explain one single prediction in detail
        4. Dependence plot — how one feature affects predictions
    
    Input:
        model      — trained LightGBM model
        X_train    — training features (used to build the explainer)
        X_test     — test features (we explain predictions on these)
        model_name — just for plot titles
    """
    
    print(f"Calculating SHAP values for {model_name}...")
    print("This may take 1-2 minutes...\n")
    
    # Create SHAP explainer
    # TreeExplainer works with LightGBM, XGBoost, CatBoost, Random Forest
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for test set
    # shap_values shape = (n_rows, n_features)
    # Each value = how much that feature contributed to that prediction
    shap_values = explainer.shap_values(X_test)
    
    print("SHAP values calculated!")
    print(f"Shape: {shap_values.shape} (rows x features)\n")
    
    return explainer, shap_values


def plot_shap_summary(shap_values, X_test, model_name="LightGBM"):
    """
    Summary plot — shows ALL features and ALL predictions.
    
    How to read it:
        - Each row = one feature
        - Each dot = one prediction
        - Red dots = high feature value
        - Blue dots = low feature value
        - Dots on the right = positive impact (pushed prediction up)
        - Dots on the left = negative impact (pushed prediction down)
    
    Example reading:
        lag_52 row: red dots on the right
        Meaning: when lag_52 is high (red), it pushes sales prediction UP
        This makes sense! High sales last year = high sales this year
    """
    print("Plotting SHAP Summary Plot...")
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_test,
        show=False,        # don't show immediately — we save first
        max_display=15     # show top 15 features
    )
    plt.title(f"{model_name} — SHAP Summary Plot", fontsize=13, pad=20)
    plt.tight_layout()
    plt.savefig(
        f"reports/figures/shap_summary_{model_name.lower()}.png",
        dpi=150, bbox_inches="tight"
    )
    plt.show()
    print("Saved!\n")


def plot_shap_bar(shap_values, X_test, model_name="LightGBM"):
    """
    Bar plot — average importance of each feature.
    
    Simpler than summary plot.
    Just shows: which features matter most on average?
    
    How to read it:
        Longer bar = more important feature overall
        This is the average of absolute SHAP values across all predictions
    """
    print("Plotting SHAP Bar Plot...")
    
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="bar",   # bar chart instead of dot plot
        show=False,
        max_display=15
    )
    plt.title(f"{model_name} — Feature Importance (SHAP)", fontsize=13, pad=20)
    plt.tight_layout()
    plt.savefig(
        f"reports/figures/shap_bar_{model_name.lower()}.png",
        dpi=150, bbox_inches="tight"
    )
    plt.show()
    print("Saved!\n")


def plot_shap_waterfall(explainer, shap_values, X_test, row_index=0, model_name="LightGBM"):
    """
    Waterfall plot — explain ONE single prediction in detail.
    
    This is the most powerful explanation plot.
    Pick any row (any prediction) and see EXACTLY why the
    model gave that number.
    
    How to read it:
        Starting from the bottom (base value = average prediction):
            Each bar shows how one feature CHANGED the prediction
            Red bar going right  = feature increased the prediction
            Blue bar going left  = feature decreased the prediction
            Final value at top   = the actual prediction made
    
    Example:
        Base value = $20,000 (average sales across all data)
        + lag_52 = +$6,000   (last year same week was high)
        + month_12 = +$3,000 (it's December)
        - unemployment = -$500 (high unemployment in this area)
        = Final prediction: $28,500
    
    Input:
        row_index — which prediction to explain (0 = first row of test set)
    """
    print(f"Plotting SHAP Waterfall for row {row_index}...")
    
    # Get the SHAP explanation for this specific row
    explanation = shap.Explanation(
        values=shap_values[row_index],
        base_values=explainer.expected_value,
        data=X_test.iloc[row_index],
        feature_names=X_test.columns.tolist()
    )
    
    plt.figure(figsize=(12, 7))
    shap.waterfall_plot(explanation, show=False, max_display=15)
    plt.title(f"{model_name} — Why did the model predict this? (Row {row_index})",
              fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(
        f"reports/figures/shap_waterfall_{model_name.lower()}.png",
        dpi=150, bbox_inches="tight"
    )
    plt.show()
    print("Saved!\n")


def plot_shap_dependence(shap_values, X_test, feature_name, model_name="LightGBM"):
    """
    Dependence plot — how does ONE feature affect predictions?
    
    Shows the relationship between:
        X axis = actual value of the feature
        Y axis = SHAP value (impact on prediction)
    
    Example for lag_52:
        X axis = sales from same week last year
        Y axis = how much that pushed the prediction up or down
        
        If you see a straight line going up-right:
        Higher last year sales → higher predicted sales this year
        Makes perfect sense!
    
    Input:
        feature_name — which feature to plot (e.g. "lag_52")
    """
    print(f"Plotting SHAP Dependence for '{feature_name}'...")
    
    feature_idx = X_test.columns.tolist().index(feature_name)
    
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature_idx,
        shap_values,
        X_test,
        show=False
    )
    plt.title(f"{model_name} — How '{feature_name}' affects predictions", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        f"reports/figures/shap_dependence_{feature_name}.png",
        dpi=150, bbox_inches="tight"
    )
    plt.show()
    print("Saved!\n")