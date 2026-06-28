import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_feature_importance(model, feature_names, model_name="Model", top_n=20):
    """
    Show which features the model thinks are most important.
    
    Why is this useful?
    - Tells you if your features are actually being used
    - Helps you understand what drives sales predictions
    - You can remove low-importance features to simplify the model
    
    Works for LightGBM, XGBoost, CatBoost, and Random Forest
    — they all have a feature_importances_ attribute.
    """
    
    # Get importance scores from the model
    importance_scores = model.feature_importances_
    
    # Put into a dataframe for easy sorting
    importance_df = pd.DataFrame({
        "feature":   feature_names,
        "importance": importance_scores
    })
    
    # Sort by importance, take top N
    importance_df = importance_df.sort_values("importance", ascending=False).head(top_n)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Blues(
        np.linspace(0.4, 0.9, len(importance_df))
    )[::-1]
    
    ax.barh(
        importance_df["feature"][::-1],
        importance_df["importance"][::-1],
        color=colors
    )
    
    ax.set_xlabel("Feature Importance Score")
    ax.set_title(f"{model_name} — Top {top_n} Most Important Features")
    
    plt.tight_layout()
    plt.savefig(
        f"reports/figures/feature_importance_{model_name.lower().replace(' ', '_')}.png",
        dpi=150, bbox_inches="tight"
    )
    plt.show()
    
    # Print top 10
    print(f"\nTop 10 features for {model_name}:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:30s} {row['importance']:.4f}")
    
    return importance_df