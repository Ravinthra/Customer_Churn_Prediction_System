"""
Feature importance analysis for customer churn prediction.

This script analyzes and displays the relative importance of
features used in the trained Random Forest model.

Usage:
    python feature_importance.py
"""
from typing import List
import joblib
import pandas as pd

FEATURE_NAMES: List[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "PaymentMethod"
]


def analyze_feature_importance(model_path: str = "best_churn_model.pkl") -> pd.DataFrame:
    """
    Load the trained model and display feature importances.
    
    Args:
        model_path: Path to the saved model bundle.
        
    Returns:
        DataFrame with features sorted by importance.
    """
    bundle = joblib.load(model_path)
    model = bundle["model"]

    importances = model.feature_importances_

    df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    return df


if __name__ == "__main__":
    importance_df = analyze_feature_importance()
    print("\nFeature Importance Analysis")
    print("=" * 40)
    print(importance_df.to_string(index=False))
