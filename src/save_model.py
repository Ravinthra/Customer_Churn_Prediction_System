"""
Model serialization module for customer churn prediction.

This module handles saving trained models and their associated
preprocessing artifacts for later use during inference.
"""
from typing import Any, Optional
import joblib


def save_best_model(
    model: Any,
    scaler: Optional[Any],
    output_path: str = "best_churn_model.pkl"
) -> None:
    """
    Save the trained model and scaler as a bundle.
    
    Args:
        model: Trained scikit-learn model object.
        scaler: Fitted StandardScaler (or None if not used).
        output_path: Path to save the model bundle.
        
    Note:
        The model and scaler are saved together to ensure
        consistent preprocessing during inference. Always use
        the same scaler that was fitted on training data.
    """
    bundle = {
        "model": model,
        "scaler": scaler
    }
    joblib.dump(bundle, output_path)
    print(f"Model saved to: {output_path}")
