"""
Model evaluation module for customer churn prediction.

This module provides comprehensive model evaluation using
multiple classification metrics.
"""
from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def evaluate(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate multiple models on test data.
    
    Args:
        models: Dictionary of model name -> fitted model object.
        X_test: Test feature matrix.
        y_test: Test target labels.
        
    Returns:
        Dictionary mapping model names to their evaluation metrics:
            - Accuracy: Overall correctness
            - Precision: True positive rate among predicted positives
            - Recall: True positive rate among actual positives
            - F1: Harmonic mean of precision and recall
            - ROC_AUC: Area under the ROC curve
            
    Note:
        ROC-AUC is often the best metric for churn prediction as it
        handles class imbalance better than accuracy.
    """
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1": round(f1_score(y_test, y_pred), 4),
            "ROC_AUC": round(roc_auc_score(y_test, y_prob), 4)
        }

    return results
