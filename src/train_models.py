"""
Model training module for customer churn prediction.

This module provides functionality to train multiple classification
models for comparison and selection.
"""
from typing import Dict, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def train_models(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    """
    Train multiple classification models on the training data.
    
    Args:
        X_train: Training feature matrix (scaled).
        y_train: Training target labels.
        
    Returns:
        Dictionary mapping model names to fitted model objects.
        
    Models trained:
        - Logistic Regression: Simple baseline with regularization
        - Decision Tree: Interpretable tree-based model
        - Random Forest: Ensemble model (typically best performer)
        - SVM: Support Vector Machine with probability calibration
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=6),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True)
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models
