"""
Data preprocessing module for customer churn prediction.

This module handles data loading, cleaning, feature encoding,
and train/test splitting for the churn prediction pipeline.
"""
from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Selected features based on domain knowledge and feature importance analysis
# These features have the highest predictive power for customer churn
IMPORTANT_FEATURES: List[str] = [
    "tenure",          # Customer loyalty indicator
    "MonthlyCharges",  # Price sensitivity indicator
    "TotalCharges",    # Customer lifetime value proxy
    "Contract",        # Commitment level (encoded: 0=Month-to-month, 1=One year, 2=Two year)
    "PaymentMethod"    # Payment behavior (encoded: 0-3 for different methods)
]


def preprocess_data(csv_path: str) -> Tuple[Tuple, StandardScaler]:
    """
    Load and preprocess customer churn data.
    
    Args:
        csv_path: Path to the CSV file containing raw customer data.
        
    Returns:
        A tuple containing:
            - (X_train, X_test, y_train, y_test): Train/test split data
            - scaler: Fitted StandardScaler for feature normalization
            
    Note:
        The scaler must be saved with the model for consistent
        preprocessing during inference.
    """
    df = pd.read_csv(csv_path)

    # Handle TotalCharges conversion (some values are whitespace)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)

    # Encode target variable
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Encode categorical features
    for col in ["Contract", "PaymentMethod"]:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Select only important features
    X = df[IMPORTANT_FEATURES]
    y = df["Churn"]

    # Normalize features for better model performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42), scaler
