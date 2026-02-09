"""
Model loader module - handles ML model loading with proper error handling.
This module is imported once at startup and caches the model.
"""
import os
import logging

import joblib

logger = logging.getLogger("predictor")

# =============================================================================
# MODEL LOADING CONFIGURATION
# =============================================================================

# Get the project root (Customer_Churn_Prediction_System folder)
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# Get model path from environment, default to project root
_model_path = os.getenv("MODEL_PATH", "best_churn_model.pkl")

# Handle relative paths by resolving from project root
if _model_path.startswith("./"):
    MODEL_PATH = os.path.join(PROJECT_ROOT, _model_path[2:])
elif not os.path.isabs(_model_path):
    MODEL_PATH = os.path.join(PROJECT_ROOT, _model_path)
else:
    MODEL_PATH = _model_path

# =============================================================================
# LOAD MODEL WITH ERROR HANDLING
# =============================================================================

MODEL_LOADED = False
MODEL_ERROR = None
model = None
scaler = None

FEATURE_NAMES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "PaymentMethod"
]


def load_model():
    """
    Load the ML model and scaler from disk.
    Returns True if successful, False otherwise.
    """
    global MODEL_LOADED, MODEL_ERROR, model, scaler
    
    try:
        logger.info(f"Loading model from: {MODEL_PATH}")
        
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        
        bundle = joblib.load(MODEL_PATH)
        
        # Support both dict format and raw model
        if isinstance(bundle, dict):
            model = bundle.get("model")
            scaler = bundle.get("scaler")
            if model is None:
                raise ValueError("Model bundle missing 'model' key")
        else:
            model = bundle
            scaler = None
        
        MODEL_LOADED = True
        MODEL_ERROR = None
        logger.info("Model loaded successfully")
        return True
        
    except Exception as e:
        MODEL_LOADED = False
        MODEL_ERROR = str(e)
        logger.error(f"Failed to load model: {e}", exc_info=True)
        return False


def get_model():
    """Get the loaded model. Returns None if not loaded."""
    return model


def get_scaler():
    """Get the loaded scaler. Returns None if not available."""
    return scaler


# Auto-load model on module import
load_model()
