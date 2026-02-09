"""
Views for the churn prediction API and frontend.
"""
import logging
import uuid

import pandas as pd
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from .serializers import ChurnInputSerializer
from . import model_loader

# =============================================================================
# LOGGING SETUP
# =============================================================================

logger = logging.getLogger("predictor")


# =============================================================================
# FRONTEND UI
# =============================================================================

def home(request):
    """Render the main prediction form."""
    return render(request, "index.html")


# =============================================================================
# PREDICTION API
# =============================================================================

@csrf_exempt
@api_view(["POST"])
def predict_churn(request):
    """
    Predict customer churn based on input features.
    
    Request body:
        - tenure: int (months)
        - MonthlyCharges: float
        - TotalCharges: float
        - Contract: int (0=Month-to-month, 1=One year, 2=Two year)
        - PaymentMethod: int (0=Electronic check, 1=Mailed check, 2=Bank transfer, 3=Credit card)
    
    Returns:
        - churn_prediction: "Yes" or "No"
        - churn_probability: float (0-100)
        - top_factors: list of top 3 influencing features
    """
    # Generate request ID for tracking
    request_id = str(uuid.uuid4())[:8]
    
    # Check if model is loaded
    if not model_loader.MODEL_LOADED:
        logger.error(f"[{request_id}] Prediction failed: Model not loaded")
        return Response(
            {"error": "Service temporarily unavailable. Model not loaded."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )
    
    # Validate input
    serializer = ChurnInputSerializer(data=request.data)
    if not serializer.is_valid():
        logger.warning(f"[{request_id}] Validation failed: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    data = serializer.validated_data
    logger.info(f"[{request_id}] Prediction request | Input: {data}")

    try:
        # Build DataFrame to preserve feature names
        features = pd.DataFrame([{
            "tenure": data["tenure"],
            "MonthlyCharges": data["MonthlyCharges"],
            "TotalCharges": data["TotalCharges"],
            "Contract": data["Contract"],
            "PaymentMethod": data["PaymentMethod"]
        }])

        # Get model and scaler
        model = model_loader.get_model()
        scaler = model_loader.get_scaler()

        # Scale features if scaler exists
        if scaler is not None:
            features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        # Get feature importance (RandomForest)
        importances = model.feature_importances_
        top_features = sorted(
            zip(model_loader.FEATURE_NAMES, importances),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        top_factors = [feature for feature, _ in top_features]

        logger.info(
            f"[{request_id}] Prediction result | Churn: {prediction} | "
            f"Probability: {round(probability * 100, 2)}%"
        )

        return Response({
            "churn_prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": round(probability * 100, 2),
            "top_factors": top_factors,
            "request_id": request_id
        }, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"[{request_id}] Prediction failed: {str(e)}", exc_info=True)
        return Response(
            {"error": "Internal server error", "request_id": request_id},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
