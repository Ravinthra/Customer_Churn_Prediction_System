"""
Health check views for monitoring and load balancer probes.
"""
import logging
from datetime import datetime

from django.http import JsonResponse

logger = logging.getLogger("predictor")


def health_check(request):
    """
    Health check endpoint for load balancers and monitoring.
    Returns the current status of the application and model.
    """
    from . import model_loader
    
    status_code = 200 if model_loader.MODEL_LOADED else 503
    
    response_data = {
        "status": "healthy" if model_loader.MODEL_LOADED else "degraded",
        "model_loaded": model_loader.MODEL_LOADED,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    
    if not model_loader.MODEL_LOADED and model_loader.MODEL_ERROR:
        response_data["model_error"] = str(model_loader.MODEL_ERROR)
    
    return JsonResponse(response_data, status=status_code)
