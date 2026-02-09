from django.urls import path
from .views import predict_churn, home
from .health import health_check

urlpatterns = [
    path('', home, name='home'),
    path('predict/', predict_churn, name='predict'),
    path('health/', health_check, name='health'),
]
