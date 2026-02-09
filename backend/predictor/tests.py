"""
Tests for the churn prediction application.
"""
from unittest.mock import patch, MagicMock
from django.test import TestCase, Client
from django.urls import reverse


class HomePageTests(TestCase):
    """Tests for the home page."""
    
    def setUp(self):
        self.client = Client()
    
    def test_home_page_loads(self):
        """Test that the home page returns 200 and uses correct template."""
        response = self.client.get(reverse('home'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'index.html')
    
    def test_home_page_contains_form(self):
        """Test that the home page contains the prediction form."""
        response = self.client.get(reverse('home'))
        self.assertContains(response, 'churnForm')
        self.assertContains(response, 'tenure')


class HealthCheckTests(TestCase):
    """Tests for the health check endpoint."""
    
    def setUp(self):
        self.client = Client()
    
    def test_health_endpoint_returns_json(self):
        """Test that health endpoint returns JSON with status."""
        response = self.client.get(reverse('health'))
        self.assertIn(response.status_code, [200, 503])  # Healthy or degraded
        data = response.json()
        self.assertIn('status', data)
        self.assertIn('model_loaded', data)
        self.assertIn('timestamp', data)


class PredictAPITests(TestCase):
    """Tests for the prediction API endpoint."""
    
    def setUp(self):
        self.client = Client()
        self.valid_payload = {
            "tenure": 12,
            "MonthlyCharges": 75.50,
            "TotalCharges": 900.00,
            "Contract": 1,
            "PaymentMethod": 2
        }
    
    @patch('predictor.model_loader.MODEL_LOADED', True)
    @patch('predictor.model_loader.get_model')
    @patch('predictor.model_loader.get_scaler')
    def test_predict_valid_input(self, mock_scaler, mock_model):
        """Test prediction with valid input returns expected response."""
        # Setup mock model
        mock_model_instance = MagicMock()
        mock_model_instance.predict.return_value = [0]
        mock_model_instance.predict_proba.return_value = [[0.7, 0.3]]
        mock_model_instance.feature_importances_ = [0.3, 0.25, 0.2, 0.15, 0.1]
        mock_model.return_value = mock_model_instance
        mock_scaler.return_value = None
        
        response = self.client.post(
            reverse('predict'),
            data=self.valid_payload,
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('churn_prediction', data)
        self.assertIn('churn_probability', data)
        self.assertIn('top_factors', data)
    
    def test_predict_missing_fields(self):
        """Test that missing required fields return 400 error."""
        incomplete_payload = {"tenure": 12}
        
        response = self.client.post(
            reverse('predict'),
            data=incomplete_payload,
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
    
    def test_predict_invalid_tenure(self):
        """Test that negative tenure returns validation error."""
        invalid_payload = self.valid_payload.copy()
        invalid_payload['tenure'] = -5
        
        response = self.client.post(
            reverse('predict'),
            data=invalid_payload,
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
    
    def test_predict_invalid_contract(self):
        """Test that invalid contract value returns validation error."""
        invalid_payload = self.valid_payload.copy()
        invalid_payload['Contract'] = 99
        
        response = self.client.post(
            reverse('predict'),
            data=invalid_payload,
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
    
    @patch('predictor.model_loader.MODEL_LOADED', False)
    def test_predict_model_not_loaded(self):
        """Test that prediction returns 503 when model is not loaded."""
        response = self.client.post(
            reverse('predict'),
            data=self.valid_payload,
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 503)


class SerializerValidationTests(TestCase):
    """Tests for serializer validation logic."""
    
    def test_total_charges_less_than_monthly_for_existing_customer(self):
        """Test that TotalCharges < MonthlyCharges fails for tenure > 0."""
        from predictor.serializers import ChurnInputSerializer
        
        invalid_data = {
            "tenure": 12,
            "MonthlyCharges": 100.00,
            "TotalCharges": 50.00,  # Less than monthly
            "Contract": 1,
            "PaymentMethod": 2
        }
        
        serializer = ChurnInputSerializer(data=invalid_data)
        self.assertFalse(serializer.is_valid())
        self.assertIn('TotalCharges', str(serializer.errors))
