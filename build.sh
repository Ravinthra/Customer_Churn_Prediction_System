#!/usr/bin/env bash
# Build script for Render deployment

set -o errexit  # Exit on error

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Downloading ML model from GitHub Releases ==="
# Download the model file if it doesn't exist
if [ ! -f "best_churn_model.pkl" ]; then
    echo "Downloading model from GitHub Releases..."
    curl -L -o best_churn_model.pkl \
        "https://github.com/Ravinthra/Customer_Churn_Prediction_System/releases/download/v1.0.0/best_churn_model.pkl"
    echo "Model downloaded successfully!"
else
    echo "Model file already exists, skipping download."
fi

echo "=== Collecting static files ==="
cd backend
python manage.py collectstatic --noinput

echo "=== Running migrations ==="
python manage.py migrate --noinput

echo "=== Build complete ==="
