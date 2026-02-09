#!/usr/bin/env bash
# Build script for Render deployment

set -o errexit  # Exit on error

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Model file check ==="
if [ -f "best_churn_model.pkl" ]; then
    echo "Model file found (Git LFS)"
else
    echo "ERROR: Model file not found!"
    exit 1
fi

echo "=== Collecting static files ==="
cd backend
python manage.py collectstatic --noinput

echo "=== Running migrations ==="
python manage.py migrate --noinput

echo "=== Build complete ==="
