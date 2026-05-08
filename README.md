# 📊 Customer Churn Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white&style=for-the-badge)
![Django](https://img.shields.io/badge/Django-4.2+-092E20?logo=django&logoColor=white&style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white&style=for-the-badge)
![Render](https://img.shields.io/badge/Deployed_on-Render-46E3B7?logo=render&logoColor=white&style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**An end-to-end Machine Learning system for predicting telecom customer churn —  
from raw CSV data to a live, production-deployed REST API with a Web UI.**

🌐 **Live Demo:** [customer-churn-prediction-ezz6.onrender.com](https://customer-churn-prediction-ezz6.onrender.com)

</div>

---

> **Demo Screenshot**
>
> ![Demo](docs/demo.png)

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [ML Pipeline](#-ml-pipeline-srcsrc)
- [Model Performance](#-model-performance)
- [Features Used](#-features-engineered-for-prediction)
- [Django REST API](#-django-rest-api)
- [API Reference](#-api-reference)
- [Production Configuration](#-production-configuration)
- [Test Suite](#-test-suite)
- [Quick Start](#-quick-start)
- [Deployment](#-deployment)
- [Environment Variables](#-environment-variables)
- [Limitations](#-limitations--assumptions)
- [Resume Summary](#-resume-ready-summary)

---

## 📖 Project Overview

Customer churn costs telecom companies millions annually. Retaining an existing customer is 5–25× cheaper than acquiring a new one. This system applies classical Machine Learning to the **IBM Telco Customer Churn dataset** (~7,000 customers) to predict churn probability before it happens — giving retention teams time to act.

### What makes this project production-grade:

| Dimension | What's implemented |
|-----------|-------------------|
| **ML Training** | 4 models trained & compared with 5 metrics |
| **Best Model** | Random Forest (ROC-AUC 0.8576) serialized as model+scaler bundle |
| **Inference API** | Django REST Framework — validated, logged, request-traced |
| **Web UI** | Single-page form with real-time prediction & feature importance |
| **Observability** | Rotating file logs + request IDs + `/health/` endpoint |
| **Security** | HSTS, SSL redirect, CSRF, XSS headers, CORS, secure cookies |
| **Deployment** | Gunicorn + WhiteNoise on Render.com via `render.yaml` + Git LFS |
| **Testing** | 9-test suite (unit + integration + mock-based) |

---

## 🏗️ Architecture

```
Customer_Churn_Prediction_System/
│
├── data/                           # Dataset (IBM Telco)
│   ├── telco_churn.csv             # Raw dataset (~7,000 rows, 21 columns)
│   └── cleaned_telco_churn.csv     # Preprocessed version (EDA artifact)
│
├── notebooks/
│   └── churn_eda.ipynb             # Full EDA: distributions, correlations, churn analysis
│
├── src/                            # Offline ML Training Pipeline
│   ├── __init__.py
│   ├── main.py                     # 4-step orchestrator (preprocess → train → evaluate → save)
│   ├── preprocess.py               # Cleaning, encoding, scaling, train/test split
│   ├── train_models.py             # 4-model training: LR, DT, RF, SVM
│   ├── evaluate_models.py          # 5-metric evaluation: Accuracy/Precision/Recall/F1/ROC-AUC
│   ├── feature_importance.py       # Post-training Random Forest feature ranking
│   ├── save_model.py               # Saves {model, scaler} bundle with joblib
│   └── clustering.py              # KMeans customer segmentation (k=3, exploratory)
│
├── backend/                        # Django Application
│   ├── manage.py
│   ├── gunicorn.conf.py            # Workers, timeout, keepalive, preload_app
│   ├── backend/                    # Django project config
│   │   ├── settings.py             # Production-safe: env-driven, HSTS, CORS, logging
│   │   ├── urls.py                 # Root URL dispatcher
│   │   ├── wsgi.py
│   │   └── asgi.py
│   └── predictor/                  # Django app: API + UI
│       ├── views.py                # predict_churn() + home()
│       ├── serializers.py          # ChurnInputSerializer — field + cross-field validation
│       ├── model_loader.py         # Singleton model cache, env-path resolution
│       ├── health.py               # /health/ — returns model_loaded + timestamp
│       ├── urls.py                 # 3 routes: /, /predict/, /health/
│       ├── tests.py                # 9 tests across 4 test classes
│       ├── templates/              # HTML templates (index.html)
│       └── static/                 # CSS + JavaScript
│
├── docs/
│   └── demo.png                    # UI screenshot for README
│
├── best_churn_model.pkl            # Trained model bundle (Git LFS, ~15 MB)
├── best_churn_model.zip            # Compressed archive
├── render.yaml                     # Render.com blueprint (build + env + health check)
├── build.sh                        # CI build: pip install → model check → collectstatic → migrate
├── requirements.txt                # Pinned dependencies
├── .env.example                    # Environment variable template
└── .gitignore / .gitattributes     # LFS tracking for .pkl
```

---

## 🔬 ML Pipeline (`src/`)

The training pipeline is a clean 4-step orchestration run via `python src/main.py`:

### Step 1 — Preprocessing (`preprocess.py`)

- **Input**: Raw `telco_churn.csv` (21 columns, mixed types)
- **TotalCharges fix**: Coerces whitespace entries to `NaN`, then drops rows — a common real-world data quality issue
- **Label encoding**: `Contract` (3 values → 0/1/2) and `PaymentMethod` (4 values → 0–3) with `LabelEncoder`
- **Target encoding**: `Churn` mapped `{"Yes": 1, "No": 0}`
- **Feature selection**: 5 domain-selected features with highest predictive power (see below)
- **Scaling**: `StandardScaler` fitted on train set only — scaler saved alongside model to prevent leakage
- **Split**: 80/20 train/test, `random_state=42` for reproducibility

### Step 2 — Model Training (`train_models.py`)

All 4 models trained on the same scaled training set:

| Model | Key Hyperparameter |
|-------|--------------------|
| Logistic Regression | `max_iter=1000` |
| Decision Tree | `max_depth=6` (prevents overfitting) |
| **Random Forest** | `n_estimators=100`, `random_state=42` |
| SVM | `probability=True` (enables `predict_proba` for ROC-AUC) |

### Step 3 — Evaluation (`evaluate_models.py`)

Each model evaluated on 5 metrics:

```python
accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
```

> **Why ROC-AUC?** The Telco dataset has class imbalance (~27% churn). ROC-AUC measures discrimination across all classification thresholds, making it more reliable than accuracy for imbalanced targets.

### Step 4 — Model Serialization (`save_model.py`)

Model and scaler saved together as a dict bundle:

```python
bundle = {"model": random_forest_model, "scaler": fitted_standard_scaler}
joblib.dump(bundle, "best_churn_model.pkl")
```

This **guarantees the exact same preprocessing transform** is applied at inference time — a critical production correctness requirement.

### Bonus — Customer Segmentation (`clustering.py`)

Exploratory KMeans clustering (k=3) on `tenure`, `MonthlyCharges`, `TotalCharges` to identify distinct customer risk segments for targeted retention campaigns.

---

## 📈 Model Performance

Evaluated on a held-out 20% test set (~1,400 customers):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|:--------:|:---------:|:------:|:--:|:-------:|
| Logistic Regression | 80.21% | 65.12% | 54.32% | 59.21% | 0.8421 |
| Decision Tree | 78.54% | 61.23% | 49.82% | 54.93% | 0.7632 |
| **Random Forest ✅** | **81.34%** | **67.45%** | **56.21%** | **61.32%** | **0.8576** |
| SVM | 79.87% | 63.54% | 52.14% | 57.28% | 0.8312 |

> **Random Forest** selected as the deployment model — highest ROC-AUC (0.8576) and F1 score, demonstrating strong churn discrimination even under class imbalance.

---

## 🔍 Features Engineered for Prediction

Domain knowledge was used to select the 5 most predictive features, reducing noise and inference latency:

| Feature | Type | Encoding | Business Meaning |
|---------|------|----------|-----------------|
| `tenure` | Integer | Raw (months 0–72) | Customer loyalty — longer tenure → lower churn risk |
| `MonthlyCharges` | Float | Scaled | Price sensitivity — high charges correlate with churn |
| `TotalCharges` | Float | Scaled | Lifetime value — often correlated with tenure |
| `Contract` | Categorical | LabelEncoded (0/1/2) | Month-to-month = highest churn risk |
| `PaymentMethod` | Categorical | LabelEncoded (0–3) | Electronic check → higher churn than auto-pay methods |

**Feature Importance from trained Random Forest** (via `feature_importance.py`):

```
tenure          → highest importance (loyalty indicator)
MonthlyCharges  → second (price-driven churn)
TotalCharges    → third (correlated CLV proxy)
Contract        → fourth (commitment signal)
PaymentMethod   → fifth (engagement proxy)
```

---

## 🌐 Django REST API

The inference layer is a Django application with Django REST Framework. Key design decisions:

### `model_loader.py` — Singleton Model Cache

```python
# Loaded once at module import time, cached in memory
bundle = joblib.load(MODEL_PATH)          # {model: ..., scaler: ...}
MODEL_LOADED = True
```

- **Path resolution**: `MODEL_PATH` is resolved from env variable, supports relative (`./`) and absolute paths
- **Startup validation**: Fails fast with `FileNotFoundError` if model is missing — surfaced via `/health/`
- **Dual format support**: Handles both dict-bundle format and raw model object (backwards compatibility)

### `serializers.py` — Input Validation Layer

`ChurnInputSerializer` enforces:
- **Field-level**: `tenure ∈ [0, 1000]`, `Contract ∈ [0, 2]`, `PaymentMethod ∈ [0, 3]`
- **Cross-field**: If `tenure > 0` then `TotalCharges ≥ MonthlyCharges` (business logic sanity check)
- Returns DRF 400 responses with field-level error messages

### `views.py` — Prediction Endpoint

```
Request → UUID request_id → Model-loaded check → Serializer validation
       → DataFrame construction → Scaler transform → predict() + predict_proba()
       → Feature importance extraction → Structured JSON response
```

Every request gets a unique 8-char `request_id` for end-to-end log correlation.

---

## 🔗 API Reference

### `GET /health/`

Returns application and model health for load balancer probes.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-02-09T06:17:47Z"
}
```

Returns `200` when healthy, `503` with `model_error` field when degraded.

---

### `POST /predict/`

Predicts customer churn from 5 input features.

**Request:**
```http
POST /predict/
Content-Type: application/json
```

```json
{
  "tenure": 3,
  "MonthlyCharges": 85.50,
  "TotalCharges": 256.50,
  "Contract": 0,
  "PaymentMethod": 0
}
```

**Field reference:**

| Field | Type | Range | Contract values | PaymentMethod values |
|-------|------|-------|----------------|---------------------|
| `tenure` | int | 0–1000 | — | — |
| `MonthlyCharges` | float | 0–1,000,000 | — | — |
| `TotalCharges` | float | 0–100,000,000 | — | — |
| `Contract` | int | 0–2 | 0=Month-to-month, 1=One year, 2=Two year | — |
| `PaymentMethod` | int | 0–3 | — | 0=Electronic check, 1=Mailed check, 2=Bank transfer, 3=Credit card |

**Response `200 OK`:**
```json
{
  "churn_prediction": "Yes",
  "churn_probability": 87.42,
  "top_factors": ["MonthlyCharges", "tenure", "Contract"],
  "request_id": "a1b2c3d4"
}
```

**Error responses:**

| Status | Cause |
|--------|-------|
| `400 Bad Request` | Missing fields, out-of-range values, or failed cross-field validation |
| `503 Service Unavailable` | Model not loaded on startup |
| `500 Internal Server Error` | Unexpected inference failure |

---

## ⚙️ Production Configuration

### Security (`settings.py`)

Enabled automatically when `DEBUG=False`:

```python
SECURE_SSL_REDIRECT = True          # Force HTTPS
SESSION_COOKIE_SECURE = True        # HTTPS-only cookies
CSRF_COOKIE_SECURE = True
SECURE_BROWSER_XSS_FILTER = True    # XSS protection header
SECURE_CONTENT_TYPE_NOSNIFF = True  # MIME sniffing prevention
X_FRAME_OPTIONS = "DENY"            # Clickjacking protection
SECURE_HSTS_SECONDS = 31536000      # HSTS: 1 year
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
```

### Logging (`settings.py`)

Dual-handler logging: rotating file + console.

```
[2026-02-09 06:17:47] INFO predictor views:73 - [a1b2c3d4] Prediction request | Input: {...}
[2026-02-09 06:17:47] INFO predictor views:107 - [a1b2c3d4] Prediction result | Churn: 1 | Probability: 87.42%
```

- **RotatingFileHandler**: 10 MB per file, 5 backup files → 50 MB max log retention
- **Request IDs**: 8-char UUID prefix on every log line → supports distributed trace correlation

### Gunicorn (`gunicorn.conf.py`)

```python
workers = cpu_count * 2 + 1       # Formula: optimal for sync workers
max_requests = 1000                # Worker recycling prevents memory leaks
max_requests_jitter = 50           # Staggered recycling avoids thundering herd
timeout = 30                       # Hard limit on request processing
preload_app = True                 # Model loaded once, shared across workers
```

### CORS (`settings.py`)

- Dev (`DEBUG=True`): `CORS_ALLOW_ALL_ORIGINS = True`
- Prod: Whitelist via `CORS_ALLOWED_ORIGINS` env variable (comma-separated)
- Credentials: `CORS_ALLOW_CREDENTIALS = True`

### Static Files

WhiteNoise serves compressed, fingerprinted static files directly from Gunicorn — no nginx/CDN required for this scale.

---

## 🧪 Test Suite

```bash
cd backend
python manage.py test predictor -v 2
```

**9 tests across 4 test classes:**

| Test Class | Tests | What's covered |
|------------|-------|---------------|
| `HomePageTests` | 2 | Home page status 200, template used, form elements present |
| `HealthCheckTests` | 1 | `/health/` returns JSON with `status`, `model_loaded`, `timestamp` |
| `PredictAPITests` | 5 | Valid prediction (mocked model), missing fields → 400, invalid tenure → 400, invalid Contract → 400, model-not-loaded → 503 |
| `SerializerValidationTests` | 1 | Cross-field: `TotalCharges < MonthlyCharges` with `tenure > 0` → validation error |

The prediction tests use `unittest.mock.patch` to mock `model_loader` state, enabling isolated testing without loading the actual `.pkl` file.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Git (with Git LFS for the model file)

```bash
# Clone the repository
git clone https://github.com/Ravinthra/Customer_Churn_Prediction_System.git
cd Customer_Churn_Prediction_System

# Pull model file via Git LFS
git lfs pull

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Default .env settings work for local development
```

### Train the Model (first time only)

```bash
# From project root
python src/main.py
```

Output:
```
==================================================
Customer Churn Prediction - Training Pipeline
==================================================

[1/4] Loading and preprocessing data...
      Training samples: 5634
      Test samples: 1409

[2/4] Training models...
      Trained 4 models

[3/4] Evaluating models...
      ...metrics comparison...

[4/4] Saving best model (Random Forest)...
Model saved to: best_churn_model.pkl
```

### Run the Server

```bash
cd backend
python manage.py runserver
# Open http://127.0.0.1:8000
```

---

## 🚢 Deployment

### Local Production Mode

```bash
cd backend
python manage.py collectstatic --noinput
gunicorn backend.wsgi:application -c gunicorn.conf.py
```

### Render.com (Cloud)

Deployment is fully automated via `render.yaml` (Render Blueprint):

```yaml
services:
  - type: web
    buildCommand: "./build.sh"
    startCommand: "cd backend && gunicorn backend.wsgi:application --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --preload"
    healthCheckPath: /health/
```

**Build steps** (`build.sh`):
1. `pip install -r requirements.txt`
2. Verify model file exists (Git LFS artifact)
3. `python manage.py collectstatic --noinput`
4. `python manage.py migrate --noinput`

> **Git LFS**: The `best_churn_model.pkl` (~15 MB) is tracked via Git LFS (`.gitattributes`). Render pulls it automatically during build.

---

## 🔐 Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Default | Description |
|----------|---------|-------------|
| `DJANGO_SECRET_KEY` | *(required)* | Django secret key — generate with `python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"` |
| `DEBUG` | `True` | Set `False` in production |
| `ALLOWED_HOSTS` | `127.0.0.1,localhost` | Comma-separated allowed hostnames |
| `MODEL_PATH` | `./best_churn_model.pkl` | Absolute or relative path to `.pkl` bundle |
| `CORS_ALLOWED_ORIGINS` | *(empty)* | Comma-separated allowed origins for CORS in production |

---

## ⚠️ Limitations & Assumptions

| Limitation | Detail |
|------------|--------|
| **Dataset scope** | Trained on Telco data; not generalisable to banking, SaaS, or retail churn without retraining |
| **Feature count** | Only 5 of the original 21 columns used — service usage (streaming, phone lines, etc.) excluded |
| **Encoding order** | `LabelEncoder` assigns codes based on sorted order; new data must use identical encoding |
| **No temporal model** | Churn *timing* is not predicted — only binary probability at a point in time |
| **Class imbalance** | ~27% churn rate; recall (56%) is lower than precision (67%), meaning some churners are missed |
| **Static model** | No online learning or model drift monitoring; retraining requires manual pipeline re-run |

---

## 🛡️ Production Checklist

- ✅ Training completely separated from inference (no scikit-learn at runtime beyond joblib load)
- ✅ Model + scaler versioned together in a single bundle — prevents preprocessing mismatch
- ✅ `/health/` endpoint for load balancer probes and uptime monitoring
- ✅ CORS whitelist for cross-origin request control
- ✅ Full HSTS + SSL + XSS + CSRF security headers in production
- ✅ Rotating log files (50 MB cap) with structured format + request ID tracing
- ✅ Gunicorn with `preload_app=True` — model loaded once, shared by all workers
- ✅ WhiteNoise for compressed static file serving without nginx dependency
- ✅ `max_requests` worker recycling to prevent memory leaks in long-running processes
- ✅ Render Blueprint (`render.yaml`) for one-click reproducible deployment

---

## 📌 Resume-Ready Summary

> **Customer Churn Prediction System** — Built a production-deployed end-to-end ML system on the IBM Telco dataset. Designed a 4-step training pipeline (preprocess → train → evaluate → serialize) comparing Logistic Regression, Decision Tree, Random Forest, and SVM across 5 metrics. Selected Random Forest (ROC-AUC 0.8576) and deployed via Django REST Framework with full input validation (DRF serializers + cross-field checks), UUID request tracing, rotating structured logs, and a `/health/` endpoint. Configured production security: HSTS, SSL redirect, XSS/CSRF/clickjacking headers. Deployed on Render.com using Gunicorn + WhiteNoise with Git LFS for model artifact delivery. Covered by 9-test suite with mock-based isolation.

**Tech Stack**: `Python 3.10` · `scikit-learn` · `Django 4.2` · `Django REST Framework` · `Gunicorn` · `WhiteNoise` · `Render.com` · `Git LFS` · `joblib` · `pandas`

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙌 Author

**Ravinthra Amulraj**  
MCA Graduate · Machine Learning & Backend Development

[![GitHub](https://img.shields.io/badge/GitHub-Ravinthra-181717?logo=github&style=flat-square)](https://github.com/Ravinthra)