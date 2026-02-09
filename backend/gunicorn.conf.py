"""
Gunicorn configuration for production deployment.

Usage:
    gunicorn backend.wsgi:application -c gunicorn.conf.py
"""
import multiprocessing
import os

# Bind to port 8000
bind = os.getenv("GUNICORN_BIND", "0.0.0.0:8000")

# Number of worker processes
# Recommended: 2-4 x CPU cores
workers = int(os.getenv("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))

# Worker class
worker_class = "sync"

# Maximum requests per worker before restart (helps prevent memory leaks)
max_requests = 1000
max_requests_jitter = 50

# Timeout for worker processes (seconds)
timeout = 30

# Keep-alive connections
keepalive = 5

# Logging
accesslog = "-"  # stdout
errorlog = "-"   # stderr
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")

# Process naming
proc_name = "churn_predictor"

# Graceful timeout
graceful_timeout = 30

# Preload app for faster worker spawning
preload_app = True
