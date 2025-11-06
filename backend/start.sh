#!/bin/bash

# Create necessary directories
mkdir -p models
mkdir -p data

# Set environment variables for production
export ENVIRONMENT=production
export DEBUG=false

# Start the application
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1