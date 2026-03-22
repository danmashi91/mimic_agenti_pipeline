# Dockerfile for MIMIC-IV Clinical Analytics API
# Optimised for Railway deployment

FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    anthropic \
    python-dotenv \
    numpy \
    pandas \
    scikit-learn \
    xgboost \
    lightgbm \
    shap \
    joblib \
    httpx

# Copy the full project
COPY . .

# Expose port (Railway sets PORT env var automatically)
EXPOSE 8000

# Start the API
CMD ["python", "start.py"]
# force rebuild
