FROM python:3.12.3-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential gcc && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn anthropic python-dotenv numpy pandas scikit-learn xgboost lightgbm shap joblib httpx
COPY . .
CMD ["python", "start.py"]