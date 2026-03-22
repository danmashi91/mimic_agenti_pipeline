FROM python:3.12.3-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential gcc && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi uvicorn anthropic python-dotenv httpx \
    joblib==1.5.3 \
    numpy==2.4.3 \
    pandas \
    scikit-learn==1.8.0 \
    xgboost==3.2.0 \
    lightgbm==4.6.0 \
    shap
COPY . .
CMD ["python", "start.py"]