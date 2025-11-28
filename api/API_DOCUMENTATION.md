# FastAPI Documentation

## Endpoints Summary

### ✅ Root & Health Endpoints
- ✅ `GET /` - Welcome message and API information
- ✅ `GET /health` - Health check with database and memory status

### ✅ Prediction Endpoints (PREFIX: /predict)
- ✅ `POST /predict/` - Single prediction
- ✅ `POST /predict/batch` - Batch predictions (max 1000)
- ✅ `GET /predict/history` - Get prediction history with pagination
- ✅ `DELETE /predict/cache` - Clear prediction cache

### ✅ Model Management Endpoints (PREFIX: /models)
- ✅ `GET /models/` - List all registered models
- ✅ `GET /models/{name}/latest` - Get latest model version (query param: stage)
- ✅ `POST /models/{name}/promote` - Promote model to stage

### ✅ Monitoring Endpoints (PREFIX: /metrics)
- ✅ `GET /metrics/drift` - Get latest data drift metrics
- ✅ `GET /metrics/performance` - Get latest model performance metrics

### ✅ Experiment Endpoints (PREFIX: /experiments)
- ✅ `GET /experiments/` - List all MLflow experiments

## Complete Endpoint List

### 1. Root Endpoints

#### GET /
Welcome message and API overview.

**Response:**
```json
{
    "message": "Sales Forecasting MLOps API",
    "version": "1.0.0",
    "author": "Amey Talkatkar",
    "docs": "/docs",
    "health": "/health"
}
```

#### GET /health
Health check with system information.

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2024-11-25T10:30:00",
    "environment": "production",
    "database": "healthy",
    "memory": {
        "used_mb": 450.5,
        "available_mb": 1549.5,
        "percent": 22.5
    }
}
```

### 2. Prediction Endpoints

#### POST /predict/
Make single prediction.

**Request:**
```json
{
    "date": "2024-12-01",
    "region": "North",
    "product": "Electronics",
    "category": "Technology",
    "price": 299.99,
    "quantity": 50
}
```

**Response:**
```json
{
    "predicted_sales": 14999.50,
    "confidence": 0.87,
    "model_version": "1",
    "model_name": "sales_forecasting_production",
    "timestamp": "2024-11-25T10:30:00"
}
```

#### POST /predict/batch
Make batch predictions (max 1000 samples).

**Request:**
```json
{
    "data": [
        {"date": "2024-12-01", "region": "North", "product": "Electronics", "price": 299.99, "quantity": 50},
        {"date": "2024-12-02", "region": "South", "product": "Clothing", "price": 49.99, "quantity": 100}
    ]
}
```

**Response:**
```json
{
    "predictions": [...],
    "count": 2,
    "model_version": "1",
    "processing_time_ms": 245.3
}
```

#### GET /predict/history
Get prediction history with pagination.

**Query Parameters:**
- `limit` (int): Number of predictions (default: 100)
- `offset` (int): Offset for pagination (default: 0)

**Example:** `GET /predict/history?limit=10&offset=0`

**Response:**
```json
{
    "predictions": [...],
    "count": 10,
    "limit": 10,
    "offset": 0,
    "timestamp": "2024-11-25T10:30:00"
}
```

#### DELETE /predict/cache
Clear model and prediction cache.

**Response:**
```json
{
    "status": "success",
    "message": "Prediction cache cleared",
    "timestamp": "2024-11-25T10:30:00"
}
```

### 3. Model Management Endpoints

#### GET /models/
List all registered models.

**Response:**
```json
{
    "models": [
        {
            "name": "sales_forecasting_production",
            "version": "1",
            "stage": "Production",
            "created_at": "2024-11-20T10:00:00",
            "metrics": {"rmse": 95.5, "mae": 75.2}
        }
    ],
    "count": 1,
    "timestamp": "2024-11-25T10:30:00"
}
```

#### GET /models/{name}/latest
Get latest model version for specified stage.

**Query Parameters:**
- `stage` (str): Model stage (default: "Production")

**Example:** `GET /models/sales_forecasting_production/latest?stage=Production`

**Response:**
```json
{
    "name": "sales_forecasting_production",
    "version": "1",
    "stage": "Production",
    "created_at": "2024-11-20T10:00:00",
    "metrics": {"rmse": 95.5}
}
```

#### POST /models/{name}/promote
Promote model to specified stage.

**Request:**
```json
{
    "version": "2",
    "stage": "Production"
}
```

**Response:**
```json
{
    "status": "success",
    "message": "Model sales_forecasting_production v2 promoted to Production",
    "timestamp": "2024-11-25T10:30:00"
}
```

### 4. Monitoring Endpoints

#### GET /metrics/drift
Get latest data drift metrics.

**Response:**
```json
{
    "timestamp": "2024-11-25T10:30:00",
    "has_drift": true,
    "drifted_features": ["price", "quantity"],
    "drift_scores": {"price": 0.08, "quantity": 0.12},
    "threshold": 0.05
}
```

#### GET /metrics/performance
Get latest model performance metrics.

**Response:**
```json
{
    "timestamp": "2024-11-25T10:30:00",
    "rmse": 98.5,
    "mae": 78.2,
    "r2": 0.85,
    "mape": 12.5,
    "sample_size": 1000
}
```

### 5. Experiment Endpoints

#### GET /experiments/
List all MLflow experiments.

**Response:**
```json
{
    "experiments": [
        {
            "experiment_id": "1",
            "name": "sales_forecasting",
            "artifact_location": "s3://...",
            "lifecycle_stage": "active"
        }
    ]
}
```

## API Usage Examples

### Python
```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict/",
    json={
        "date": "2024-12-01",
        "region": "North",
        "product": "Electronics",
        "price": 299.99,
        "quantity": 50
    }
)
print(response.json())

# Get model info
response = requests.get("http://localhost:8000/models/sales_forecasting_production/latest")
print(response.json())
```

### cURL
```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2024-12-01",
    "region": "North",
    "product": "Electronics",
    "price": 299.99,
    "quantity": 50
  }'

# List models
curl http://localhost:8000/models/
```

## Running the API

```bash
# Development
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Interactive Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
