"""API Endpoint Tests"""
import pytest
from fastapi.testclient import TestClient
import sys
sys.path.insert(0, '/home/ubuntu/sales-forecasting-mlops')

from api.main import app

client = TestClient(app)

def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_endpoint():
    """Test health check."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data

def test_predict_endpoint():
    """Test single prediction."""
    payload = {
        "date": "2024-12-01",
        "region": "North",
        "product": "Electronics",
        "price": 299.99,
        "quantity": 50
    }
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_sales" in data
    assert "model_version" in data
