"""
Pydantic Models for API Request/Response Validation

Type-safe models for:
- Prediction requests and responses
- Model information
- Monitoring metrics

Author: Amey Talkatkar
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime, date


# ==================== Prediction Models ====================

class PredictionInput(BaseModel):
    """
    Single prediction input.
    
    Example:
        {
            "date": "2024-12-01",
            "region": "North",
            "product": "Electronics",
            "category": "Technology",
            "price": 299.99,
            "quantity": 50
        }
    """
    date: date = Field(..., description="Date for prediction")
    region: str = Field(..., description="Sales region")
    product: str = Field(..., description="Product name")
    category: Optional[str] = Field(None, description="Product category")
    price: float = Field(..., gt=0, description="Product price (must be positive)")
    quantity: int = Field(..., gt=0, description="Quantity (must be positive)")
    
    @field_validator('region')
    @classmethod
    def validate_region(cls, v):
        """Validate region is not empty."""
        if not v or not v.strip():
            raise ValueError("Region cannot be empty")
        return v.strip()
    
    @field_validator('product')
    @classmethod
    def validate_product(cls, v):
        """Validate product is not empty."""
        if not v or not v.strip():
            raise ValueError("Product cannot be empty")
        return v.strip()


class PredictionOutput(BaseModel):
    """
    Single prediction output.
    
    Example:
        {
            "predicted_sales": 14999.50,
            "confidence": 0.87,
            "model_version": "1",
            "model_name": "sales_forecasting_production",
            "timestamp": "2024-11-25T10:30:00"
        }
    """
    predicted_sales: float = Field(..., description="Predicted sales amount")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Prediction confidence (0-1)")
    model_version: str = Field(..., description="Model version used")
    model_name: str = Field(..., description="Model name")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")


class BatchPredictionInput(BaseModel):
    """
    Batch prediction input.
    
    Example:
        {
            "data": [
                {"date": "2024-12-01", "region": "North", ...},
                {"date": "2024-12-02", "region": "South", ...}
            ]
        }
    """
    data: List[PredictionInput] = Field(..., min_length=1, max_length=1000, 
                                       description="List of prediction inputs (max 1000)")


class BatchPredictionOutput(BaseModel):
    """
    Batch prediction output.
    
    Example:
        {
            "predictions": [...],
            "count": 10,
            "model_version": "1",
            "processing_time_ms": 245.3
        }
    """
    predictions: List[PredictionOutput] = Field(..., description="List of predictions")
    count: int = Field(..., description="Number of predictions")
    model_version: str = Field(..., description="Model version used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


# ==================== Model Registry Models ====================

class ModelInfo(BaseModel):
    """
    Model information from registry.
    
    Example:
        {
            "name": "sales_forecasting_production",
            "version": "1",
            "stage": "Production",
            "created_at": "2024-11-20T10:00:00",
            "metrics": {"rmse": 95.5, "mae": 75.2, "r2": 0.87}
        }
    """
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    stage: str = Field(..., description="Model stage (Staging/Production/Archived)")
    created_at: Optional[datetime] = Field(None, description="Model creation timestamp")
    metrics: Optional[Dict[str, float]] = Field(None, description="Model metrics")
    tags: Optional[Dict[str, str]] = Field(None, description="Model tags")


class ModelList(BaseModel):
    """
    List of models.
    
    Example:
        {
            "models": [...],
            "count": 5,
            "timestamp": "2024-11-25T10:30:00"
        }
    """
    models: List[ModelInfo] = Field(..., description="List of models")
    count: int = Field(..., description="Number of models")
    timestamp: datetime = Field(default_factory=datetime.now, description="Query timestamp")


class ModelTransition(BaseModel):
    """
    Model stage transition request.
    
    Example:
        {
            "version": "2",
            "stage": "Production"
        }
    """
    version: str = Field(..., description="Model version to transition")
    stage: str = Field(..., description="Target stage")
    
    @field_validator('stage')
    @classmethod
    def validate_stage(cls, v):
        """Validate stage is one of allowed values."""
        allowed_stages = ['Staging', 'Production', 'Archived']
        if v not in allowed_stages:
            raise ValueError(f"Stage must be one of {allowed_stages}")
        return v


# ==================== Monitoring Models ====================

class DriftMetrics(BaseModel):
    """
    Data drift metrics.
    
    Example:
        {
            "timestamp": "2024-11-25T10:30:00",
            "has_drift": true,
            "drifted_features": ["price", "quantity"],
            "drift_scores": {"price": 0.08, "quantity": 0.12}
        }
    """
    timestamp: datetime = Field(default_factory=datetime.now, description="Metrics timestamp")
    has_drift: bool = Field(..., description="Whether drift was detected")
    drifted_features: List[str] = Field(..., description="Features with detected drift")
    drift_scores: Dict[str, float] = Field(..., description="Drift scores per feature")
    threshold: float = Field(..., description="Drift detection threshold used")


class PerformanceMetrics(BaseModel):
    """
    Model performance metrics.
    
    Example:
        {
            "timestamp": "2024-11-25T10:30:00",
            "rmse": 98.5,
            "mae": 78.2,
            "r2": 0.85,
            "mape": 12.5,
            "sample_size": 1000
        }
    """
    timestamp: datetime = Field(default_factory=datetime.now, description="Metrics timestamp")
    rmse: float = Field(..., description="Root Mean Squared Error")
    mae: float = Field(..., description="Mean Absolute Error")
    r2: float = Field(..., description="R-squared score")
    mape: Optional[float] = Field(None, description="Mean Absolute Percentage Error")
    sample_size: int = Field(..., description="Number of samples in evaluation")


class SystemHealth(BaseModel):
    """
    System health metrics.
    
    Example:
        {
            "timestamp": "2024-11-25T10:30:00",
            "api_status": "healthy",
            "database_status": "healthy",
            "model_status": "healthy",
            "memory_usage_mb": 450.5,
            "memory_percent": 22.5
        }
    """
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    api_status: str = Field(..., description="API status")
    database_status: str = Field(..., description="Database status")
    model_status: str = Field(..., description="Model status")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    memory_percent: float = Field(..., description="Memory usage percentage")


# ==================== Experiment Models ====================

class ExperimentInfo(BaseModel):
    """
    MLflow experiment information.
    
    Example:
        {
            "experiment_id": "1",
            "name": "sales_forecasting",
            "artifact_location": "s3://...",
            "lifecycle_stage": "active",
            "tags": {}
        }
    """
    experiment_id: str = Field(..., description="Experiment ID")
    name: str = Field(..., description="Experiment name")
    artifact_location: Optional[str] = Field(None, description="Artifact storage location")
    lifecycle_stage: str = Field(..., description="Experiment lifecycle stage")
    tags: Optional[Dict[str, str]] = Field(None, description="Experiment tags")


class RunInfo(BaseModel):
    """
    MLflow run information.
    
    Example:
        {
            "run_id": "abc123...",
            "experiment_id": "1",
            "status": "FINISHED",
            "start_time": "2024-11-25T10:00:00",
            "end_time": "2024-11-25T10:15:00",
            "metrics": {"rmse": 95.5},
            "params": {"n_estimators": 100}
        }
    """
    run_id: str = Field(..., description="Run ID")
    experiment_id: str = Field(..., description="Experiment ID")
    status: str = Field(..., description="Run status")
    start_time: Optional[datetime] = Field(None, description="Run start time")
    end_time: Optional[datetime] = Field(None, description="Run end time")
    metrics: Optional[Dict[str, float]] = Field(None, description="Run metrics")
    params: Optional[Dict[str, Any]] = Field(None, description="Run parameters")
    tags: Optional[Dict[str, str]] = Field(None, description="Run tags")


class ExperimentList(BaseModel):
    """
    List of experiments.
    """
    experiments: List[ExperimentInfo] = Field(..., description="List of experiments")
    count: int = Field(..., description="Number of experiments")
    timestamp: datetime = Field(default_factory=datetime.now, description="Query timestamp")


# ==================== Error Models ====================

class ErrorResponse(BaseModel):
    """
    Error response model.
    
    Example:
        {
            "error": "Validation error",
            "detail": "Price must be positive",
            "timestamp": "2024-11-25T10:30:00"
        }
    """
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
