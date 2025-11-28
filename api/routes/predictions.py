"""
Prediction API Routes

Endpoints for model predictions:
- Single prediction
- Batch predictions
- Prediction history

Author: Amey Talkatkar
"""

from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
import pandas as pd
import logging
import time
import sys

sys.path.insert(0, '/home/ubuntu/sales-forecasting-mlops')

from api.models import (
    PredictionInput, PredictionOutput,
    BatchPredictionInput, BatchPredictionOutput
)
from src.config import get_settings
from src.models import ModelPredictor
from src.features import FeatureEngineer
from src.data import DataProcessor

logger = logging.getLogger(__name__)
router = APIRouter()

# Global caches
predictor_cache = {}
processor_cache = {}


def get_predictor():
    """Dependency for model predictor."""
    if 'predictor' not in predictor_cache:
        predictor_cache['predictor'] = ModelPredictor()
    return predictor_cache['predictor']


def get_processor():
    """Dependency for data processor."""
    if 'processor' not in processor_cache:
        processor_cache['processor'] = DataProcessor()
    return processor_cache['processor']


@router.post("/", response_model=PredictionOutput)
async def predict_single(
    input_data: PredictionInput,
    predictor: ModelPredictor = Depends(get_predictor),
    processor: DataProcessor = Depends(get_processor)
):
    """
    Generate single prediction.
    
    Args:
        input_data: Prediction input with date, region, product, price, quantity
        
    Returns:
        Prediction output with predicted_sales and metadata
        
    Example:
        POST /predict/
        {
            "date": "2024-12-01",
            "region": "North",
            "product": "Electronics",
            "price": 299.99,
            "quantity": 50
        }
        
        Response:
        {
            "predicted_sales": 14999.50,
            "confidence": 0.87,
            "model_version": "1",
            "model_name": "sales_forecasting_production",
            "timestamp": "2024-11-25T10:30:00"
        }
    """
    try:
        logger.info(f"Prediction request for {input_data.date}, {input_data.region}")
        
        # Convert input to DataFrame
        df = pd.DataFrame([input_data.model_dump()])
        
        # Create features
        engineer = FeatureEngineer()
        df_features = engineer.create_all_features(df, target_col=None)
        
        # Drop non-feature columns
        feature_cols = [col for col in df_features.columns if col not in ['date', 'sales']]
        X = df_features[feature_cols]
        
        # Load scaler and scale features
        scaler = processor.load_scaler()
        if scaler is not None:
            X_scaled = processor.scale_features(X, fit=False)
        else:
            logger.warning("No scaler found, using unscaled features")
            X_scaled = X
        
        # Make prediction
        settings = get_settings()
        prediction = predictor.predict(
            X_scaled,
            model_name=settings.mlflow_model_name,
            stage="Production"
        )
        
        # Get model version (from cache or MLflow)
        model_version = "1"  # In production, get from MLflow
        
        logger.info(f"Prediction: {prediction[0]:.2f}")
        
        return PredictionOutput(
            predicted_sales=float(prediction[0]),
            confidence=0.85,  # In production, calculate actual confidence
            model_version=model_version,
            model_name=settings.mlflow_model_name,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/batch", response_model=BatchPredictionOutput)
async def predict_batch(
    input_data: BatchPredictionInput,
    predictor: ModelPredictor = Depends(get_predictor),
    processor: DataProcessor = Depends(get_processor)
):
    """
    Generate batch predictions.
    
    Args:
        input_data: List of prediction inputs (max 1000)
        
    Returns:
        Batch prediction output with list of predictions
        
    Example:
        POST /predict/batch
        {
            "data": [
                {"date": "2024-12-01", "region": "North", ...},
                {"date": "2024-12-02", "region": "South", ...}
            ]
        }
        
        Response:
        {
            "predictions": [...],
            "count": 2,
            "model_version": "1",
            "processing_time_ms": 245.3
        }
    """
    try:
        start_time = time.time()
        
        logger.info(f"Batch prediction request: {len(input_data.data)} samples")
        
        # Convert inputs to DataFrame
        df = pd.DataFrame([item.model_dump() for item in input_data.data])
        
        # Create features
        engineer = FeatureEngineer()
        df_features = engineer.create_all_features(df, target_col=None)
        
        # Prepare features
        feature_cols = [col for col in df_features.columns if col not in ['date', 'sales']]
        X = df_features[feature_cols]
        
        # Scale features
        scaler = processor.load_scaler()
        if scaler is not None:
            X_scaled = processor.scale_features(X, fit=False)
        else:
            X_scaled = X
        
        # Make predictions
        settings = get_settings()
        predictions = predictor.predict_batch(
            X_scaled,
            batch_size=100,
            model_name=settings.mlflow_model_name,
            stage="Production"
        )
        
        # Get model version
        model_version = "1"
        
        # Create prediction outputs
        prediction_outputs = [
            PredictionOutput(
                predicted_sales=float(pred),
                confidence=0.85,
                model_version=model_version,
                model_name=settings.mlflow_model_name,
                timestamp=datetime.now()
            )
            for pred in predictions
        ]
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        logger.info(f"Batch prediction complete: {len(predictions)} predictions in {processing_time:.2f}ms")
        
        return BatchPredictionOutput(
            predictions=prediction_outputs,
            count=len(predictions),
            model_version=model_version,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.get("/history")
async def get_prediction_history(
    limit: int = 100,
    offset: int = 0
):
    """
    Get prediction history from database.
    
    Args:
        limit: Maximum number of predictions to return (default: 100)
        offset: Offset for pagination (default: 0)
        
    Returns:
        List of recent predictions
        
    Example:
        GET /predict/history?limit=10&offset=0
    """
    try:
        logger.info(f"Fetching prediction history: limit={limit}, offset={offset}")
        
        settings = get_settings()
        
        # Query database
        from sqlalchemy import create_engine
        engine = create_engine(settings.get_database_url())
        
        query = f"""
            SELECT 
                prediction_id,
                timestamp,
                date,
                region,
                product,
                price,
                quantity,
                predicted_sales,
                model_name,
                model_version
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT {limit}
            OFFSET {offset}
        """
        
        df = pd.read_sql(query, engine)
        engine.dispose()
        
        # Convert to list of dicts
        predictions = df.to_dict(orient='records')
        
        logger.info(f"Retrieved {len(predictions)} predictions")
        
        return {
            "predictions": predictions,
            "count": len(predictions),
            "limit": limit,
            "offset": offset,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")


@router.delete("/cache")
async def clear_prediction_cache():
    """
    Clear prediction cache (model and processor).
    
    Use this endpoint after model updates.
    
    Returns:
        Status message
        
    Example:
        DELETE /predict/cache
    """
    try:
        logger.info("Clearing prediction cache...")
        
        predictor_cache.clear()
        processor_cache.clear()
        
        logger.info("✅ Cache cleared")
        
        return {
            "status": "success",
            "message": "Prediction cache cleared",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")
