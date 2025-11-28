"""
Monitoring API Routes
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
import pandas as pd
import sys
sys.path.insert(0, '/home/ubuntu/sales-forecasting-mlops')

from api.models import DriftMetrics, PerformanceMetrics
from src.monitoring import DriftDetector, PerformanceMonitor
from src.config import get_settings

router = APIRouter()

@router.get("/drift", response_model=DriftMetrics)
async def get_drift_metrics():
    """Get latest data drift metrics."""
    try:
        settings = get_settings()
        detector = DriftDetector()
        
        # Load data from database
        from sqlalchemy import create_engine
        engine = create_engine(settings.get_database_url())
        
        query = """
            SELECT * FROM drift_metrics 
            ORDER BY timestamp DESC LIMIT 1
        """
        df = pd.read_sql(query, engine)
        engine.dispose()
        
        if df.empty:
            return DriftMetrics(
                has_drift=False,
                drifted_features=[],
                drift_scores={},
                threshold=settings.drift_detection_threshold
            )
        
        row = df.iloc[0]
        return DriftMetrics(
            timestamp=row['timestamp'],
            has_drift=row['has_drift'],
            drifted_features=row['drifted_features'].split(',') if row['drifted_features'] else [],
            drift_scores={},  # Parse from database
            threshold=settings.drift_detection_threshold
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance", response_model=PerformanceMetrics)
async def get_performance_metrics():
    """Get latest model performance metrics."""
    try:
        settings = get_settings()
        
        from sqlalchemy import create_engine
        engine = create_engine(settings.get_database_url())
        
        query = """
            SELECT * FROM model_metrics 
            ORDER BY timestamp DESC LIMIT 1
        """
        df = pd.read_sql(query, engine)
        engine.dispose()
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No metrics found")
        
        row = df.iloc[0]
        return PerformanceMetrics(
            timestamp=row['timestamp'],
            rmse=row['rmse'],
            mae=row['mae'],
            r2=row['r2_score'],
            mape=row.get('mape'),
            sample_size=row.get('num_predictions', 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
