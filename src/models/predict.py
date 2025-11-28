"""
Model Prediction Module

Make predictions with trained models.

Author: Amey Talkatkar
"""

from typing import Optional, Union, List
import pandas as pd
import numpy as np
import mlflow

from src.config import get_settings
from src.utils import setup_logging


logger = setup_logging(__name__)


class ModelPredictor:
    """
    Make predictions with trained models.
    
    Examples:
        >>> predictor = ModelPredictor()
        >>> predictions = predictor.predict(X_test, run_id="abc123")
    """
    
    def __init__(self):
        """Initialize predictor."""
        self.settings = get_settings()
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        self.settings.setup_mlflow_auth()
        self.model_cache = {}
        logger.info("ModelPredictor initialized")
    
    def predict(
        self,
        X: pd.DataFrame,
        run_id: Optional[str] = None,
        model_name: Optional[str] = None,
        stage: str = "Production"
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features DataFrame
            run_id: MLflow run ID (if loading from run)
            model_name: Model name (if loading from registry)
            stage: Model stage (if loading from registry)
            
        Returns:
            Predictions array
        """
        # Load model
        if run_id:
            model_uri = f"runs:/{run_id}/model"
        elif model_name:
            model_uri = f"models:/{model_name}/{stage}"
        else:
            raise ValueError("Provide either run_id or model_name")
        
        logger.info(f"Loading model from: {model_uri}")
        
        # Check cache
        if model_uri not in self.model_cache:
            self.model_cache[model_uri] = mlflow.pyfunc.load_model(model_uri)
            logger.info("Model loaded and cached")
        else:
            logger.info("Using cached model")
        
        model = self.model_cache[model_uri]
        
        # Make predictions
        predictions = model.predict(X)
        logger.info(f"Generated {len(predictions)} predictions")
        
        return predictions
    
    def predict_batch(
        self,
        X: pd.DataFrame,
        batch_size: int = 1000,
        **kwargs
    ) -> np.ndarray:
        """Make predictions in batches."""
        logger.info(f"Making batch predictions (batch_size={batch_size})")
        
        predictions = []
        for i in range(0, len(X), batch_size):
            batch = X.iloc[i:i+batch_size]
            batch_preds = self.predict(batch, **kwargs)
            predictions.extend(batch_preds)
        
        return np.array(predictions)
    
    def clear_cache(self) -> None:
        """Clear model cache."""
        self.model_cache.clear()
        logger.info("Model cache cleared")


if __name__ == "__main__":
    predictor = ModelPredictor()
    logger.info("Model predictor tests completed")
