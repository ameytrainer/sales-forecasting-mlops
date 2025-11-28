"""
Model Evaluation Module

Calculate and report model performance metrics.

Author: Amey Talkatkar
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import get_settings
from src.utils import setup_logging


logger = setup_logging(__name__)


class ModelEvaluator:
    """
    Evaluate model performance with various metrics.
    
    Examples:
        >>> evaluator = ModelEvaluator()
        >>> metrics = evaluator.calculate_metrics(y_true, y_pred)
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.settings = get_settings()
        logger.info("ModelEvaluator initialized")
    
    def calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
        }
    
    def compare_models(
        self,
        results: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """Compare multiple model results."""
        return pd.DataFrame(results).T.sort_values('rmse')


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    y_true = pd.Series([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    logger.info(f"Metrics: {metrics}")
