"""
Model Training and Management Package

Contains modules for model training, evaluation, registry, and prediction.

Modules:
- train: Train machine learning models with MLflow tracking
- evaluate: Evaluate model performance with various metrics
- registry: Manage model registry operations (register, promote, transition)
- predict: Make predictions with trained models

Author: Amey Talkatkar
"""

from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator
from src.models.registry import ModelRegistry
from src.models.predict import ModelPredictor

__all__ = [
    "ModelTrainer",
    "ModelEvaluator",
    "ModelRegistry",
    "ModelPredictor",
]
