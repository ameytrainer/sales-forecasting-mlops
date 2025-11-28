"""
Model Training Module

Train machine learning models with ML

flow tracking:
- Linear Regression
- Random Forest
- XGBoost
- Automatic hyperparameter tracking
- Model artifact logging

Author: Amey Talkatkar
"""

import os
from typing import Dict, Any, Optional, Tuple
import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from src.config import get_settings
from src.utils import setup_logging, timer, get_git_commit_hash


logger = setup_logging(__name__)


class ModelTrainer:
    """
    Train machine learning models with MLflow tracking.
    
    Examples:
        >>> trainer = ModelTrainer()
        >>> run_id, metrics = trainer.train_model('rf', X_train, y_train, X_test, y_test)
    """
    
    def __init__(self):
        """Initialize model trainer with settings."""
        self.settings = get_settings()
        self._setup_mlflow()
        logger.info("ModelTrainer initialized")
    
    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        
        # Set authentication if provided
        self.settings.setup_mlflow_auth()
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(
                self.settings.mlflow_experiment_name
            )
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    self.settings.mlflow_experiment_name
                )
                logger.info(f"Created experiment: {self.settings.mlflow_experiment_name}")
            else:
                experiment_id = experiment.experiment_id
            
            mlflow.set_experiment(self.settings.mlflow_experiment_name)
            logger.info(f"Using MLflow experiment: {self.settings.mlflow_experiment_name}")
            
        except Exception as e:
            logger.error(f"Failed to setup MLflow: {e}")
            raise
    
    @timer
    def train_model(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        params: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None
    ) -> Tuple[str, str, Dict[str, float]]:
        """
        Train a model with MLflow tracking.
        
        Args:
            model_type: Model type ('lr', 'rf', 'xgb')
            X_train: Training features
            y_train: Training target
            X_test: Test features (optional)
            y_test: Test target (optional)
            params: Model parameters (uses defaults if None)
            run_name: MLflow run name
            
        Returns:
            Tuple of (run_id, model_name, metrics_dict)
            
        Example:
            >>> run_id, model_name, metrics = trainer.train_model('rf', X_train, y_train)
        """
        logger.info(f"Training {model_type} model...")
        
        # Get model parameters
        if params is None:
            params = self.settings.get_model_params(model_type)
        
        # Generate run name
        if run_name is None:
            from datetime import datetime
            run_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Start MLflow run
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            logger.info(f"MLflow run ID: {run_id}")
            
            # Log basic info
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("git_commit", get_git_commit_hash())
            
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("train_size", len(X_train))
            if X_test is not None:
                mlflow.log_param("test_size", len(X_test))
            
            # Train model
            start_time = time.time()
            model = self._get_model(model_type, params)
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            mlflow.log_metric("training_time_seconds", training_time)
            logger.info(f"Training completed in {training_time:.2f}s")
            
            # Make predictions and evaluate
            y_train_pred = model.predict(X_train)
            
            from src.models.evaluate import ModelEvaluator
            evaluator = ModelEvaluator()
            
            # Training metrics
            train_metrics = evaluator.calculate_metrics(y_train, y_train_pred)
            for metric_name, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", value)
            
            logger.info(f"Training metrics: {train_metrics}")
            
            # Test metrics (if provided)
            test_metrics = {}
            if X_test is not None and y_test is not None:
                y_test_pred = model.predict(X_test)
                test_metrics = evaluator.calculate_metrics(y_test, y_test_pred)
                for metric_name, value in test_metrics.items():
                    mlflow.log_metric(f"test_{metric_name}", value)
                
                logger.info(f"Test metrics: {test_metrics}")
            
            # Log model
            model_name = f"{model_type}_model"
            
            if model_type in ['lr', 'rf']:
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name=None  # Register separately if needed
                )
            elif model_type == 'xgb':
                mlflow.xgboost.log_model(
                    model,
                    "model",
                    registered_model_name=None
                )
            
            logger.info(f"Model logged to MLflow")
            
            # Combine metrics
            all_metrics = {**train_metrics}
            if test_metrics:
                all_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
            
            return run_id, model_name, all_metrics
    
    def _get_model(self, model_type: str, params: Dict[str, Any]):
        """Get model instance based on type."""
        if model_type == 'lr':
            return LinearRegression(**params)
        elif model_type == 'rf':
            return RandomForestRegressor(**params)
        elif model_type == 'xgb':
            return xgb.XGBRegressor(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @timer
    def train_multiple_models(
        self,
        model_types: list,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None
    ) -> Dict[str, Tuple[str, Dict[str, float]]]:
        """
        Train multiple models and return results.
        
        Args:
            model_types: List of model types to train
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary mapping model_type to (run_id, metrics)
            
        Example:
            >>> results = trainer.train_multiple_models(['lr', 'rf', 'xgb'], X_train, y_train)
        """
        logger.info(f"Training {len(model_types)} models...")
        
        results = {}
        
        for model_type in model_types:
            try:
                run_id, model_name, metrics = self.train_model(
                    model_type, X_train, y_train, X_test, y_test
                )
                results[model_type] = (run_id, metrics)
                logger.info(f"✅ {model_type}: {metrics.get('test_rmse', 'N/A')}")
            except Exception as e:
                logger.error(f"❌ Failed to train {model_type}: {e}")
                results[model_type] = (None, {})
        
        return results
    
    def get_best_model(
        self,
        results: Dict[str, Tuple[str, Dict[str, float]]],
        metric: str = 'test_rmse',
        ascending: bool = True
    ) -> Tuple[str, str, Dict[str, float]]:
        """
        Get best model from training results.
        
        Args:
            results: Results from train_multiple_models
            metric: Metric to use for comparison
            ascending: True if lower is better
            
        Returns:
            Tuple of (best_model_type, run_id, metrics)
        """
        valid_results = {
            model_type: (run_id, metrics)
            for model_type, (run_id, metrics) in results.items()
            if run_id is not None and metric in metrics
        }
        
        if not valid_results:
            raise ValueError("No valid results found")
        
        # Find best model
        sorted_results = sorted(
            valid_results.items(),
            key=lambda x: x[1][1][metric],
            reverse=not ascending
        )
        
        best_model_type = sorted_results[0][0]
        best_run_id = sorted_results[0][1][0]
        best_metrics = sorted_results[0][1][1]
        
        logger.info(f"Best model: {best_model_type} (run_id: {best_run_id})")
        logger.info(f"Best {metric}: {best_metrics[metric]:.4f}")
        
        return best_model_type, best_run_id, best_metrics


if __name__ == "__main__":
    # Test model trainer
    trainer = ModelTrainer()
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    X_train = pd.DataFrame({
        f'feature_{i}': np.random.randn(n_samples)
        for i in range(10)
    })
    y_train = pd.Series(
        2 * X_train['feature_0'] + X_train['feature_1'] + np.random.randn(n_samples)
    )
    
    X_test = pd.DataFrame({
        f'feature_{i}': np.random.randn(200)
        for i in range(10)
    })
    y_test = pd.Series(
        2 * X_test['feature_0'] + X_test['feature_1'] + np.random.randn(200)
    )
    
    logger.info("Testing model training...")
    
    # Train single model
    run_id, model_name, metrics = trainer.train_model(
        'lr', X_train, y_train, X_test, y_test
    )
    logger.info(f"Trained linear regression: {metrics}")
    
    # Train multiple models
    results = trainer.train_multiple_models(
        ['lr', 'rf'], X_train, y_train, X_test, y_test
    )
    
    # Get best model
    best_type, best_run_id, best_metrics = trainer.get_best_model(results)
    logger.info(f"Best model: {best_type} with metrics {best_metrics}")
    
    logger.info("Model trainer tests completed")
