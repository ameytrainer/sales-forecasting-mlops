"""
Model Training and Evaluation Tests

Tests for src/models/ module.

Author: Amey Talkatkar
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
import sys
sys.path.insert(0, '/home/ubuntu/sales-forecasting-mlops')

from src.models import ModelTrainer, ModelEvaluator, ModelRegistry, ModelPredictor


@pytest.fixture
def regression_data():
    """Generate synthetic regression data."""
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    y_series = pd.Series(y, name='target')
    
    # Split into train/test
    split_idx = 800
    X_train = X_df.iloc[:split_idx]
    X_test = X_df.iloc[split_idx:]
    y_train = y_series.iloc[:split_idx]
    y_test = y_series.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test


class TestModelTrainer:
    """Tests for ModelTrainer class."""
    
    def test_linear_regression_training(self, regression_data):
        """Test linear regression training."""
        X_train, X_test, y_train, y_test = regression_data
        
        trainer = ModelTrainer()
        model, metrics = trainer.train_linear_regression(X_train, y_train, X_test, y_test)
        
        assert model is not None
        assert 'train_rmse' in metrics
        assert 'test_rmse' in metrics
        assert metrics['test_rmse'] > 0
    
    def test_random_forest_training(self, regression_data):
        """Test random forest training."""
        X_train, X_test, y_train, y_test = regression_data
        
        trainer = ModelTrainer()
        model, metrics = trainer.train_random_forest(
            X_train, y_train, X_test, y_test,
            n_estimators=10,  # Small for testing speed
            max_depth=5
        )
        
        assert model is not None
        assert 'train_rmse' in metrics
        assert 'test_rmse' in metrics
        assert metrics['test_r2'] <= 1.0
    
    def test_xgboost_training(self, regression_data):
        """Test XGBoost training."""
        X_train, X_test, y_train, y_test = regression_data
        
        trainer = ModelTrainer()
        model, metrics = trainer.train_xgboost(
            X_train, y_train, X_test, y_test,
            n_estimators=10,
            max_depth=3
        )
        
        assert model is not None
        assert 'train_rmse' in metrics
        assert 'test_rmse' in metrics
    
    def test_cross_validation(self, regression_data):
        """Test cross-validation scoring."""
        X_train, X_test, y_train, y_test = regression_data
        
        trainer = ModelTrainer()
        model, _ = trainer.train_linear_regression(X_train, y_train, X_test, y_test)
        
        cv_scores = trainer.cross_validate(model, X_train, y_train, cv=3)
        
        assert 'test_neg_mean_squared_error' in cv_scores
        assert len(cv_scores['test_neg_mean_squared_error']) == 3


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""
    
    def test_calculate_metrics(self, regression_data):
        """Test metrics calculation."""
        X_train, X_test, y_train, y_test = regression_data
        
        # Train simple model
        trainer = ModelTrainer()
        model, _ = trainer.train_linear_regression(X_train, y_train, X_test, y_test)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        evaluator = ModelEvaluator()
        metrics = evaluator.calculate_metrics(y_test, y_pred)
        
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'mape' in metrics
        
        # Check ranges
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['r2'] <= 1.0
    
    def test_compare_models(self, regression_data):
        """Test model comparison."""
        X_train, X_test, y_train, y_test = regression_data
        
        # Train two models
        trainer = ModelTrainer()
        
        model1, metrics1 = trainer.train_linear_regression(X_train, y_train, X_test, y_test)
        model2, metrics2 = trainer.train_random_forest(
            X_train, y_train, X_test, y_test,
            n_estimators=10
        )
        
        models_dict = {
            'linear_regression': {'model': model1, 'metrics': metrics1},
            'random_forest': {'model': model2, 'metrics': metrics2}
        }
        
        # Compare
        evaluator = ModelEvaluator()
        comparison = evaluator.compare_models(models_dict, metric='test_rmse')
        
        assert 'best_model' in comparison
        assert 'rankings' in comparison
        assert 'comparison_df' in comparison
        
        # Check comparison DataFrame
        df = comparison['comparison_df']
        assert len(df) == 2
        assert 'test_rmse' in df.columns


class TestModelPredictor:
    """Tests for ModelPredictor class."""
    
    def test_predict_with_local_model(self, regression_data):
        """Test prediction with local model."""
        X_train, X_test, y_train, y_test = regression_data
        
        # Train model
        trainer = ModelTrainer()
        model, _ = trainer.train_linear_regression(X_train, y_train, X_test, y_test)
        
        # Predict
        predictor = ModelPredictor()
        predictor.model = model  # Set model directly for testing
        
        predictions = predictor.predict(X_test[:10])
        
        assert len(predictions) == 10
        assert isinstance(predictions, np.ndarray)
    
    def test_predict_batch(self, regression_data):
        """Test batch prediction."""
        X_train, X_test, y_train, y_test = regression_data
        
        trainer = ModelTrainer()
        model, _ = trainer.train_linear_regression(X_train, y_train, X_test, y_test)
        
        predictor = ModelPredictor()
        predictor.model = model
        
        predictions = predictor.predict_batch(X_test, batch_size=50)
        
        assert len(predictions) == len(X_test)


class TestModelRegistry:
    """Tests for ModelRegistry class (integration with MLflow)."""
    
    @pytest.mark.integration
    def test_register_model(self, regression_data):
        """Test model registration (requires MLflow)."""
        # Skip if MLflow not configured
        pytest.skip("Requires MLflow configuration")
        
        X_train, X_test, y_train, y_test = regression_data
        
        trainer = ModelTrainer()
        model, _ = trainer.train_linear_regression(X_train, y_train, X_test, y_test)
        
        registry = ModelRegistry()
        
        # This would require MLflow setup
        # version = registry.register_model(model, "test_model")
        # assert version is not None
    
    @pytest.mark.integration
    def test_transition_model(self):
        """Test model stage transition."""
        pytest.skip("Requires MLflow configuration")


# Performance tests
class TestModelPerformance:
    """Performance and memory tests."""
    
    def test_prediction_speed(self, regression_data):
        """Test prediction speed."""
        import time
        
        X_train, X_test, y_train, y_test = regression_data
        
        trainer = ModelTrainer()
        model, _ = trainer.train_linear_regression(X_train, y_train, X_test, y_test)
        
        predictor = ModelPredictor()
        predictor.model = model
        
        # Time 1000 predictions
        start = time.time()
        _ = predictor.predict(X_test)
        duration = time.time() - start
        
        # Should be fast (< 1 second for 200 samples)
        assert duration < 1.0, f"Prediction too slow: {duration:.3f}s"
    
    def test_memory_usage(self, regression_data):
        """Test memory usage during training."""
        import tracemalloc
        
        X_train, X_test, y_train, y_test = regression_data
        
        tracemalloc.start()
        
        trainer = ModelTrainer()
        _ = trainer.train_linear_regression(X_train, y_train, X_test, y_test)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Peak memory should be reasonable (< 100 MB for this test)
        peak_mb = peak / 1024 / 1024
        assert peak_mb < 100, f"Memory usage too high: {peak_mb:.2f} MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
