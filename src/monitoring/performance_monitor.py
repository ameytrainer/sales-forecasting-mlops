"""
Performance Monitoring Module

Monitor model performance over time and detect degradation.

Author: Amey Talkatkar
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.config import get_settings
from src.utils import setup_logging
from src.models.evaluate import ModelEvaluator


logger = setup_logging(__name__)


class PerformanceMonitor:
    """
    Monitor model performance and detect degradation.
    
    Examples:
        >>> monitor = PerformanceMonitor()
        >>> report = monitor.check_performance(predictions_df, baseline_metrics)
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self.settings = get_settings()
        self.evaluator = ModelEvaluator()
        logger.info("PerformanceMonitor initialized")
    
    def check_performance(
        self,
        predictions_df: pd.DataFrame,
        baseline_metrics: Dict[str, float],
        y_true_col: str = 'actual',
        y_pred_col: str = 'predicted',
        window_days: Optional[int] = None
    ) -> Dict:
        """
        Check current performance against baseline.
        
        Args:
            predictions_df: DataFrame with predictions and actuals
            baseline_metrics: Baseline metrics to compare against
            y_true_col: Column name for actual values
            y_pred_col: Column name for predicted values
            window_days: Days to analyze (default from settings)
            
        Returns:
            Performance report dictionary
        """
        window_days = window_days or self.settings.monitoring_window_days
        
        logger.info(f"Checking performance for last {window_days} days")
        
        # Filter to recent data
        if 'timestamp' in predictions_df.columns:
            cutoff_date = datetime.now() - timedelta(days=window_days)
            recent_data = predictions_df[
                pd.to_datetime(predictions_df['timestamp']) >= cutoff_date
            ]
        else:
            recent_data = predictions_df.tail(window_days * 100)  # Approximate
        
        if len(recent_data) == 0:
            logger.warning("No recent data found")
            return {'error': 'No recent data'}
        
        logger.info(f"Analyzing {len(recent_data)} recent predictions")
        
        # Calculate current metrics
        current_metrics = self.evaluator.calculate_metrics(
            recent_data[y_true_col],
            recent_data[y_pred_col].values
        )
        
        # Compare with baseline
        degradation_detected = False
        metric_comparisons = {}
        
        for metric, baseline_value in baseline_metrics.items():
            if metric in current_metrics:
                current_value = current_metrics[metric]
                
                # Calculate degradation percentage
                if baseline_value != 0:
                    degradation_pct = (
                        (current_value - baseline_value) / abs(baseline_value) * 100
                    )
                else:
                    degradation_pct = 0
                
                # Check if degraded (higher is worse for error metrics)
                is_degraded = abs(degradation_pct) > (
                    self.settings.performance_degradation_threshold * 100
                )
                
                metric_comparisons[metric] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'degradation_pct': degradation_pct,
                    'is_degraded': is_degraded
                }
                
                if is_degraded and metric in ['rmse', 'mae', 'mape']:
                    degradation_detected = True
                    logger.warning(
                        f"⚠️  Performance degradation in {metric}: "
                        f"{baseline_value:.4f} → {current_value:.4f} "
                        f"({degradation_pct:+.2f}%)"
                    )
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'window_days': window_days,
            'sample_size': len(recent_data),
            'degradation_detected': degradation_detected,
            'current_metrics': current_metrics,
            'baseline_metrics': baseline_metrics,
            'metric_comparisons': metric_comparisons
        }
        
        if degradation_detected:
            logger.warning("⚠️  Performance degradation detected!")
        else:
            logger.info("✅ Performance within acceptable range")
        
        return report
    
    def calculate_performance_trend(
        self,
        predictions_df: pd.DataFrame,
        y_true_col: str = 'actual',
        y_pred_col: str = 'predicted',
        window_size: int = 100
    ) -> pd.DataFrame:
        """
        Calculate performance metrics over rolling windows.
        
        Args:
            predictions_df: DataFrame with predictions
            y_true_col: Column for actual values
            y_pred_col: Column for predicted values
            window_size: Rolling window size
            
        Returns:
            DataFrame with rolling metrics
        """
        logger.info(f"Calculating performance trend (window={window_size})")
        
        # Sort by timestamp if available
        if 'timestamp' in predictions_df.columns:
            df = predictions_df.sort_values('timestamp').copy()
        else:
            df = predictions_df.copy()
        
        # Calculate rolling metrics
        trends = []
        
        for i in range(window_size, len(df), window_size // 2):
            window_data = df.iloc[i-window_size:i]
            
            metrics = self.evaluator.calculate_metrics(
                window_data[y_true_col],
                window_data[y_pred_col].values
            )
            
            if 'timestamp' in df.columns:
                metrics['timestamp'] = window_data['timestamp'].iloc[-1]
            else:
                metrics['row_index'] = i
            
            trends.append(metrics)
        
        trend_df = pd.DataFrame(trends)
        logger.info(f"Calculated {len(trend_df)} trend points")
        
        return trend_df
    
    def should_retrain(
        self,
        performance_report: Dict,
        drift_report: Optional[Dict] = None
    ) -> Dict[str, bool]:
        """
        Determine if model should be retrained.
        
        Args:
            performance_report: Performance check report
            drift_report: Drift detection report (optional)
            
        Returns:
            Dictionary with retraining recommendation
        """
        reasons = []
        
        # Check performance degradation
        if performance_report.get('degradation_detected', False):
            reasons.append('performance_degradation')
        
        # Check drift
        if drift_report and drift_report.get('has_drift', False):
            drift_pct = drift_report.get('drift_percentage', 0)
            if drift_pct > 20:  # More than 20% features drifted
                reasons.append('significant_data_drift')
        
        should_retrain = len(reasons) > 0
        
        recommendation = {
            'should_retrain': should_retrain,
            'reasons': reasons,
            'priority': 'HIGH' if should_retrain else 'LOW'
        }
        
        if should_retrain:
            logger.warning(f"🔄 Retraining recommended: {reasons}")
        else:
            logger.info("✅ No retraining needed")
        
        return recommendation


if __name__ == "__main__":
    monitor = PerformanceMonitor()
    
    # Test performance monitoring
    np.random.seed(42)
    predictions_df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='H'),
        'actual': np.random.randn(1000) * 10 + 100,
        'predicted': np.random.randn(1000) * 10 + 100
    })
    
    baseline_metrics = {
        'rmse': 10.0,
        'mae': 8.0,
        'r2': 0.85
    }
    
    report = monitor.check_performance(predictions_df, baseline_metrics)
    logger.info(f"Performance report: {report}")
    
    # Test retraining recommendation
    recommendation = monitor.should_retrain(report)
    logger.info(f"Retraining recommendation: {recommendation}")
    
    logger.info("Performance monitor tests completed")
