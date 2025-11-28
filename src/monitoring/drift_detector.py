"""
Drift Detection Module

Detect data drift and prediction drift using statistical tests.

Author: Amey Talkatkar
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from scipy import stats

from src.config import get_settings
from src.utils import setup_logging


logger = setup_logging(__name__)


class DriftDetector:
    """
    Detect data and prediction drift.
    
    Examples:
        >>> detector = DriftDetector()
        >>> report = detector.detect_drift(reference_data, current_data)
    """
    
    def __init__(self):
        """Initialize drift detector."""
        self.settings = get_settings()
        logger.info("DriftDetector initialized")
    
    def detect_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        features: Optional[List[str]] = None,
        threshold: Optional[float] = None
    ) -> Dict:
        """
        Detect drift using Kolmogorov-Smirnov test.
        
        Args:
            reference_data: Reference (training) data
            current_data: Current (production) data
            features: Features to check (all numeric if None)
            threshold: P-value threshold (default from settings)
            
        Returns:
            Drift report dictionary
        """
        threshold = threshold or self.settings.drift_detection_threshold
        
        if features is None:
            features = reference_data.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Checking drift for {len(features)} features")
        
        drift_results = {}
        drifted_features = []
        
        for feature in features:
            if feature not in reference_data.columns or feature not in current_data.columns:
                continue
            
            # Kolmogorov-Smirnov test
            statistic, pvalue = stats.ks_2samp(
                reference_data[feature].dropna(),
                current_data[feature].dropna()
            )
            
            drift_detected = pvalue < threshold
            
            drift_results[feature] = {
                'ks_statistic': float(statistic),
                'p_value': float(pvalue),
                'drift_detected': drift_detected
            }
            
            if drift_detected:
                drifted_features.append(feature)
                logger.warning(
                    f"Drift detected in {feature}: "
                    f"KS statistic={statistic:.4f}, p-value={pvalue:.4f}"
                )
        
        has_drift = len(drifted_features) > 0
        
        report = {
            'has_drift': has_drift,
            'drifted_features': drifted_features,
            'drift_count': len(drifted_features),
            'total_features': len(features),
            'drift_percentage': len(drifted_features) / len(features) * 100,
            'threshold': threshold,
            'feature_results': drift_results
        }
        
        if has_drift:
            logger.warning(f"⚠️  Drift detected in {len(drifted_features)} features")
        else:
            logger.info("✅ No drift detected")
        
        return report
    
    def calculate_psi(
        self,
        reference: pd.Series,
        current: pd.Series,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins for discretization
            
        Returns:
            PSI score
        """
        # Create bins based on reference
        breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)
        
        # Bin both distributions
        ref_binned = np.digitize(reference, breakpoints[:-1])
        cur_binned = np.digitize(current, breakpoints[:-1])
        
        # Calculate distributions
        ref_dist = np.bincount(ref_binned, minlength=len(breakpoints)) / len(reference)
        cur_dist = np.bincount(cur_binned, minlength=len(breakpoints)) / len(current)
        
        # Add small constant to avoid log(0)
        ref_dist = ref_dist + 1e-10
        cur_dist = cur_dist + 1e-10
        
        # Calculate PSI
        psi = np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist))
        
        return float(psi)


if __name__ == "__main__":
    detector = DriftDetector()
    
    # Test drift detection
    np.random.seed(42)
    ref_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(0, 1, 1000)
    })
    
    # Current data with drift in feature1
    cur_data = pd.DataFrame({
        'feature1': np.random.normal(0.5, 1, 1000),  # Mean shifted
        'feature2': np.random.normal(0, 1, 1000)
    })
    
    report = detector.detect_drift(ref_data, cur_data)
    logger.info(f"Drift report: {report}")
    
    logger.info("Drift detector tests completed")
