"""
Feature Selection Module

Select important features using various methods:
- Correlation-based selection
- Variance threshold
- Recursive feature elimination (RFE)
- Feature importance from tree models
- Statistical tests

Author: Amey Talkatkar
"""

from typing import List, Optional, Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_regression,
    mutual_info_regression,
    RFE
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.config import get_settings
from src.utils import setup_logging, timer


logger = setup_logging(__name__)


class FeatureSelector:
    """
    Select important features using various methods.
    
    Examples:
        >>> selector = FeatureSelector()
        >>> selected_features = selector.select_by_importance(X_train, y_train, k=20)
    """
    
    def __init__(self):
        """Initialize feature selector with settings."""
        self.settings = get_settings()
        self.selected_features = []
        logger.info("FeatureSelector initialized")
    
    @timer
    def select_by_variance(
        self,
        X: pd.DataFrame,
        threshold: float = 0.01
    ) -> List[str]:
        """
        Select features with variance above threshold.
        
        Args:
            X: Features DataFrame
            threshold: Variance threshold
            
        Returns:
            List of selected feature names
            
        Example:
            >>> features = selector.select_by_variance(X_train, threshold=0.01)
        """
        logger.info(f"Selecting features by variance (threshold={threshold})...")
        
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        
        selected_features = X.columns[selector.get_support()].tolist()
        removed_features = X.columns[~selector.get_support()].tolist()
        
        logger.info(f"Selected {len(selected_features)} features")
        logger.info(f"Removed {len(removed_features)} low-variance features")
        
        if removed_features:
            logger.debug(f"Removed features: {removed_features}")
        
        return selected_features
    
    @timer
    def select_by_correlation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.1,
        method: str = 'pearson'
    ) -> List[str]:
        """
        Select features based on correlation with target.
        
        Args:
            X: Features DataFrame
            y: Target Series
            threshold: Minimum absolute correlation
            method: Correlation method ('pearson', 'spearman')
            
        Returns:
            List of selected feature names
            
        Example:
            >>> features = selector.select_by_correlation(X_train, y_train)
        """
        logger.info(f"Selecting features by correlation (threshold={threshold})...")
        
        # Calculate correlations
        correlations = pd.DataFrame({
            'feature': X.columns,
            'correlation': [X[col].corr(y, method=method) for col in X.columns]
        })
        
        # Select features with absolute correlation above threshold
        correlations['abs_correlation'] = correlations['correlation'].abs()
        selected = correlations[correlations['abs_correlation'] >= threshold]
        
        # Sort by absolute correlation
        selected = selected.sort_values('abs_correlation', ascending=False)
        
        logger.info(f"Selected {len(selected)} features")
        logger.info(f"Top 5 correlations:\n{selected.head()}")
        
        return selected['feature'].tolist()
    
    @timer
    def select_by_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: int = 20,
        model_type: str = 'rf'
    ) -> List[str]:
        """
        Select top k features by importance from tree model.
        
        Args:
            X: Features DataFrame
            y: Target Series
            k: Number of features to select
            model_type: Model type ('rf' or 'xgb')
            
        Returns:
            List of selected feature names
            
        Example:
            >>> features = selector.select_by_importance(X_train, y_train, k=20)
        """
        logger.info(f"Selecting top {k} features by importance ({model_type})...")
        
        # Train model
        if model_type == 'rf':
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=self.settings.model_random_state,
                n_jobs=-1
            )
        elif model_type == 'xgb':
            import xgboost as xgb
            model = xgb.XGBRegressor(
                n_estimators=100,
                random_state=self.settings.model_random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X, y)
        
        # Get feature importances
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select top k features
        selected_features = importances.head(k)['feature'].tolist()
        
        logger.info(f"Selected {len(selected_features)} features")
        logger.info(f"Top 5 important features:\n{importances.head()}")
        
        return selected_features
    
    @timer
    def select_by_rfe(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 20
    ) -> List[str]:
        """
        Select features using Recursive Feature Elimination.
        
        Args:
            X: Features DataFrame
            y: Target Series
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
            
        Example:
            >>> features = selector.select_by_rfe(X_train, y_train, n_features=20)
        """
        logger.info(f"Selecting {n_features} features using RFE...")
        
        # Use Linear Regression as estimator
        estimator = LinearRegression()
        
        # Apply RFE
        selector = RFE(
            estimator=estimator,
            n_features_to_select=n_features,
            step=1
        )
        
        selector.fit(X, y)
        
        # Get selected features
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Get rankings
        rankings = pd.DataFrame({
            'feature': X.columns,
            'ranking': selector.ranking_
        }).sort_values('ranking')
        
        logger.info(f"Selected {len(selected_features)} features")
        logger.info(f"Top features by ranking:\n{rankings.head()}")
        
        return selected_features
    
    @timer
    def select_by_statistical_test(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: int = 20,
        score_func: str = 'f_regression'
    ) -> List[str]:
        """
        Select features using statistical tests.
        
        Args:
            X: Features DataFrame
            y: Target Series
            k: Number of features to select
            score_func: Scoring function ('f_regression', 'mutual_info')
            
        Returns:
            List of selected feature names
            
        Example:
            >>> features = selector.select_by_statistical_test(X_train, y_train)
        """
        logger.info(f"Selecting {k} features using {score_func}...")
        
        # Choose scoring function
        if score_func == 'f_regression':
            func = f_regression
        elif score_func == 'mutual_info':
            func = mutual_info_regression
        else:
            raise ValueError(f"Unknown score function: {score_func}")
        
        # Apply selection
        selector = SelectKBest(score_func=func, k=k)
        selector.fit(X, y)
        
        # Get selected features
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Get scores
        scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        logger.info(f"Selected {len(selected_features)} features")
        logger.info(f"Top features by score:\n{scores.head()}")
        
        return selected_features
    
    @timer
    def remove_correlated_features(
        self,
        X: pd.DataFrame,
        threshold: float = 0.95
    ) -> List[str]:
        """
        Remove highly correlated features.
        
        Args:
            X: Features DataFrame
            threshold: Correlation threshold for removal
            
        Returns:
            List of features to keep
            
        Example:
            >>> features = selector.remove_correlated_features(X_train)
        """
        logger.info(f"Removing correlated features (threshold={threshold})...")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Get upper triangle
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation above threshold
        to_drop = [
            column for column in upper.columns
            if any(upper[column] > threshold)
        ]
        
        # Features to keep
        features_to_keep = [col for col in X.columns if col not in to_drop]
        
        logger.info(f"Removed {len(to_drop)} correlated features")
        logger.info(f"Kept {len(features_to_keep)} features")
        
        if to_drop:
            logger.debug(f"Removed features: {to_drop}")
        
        return features_to_keep
    
    @timer
    def select_best_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: int = 20,
        methods: Optional[List[str]] = None
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Select features using multiple methods and get consensus.
        
        Args:
            X: Features DataFrame
            y: Target Series
            k: Number of features to select
            methods: List of methods to use
            
        Returns:
            Tuple of (consensus features, all method results)
            
        Example:
            >>> features, results = selector.select_best_features(X_train, y_train)
        """
        logger.info(f"Selecting best {k} features using multiple methods...")
        
        methods = methods or ['importance', 'correlation', 'statistical']
        
        results = {}
        
        # Apply each method
        if 'importance' in methods:
            results['importance'] = self.select_by_importance(X, y, k=k)
        
        if 'correlation' in methods:
            results['correlation'] = self.select_by_correlation(X, y, threshold=0.05)[:k]
        
        if 'statistical' in methods:
            results['statistical'] = self.select_by_statistical_test(X, y, k=k)
        
        if 'rfe' in methods:
            results['rfe'] = self.select_by_rfe(X, y, n_features=k)
        
        # Get consensus (features selected by multiple methods)
        all_features = []
        for features in results.values():
            all_features.extend(features)
        
        # Count occurrences
        feature_counts = pd.Series(all_features).value_counts()
        
        # Select features that appear in at least 2 methods
        min_votes = max(2, len(methods) // 2)
        consensus_features = feature_counts[feature_counts >= min_votes].index.tolist()
        
        # If not enough consensus, take top k by votes
        if len(consensus_features) < k:
            consensus_features = feature_counts.head(k).index.tolist()
        else:
            consensus_features = consensus_features[:k]
        
        logger.info(f"Consensus features: {len(consensus_features)}")
        logger.info(f"Feature vote counts:\n{feature_counts.head(10)}")
        
        self.selected_features = consensus_features
        
        return consensus_features, results
    
    def get_selected_features(self) -> List[str]:
        """Get list of selected features."""
        return self.selected_features
    
    def get_feature_importance_plot_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get data for feature importance plotting.
        
        Args:
            X: Features DataFrame
            y: Target Series
            top_n: Number of top features
            
        Returns:
            DataFrame with feature importance data
        """
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=self.settings.model_random_state,
            n_jobs=-1
        )
        model.fit(X, y)
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df


if __name__ == "__main__":
    # Test feature selector
    selector = FeatureSelector()
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    X = pd.DataFrame({
        f'feature_{i}': np.random.randn(n_samples)
        for i in range(30)
    })
    
    # Create target with some correlation to features
    y = pd.Series(
        2 * X['feature_0'] + 
        1.5 * X['feature_1'] - 
        X['feature_2'] + 
        np.random.randn(n_samples) * 0.1
    )
    
    logger.info("Testing feature selection methods...")
    
    # Test different methods
    features_variance = selector.select_by_variance(X, threshold=0.1)
    logger.info(f"Variance selection: {len(features_variance)} features")
    
    features_correlation = selector.select_by_correlation(X, y, threshold=0.1)
    logger.info(f"Correlation selection: {len(features_correlation)} features")
    
    features_importance = selector.select_by_importance(X, y, k=10)
    logger.info(f"Importance selection: {len(features_importance)} features")
    
    # Test consensus selection
    consensus_features, all_results = selector.select_best_features(X, y, k=10)
    logger.info(f"Consensus selection: {len(consensus_features)} features")
    
    logger.info("Feature selector tests completed")
