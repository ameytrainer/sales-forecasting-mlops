"""
Feature Engineering Module

Creates features for sales forecasting:
- Lag features (historical values)
- Rolling window statistics
- Time-based features (day, month, seasonality)
- Interaction features

Author: Amey Talkatkar
"""

from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from datetime import datetime

from src.config import get_settings
from src.utils import setup_logging, timer, memory_monitor


logger = setup_logging(__name__)


class FeatureEngineer:
    """
    Create features for time series forecasting.
    
    Examples:
        >>> engineer = FeatureEngineer()
        >>> df_features = engineer.create_all_features(df)
    """
    
    def __init__(self):
        """Initialize feature engineer with settings."""
        self.settings = get_settings()
        self.feature_names = []
        logger.info("FeatureEngineer initialized")
    
    @timer
    @memory_monitor
    def create_all_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'sales',
        group_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create all features in one go.
        
        Args:
            df: Input DataFrame with date column
            target_col: Target column for lag features
            group_cols: Columns to group by for lag/rolling features
            
        Returns:
            DataFrame with all features
            
        Example:
            >>> df_features = engineer.create_all_features(df)
        """
        logger.info("Creating all features...")
        
        df = df.copy()
        group_cols = group_cols or ['region', 'product']
        
        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        
        # Create time-based features
        df = self.create_time_features(df)
        
        # Create lag features
        df = self.create_lag_features(df, target_col, group_cols)
        
        # Create rolling window features
        df = self.create_rolling_features(df, target_col, group_cols)
        
        # Create interaction features
        df = self.create_interaction_features(df)
        
        # Store feature names
        self.feature_names = [col for col in df.columns if col not in ['date', target_col]]
        
        logger.info(f"Created {len(self.feature_names)} features")
        logger.info(f"Features: {self.feature_names}")
        
        return df
    
    @timer
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from date column.
        
        Args:
            df: Input DataFrame with date column
            
        Returns:
            DataFrame with time features
            
        Features created:
        - year, month, day, day_of_week, day_of_year
        - is_weekend, is_month_start, is_month_end, is_quarter_start, is_quarter_end
        - week_of_year, quarter
        - season (Winter, Spring, Summer, Fall)
        
        Example:
            >>> df = engineer.create_time_features(df)
        """
        logger.info("Creating time-based features...")
        
        df = df.copy()
        
        if 'date' not in df.columns:
            logger.warning("No date column found, skipping time features")
            return df
        
        date_col = pd.to_datetime(df['date'])
        
        # Basic time features
        df['year'] = date_col.dt.year
        df['month'] = date_col.dt.month
        df['day'] = date_col.dt.day
        df['day_of_week'] = date_col.dt.dayofweek
        df['day_of_year'] = date_col.dt.dayofyear
        df['week_of_year'] = date_col.dt.isocalendar().week
        df['quarter'] = date_col.dt.quarter
        
        # Boolean features
        df['is_weekend'] = (date_col.dt.dayofweek >= 5).astype(int)
        df['is_month_start'] = date_col.dt.is_month_start.astype(int)
        df['is_month_end'] = date_col.dt.is_month_end.astype(int)
        df['is_quarter_start'] = date_col.dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = date_col.dt.is_quarter_end.astype(int)
        
        # Season (meteorological)
        df['season'] = date_col.dt.month % 12 // 3 + 1
        season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
        df['season'] = df['season'].map(season_map)
        
        # Cyclical encoding for month and day_of_week
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        logger.info("Created 17 time-based features")
        
        return df
    
    @timer
    def create_lag_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        group_cols: List[str],
        lags: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Create lag features (historical values).
        
        Args:
            df: Input DataFrame
            target_col: Column to create lags for
            group_cols: Columns to group by
            lags: List of lag periods (default from settings)
            
        Returns:
            DataFrame with lag features
            
        Example:
            >>> df = engineer.create_lag_features(df, 'sales', ['region', 'product'])
        """
        logger.info(f"Creating lag features for {target_col}...")
        
        df = df.copy()
        lags = lags or self.settings.lag_features
        
        # Sort by date within groups
        df = df.sort_values(['date'] + group_cols)
        
        # Create lag features
        for lag in lags:
            feature_name = f'{target_col}_lag_{lag}'
            df[feature_name] = df.groupby(group_cols)[target_col].shift(lag)
            logger.debug(f"Created {feature_name}")
        
        logger.info(f"Created {len(lags)} lag features")
        
        return df
    
    @timer
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        group_cols: List[str],
        windows: Optional[List[int]] = None,
        aggregations: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create rolling window features.
        
        Args:
            df: Input DataFrame
            target_col: Column to create rolling features for
            group_cols: Columns to group by
            windows: List of window sizes (default from settings)
            aggregations: List of aggregation functions
            
        Returns:
            DataFrame with rolling features
            
        Example:
            >>> df = engineer.create_rolling_features(df, 'sales', ['region'])
        """
        logger.info(f"Creating rolling window features for {target_col}...")
        
        df = df.copy()
        windows = windows or self.settings.rolling_windows
        aggregations = aggregations or ['mean', 'std', 'min', 'max']
        
        # Sort by date within groups
        df = df.sort_values(['date'] + group_cols)
        
        # Create rolling features
        for window in windows:
            for agg in aggregations:
                feature_name = f'{target_col}_rolling_{window}_{agg}'
                df[feature_name] = df.groupby(group_cols)[target_col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).agg(agg)
                )
                logger.debug(f"Created {feature_name}")
        
        logger.info(f"Created {len(windows) * len(aggregations)} rolling features")
        
        return df
    
    @timer
    def create_interaction_features(
        self,
        df: pd.DataFrame,
        numeric_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create interaction features between numeric columns.
        
        Args:
            df: Input DataFrame
            numeric_cols: Columns to create interactions for
            
        Returns:
            DataFrame with interaction features
            
        Example:
            >>> df = engineer.create_interaction_features(df, ['price', 'quantity'])
        """
        logger.info("Creating interaction features...")
        
        df = df.copy()
        
        # Auto-detect numeric columns if not specified
        if numeric_cols is None:
            numeric_cols = ['price', 'quantity']
            numeric_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(numeric_cols) < 2:
            logger.warning("Need at least 2 numeric columns for interactions")
            return df
        
        # Create interactions
        interaction_count = 0
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                # Multiplication
                feature_name = f'{col1}_x_{col2}'
                df[feature_name] = df[col1] * df[col2]
                interaction_count += 1
                
                # Division (with safe handling)
                if (df[col2] != 0).all():
                    feature_name = f'{col1}_div_{col2}'
                    df[feature_name] = df[col1] / df[col2]
                    interaction_count += 1
        
        logger.info(f"Created {interaction_count} interaction features")
        
        return df
    
    @timer
    def create_aggregate_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        group_cols: List[str],
        aggregations: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create aggregate features by group.
        
        Args:
            df: Input DataFrame
            target_col: Column to aggregate
            group_cols: Columns to group by
            aggregations: Aggregation functions
            
        Returns:
            DataFrame with aggregate features
            
        Example:
            >>> df = engineer.create_aggregate_features(df, 'sales', ['region'])
        """
        logger.info(f"Creating aggregate features for {target_col}...")
        
        df = df.copy()
        aggregations = aggregations or ['mean', 'sum', 'std', 'min', 'max', 'count']
        
        for col in group_cols:
            for agg in aggregations:
                feature_name = f'{target_col}_{col}_{agg}'
                df[feature_name] = df.groupby(col)[target_col].transform(agg)
                logger.debug(f"Created {feature_name}")
        
        logger.info(f"Created {len(group_cols) * len(aggregations)} aggregate features")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of created feature names.
        
        Returns:
            List of feature names
        """
        return self.feature_names
    
    def get_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics for all features.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Summary DataFrame
        """
        numeric_features = df.select_dtypes(include=[np.number]).columns
        
        summary = df[numeric_features].describe().T
        summary['missing'] = df[numeric_features].isnull().sum()
        summary['missing_pct'] = (summary['missing'] / len(df) * 100).round(2)
        
        return summary


if __name__ == "__main__":
    # Test feature engineer
    engineer = FeatureEngineer()
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'date': dates,
        'region': np.random.choice(['North', 'South'], 100),
        'product': np.random.choice(['A', 'B'], 100),
        'price': np.random.uniform(10, 500, 100),
        'quantity': np.random.randint(1, 100, 100),
        'sales': np.random.uniform(100, 5000, 100),
    })
    
    logger.info("Testing feature creation...")
    df_features = engineer.create_all_features(sample_data)
    
    logger.info(f"Original shape: {sample_data.shape}")
    logger.info(f"Features shape: {df_features.shape}")
    logger.info(f"Created {len(engineer.get_feature_names())} features")
    
    # Get feature summary
    summary = engineer.get_feature_summary(df_features)
    logger.info(f"\nFeature summary:\n{summary}")
    
    logger.info("Feature engineer tests completed")
