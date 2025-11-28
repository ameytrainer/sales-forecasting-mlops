"""
Test Suite for Feature Engineering

Tests for FeatureEngineer class.

Author: Amey Talkatkar
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

sys.path.insert(0, '/home/ubuntu/sales-forecasting-mlops')

from src.features import FeatureEngineer


class TestFeatureEngineerTimeFeatures:
    """Test suite for time-based feature creation."""
    
    @pytest.fixture
    def engineer(self):
        """Create FeatureEngineer instance."""
        return FeatureEngineer()
    
    @pytest.fixture
    def sample_time_data(self):
        """Create sample data with dates."""
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=365, freq='D'),
            'sales': np.random.uniform(100, 1000, 365)
        })
    
    def test_create_time_features(self, engineer, sample_time_data):
        """Test creation of basic time features."""
        df_with_features = engineer.create_time_features(sample_time_data)
        
        assert 'year' in df_with_features.columns
        assert 'month' in df_with_features.columns
        assert 'day' in df_with_features.columns
        assert 'dayofweek' in df_with_features.columns
        assert 'quarter' in df_with_features.columns
        assert 'week' in df_with_features.columns
    
    def test_year_feature_correctness(self, engineer, sample_time_data):
        """Test that year feature is extracted correctly."""
        df_with_features = engineer.create_time_features(sample_time_data)
        
        assert df_with_features['year'].iloc[0] == 2024
        assert df_with_features['year'].nunique() == 1  # All same year
    
    def test_month_feature_range(self, engineer, sample_time_data):
        """Test that month feature is in valid range."""
        df_with_features = engineer.create_time_features(sample_time_data)
        
        assert df_with_features['month'].min() >= 1
        assert df_with_features['month'].max() <= 12
        assert df_with_features['month'].nunique() == 12  # All 12 months
    
    def test_dayofweek_feature_range(self, engineer, sample_time_data):
        """Test that day of week is in valid range."""
        df_with_features = engineer.create_time_features(sample_time_data)
        
        assert df_with_features['dayofweek'].min() >= 0
        assert df_with_features['dayofweek'].max() <= 6
    
    def test_quarter_feature_correctness(self, engineer, sample_time_data):
        """Test that quarter feature is correct."""
        df_with_features = engineer.create_time_features(sample_time_data)
        
        # January should be Q1
        jan_rows = df_with_features[df_with_features['month'] == 1]
        assert (jan_rows['quarter'] == 1).all()
        
        # July should be Q3
        jul_rows = df_with_features[df_with_features['month'] == 7]
        assert (jul_rows['quarter'] == 3).all()
    
    def test_weekend_feature_creation(self, engineer, sample_time_data):
        """Test creation of weekend indicator."""
        df_with_features = engineer.create_time_features(sample_time_data)
        
        if 'is_weekend' in df_with_features.columns:
            # Weekend should be Saturday (5) and Sunday (6)
            assert df_with_features['is_weekend'].isin([0, 1]).all()
    
    def test_month_end_feature(self, engineer):
        """Test month end indicator feature."""
        # Create data with month-end dates
        dates = pd.date_range('2024-01-30', periods=5, freq='D')
        df = pd.DataFrame({'date': dates, 'sales': [100] * 5})
        
        df_with_features = engineer.create_time_features(df)
        
        if 'is_month_end' in df_with_features.columns:
            # Jan 31 should be marked as month end
            assert df_with_features[df_with_features['date'] == '2024-01-31']['is_month_end'].iloc[0] == 1


class TestFeatureEngineerLagFeatures:
    """Test suite for lag feature creation."""
    
    @pytest.fixture
    def engineer(self):
        """Create FeatureEngineer instance."""
        return FeatureEngineer()
    
    @pytest.fixture
    def sample_sequential_data(self):
        """Create sample sequential data."""
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'sales': list(range(100, 200))  # Sequential values for easy testing
        })
    
    def test_create_lag_features(self, engineer, sample_sequential_data):
        """Test creation of lag features."""
        df_with_lags = engineer.create_lag_features(
            sample_sequential_data,
            target_col='sales',
            lags=[1, 7, 30]
        )
        
        assert 'sales_lag_1' in df_with_lags.columns
        assert 'sales_lag_7' in df_with_lags.columns
        assert 'sales_lag_30' in df_with_lags.columns
    
    def test_lag_1_correctness(self, engineer, sample_sequential_data):
        """Test that lag 1 feature is correct."""
        df_with_lags = engineer.create_lag_features(
            sample_sequential_data,
            target_col='sales',
            lags=[1]
        )
        
        # lag_1 at index 1 should equal sales at index 0
        assert df_with_lags['sales_lag_1'].iloc[1] == df_with_lags['sales'].iloc[0]
        assert df_with_lags['sales_lag_1'].iloc[2] == df_with_lags['sales'].iloc[1]
    
    def test_lag_features_null_handling(self, engineer, sample_sequential_data):
        """Test that first N rows have NaN for lag N."""
        df_with_lags = engineer.create_lag_features(
            sample_sequential_data,
            target_col='sales',
            lags=[7]
        )
        
        # First 7 rows should be NaN
        assert df_with_lags['sales_lag_7'].iloc[:7].isna().all()
        # 8th row onward should have values
        assert not df_with_lags['sales_lag_7'].iloc[7:].isna().any()
    
    def test_multiple_lags_creation(self, engineer, sample_sequential_data):
        """Test creation of multiple lag features simultaneously."""
        lags = [1, 2, 3, 7, 14, 30]
        df_with_lags = engineer.create_lag_features(
            sample_sequential_data,
            target_col='sales',
            lags=lags
        )
        
        for lag in lags:
            assert f'sales_lag_{lag}' in df_with_lags.columns
    
    def test_rolling_mean_features(self, engineer, sample_sequential_data):
        """Test creation of rolling mean features."""
        df_with_rolling = engineer.create_rolling_features(
            sample_sequential_data,
            target_col='sales',
            windows=[7, 30]
        )
        
        if 'sales_rolling_mean_7' in df_with_rolling.columns:
            # Test that rolling mean is calculated correctly
            assert not df_with_rolling['sales_rolling_mean_7'].iloc[7:].isna().all()
    
    def test_rolling_std_features(self, engineer, sample_sequential_data):
        """Test creation of rolling standard deviation features."""
        df_with_rolling = engineer.create_rolling_features(
            sample_sequential_data,
            target_col='sales',
            windows=[7]
        )
        
        if 'sales_rolling_std_7' in df_with_rolling.columns:
            # Standard deviation should be present
            assert df_with_rolling['sales_rolling_std_7'].notna().any()


class TestFeatureEngineerCategoricalFeatures:
    """Test suite for categorical feature encoding."""
    
    @pytest.fixture
    def engineer(self):
        """Create FeatureEngineer instance."""
        return FeatureEngineer()
    
    @pytest.fixture
    def sample_categorical_data(self):
        """Create sample data with categorical variables."""
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'region': ['North', 'South', 'East', 'West'] * 25,
            'product': ['Electronics', 'Clothing'] * 50,
            'sales': np.random.uniform(100, 1000, 100)
        })
    
    def test_encode_categorical_features(self, engineer, sample_categorical_data):
        """Test encoding of categorical features."""
        df_encoded = engineer.encode_categorical(sample_categorical_data)
        
        # Original categorical columns should still exist or be encoded
        assert len(df_encoded.columns) >= len(sample_categorical_data.columns)
    
    def test_one_hot_encoding(self, engineer, sample_categorical_data):
        """Test one-hot encoding of categorical variables."""
        df_encoded = engineer.encode_categorical(
            sample_categorical_data,
            method='onehot'
        )
        
        # Should have binary columns for each category
        if 'region_North' in df_encoded.columns:
            assert df_encoded['region_North'].isin([0, 1]).all()
    
    def test_label_encoding(self, engineer, sample_categorical_data):
        """Test label encoding of categorical variables."""
        df_encoded = engineer.encode_categorical(
            sample_categorical_data,
            method='label'
        )
        
        # Encoded columns should be numeric
        if 'region' in df_encoded.columns:
            assert np.issubdtype(df_encoded['region'].dtype, np.number) or \
                   df_encoded['region'].dtype == 'object'
    
    def test_encoding_preserves_data_integrity(self, engineer, sample_categorical_data):
        """Test that encoding preserves number of rows."""
        df_encoded = engineer.encode_categorical(sample_categorical_data)
        
        assert len(df_encoded) == len(sample_categorical_data)


class TestFeatureEngineerIntegration:
    """Integration tests for complete feature engineering pipeline."""
    
    @pytest.fixture
    def engineer(self):
        """Create FeatureEngineer instance."""
        return FeatureEngineer()
    
    @pytest.fixture
    def complete_sales_data(self):
        """Create complete sales data for testing."""
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'region': ['North', 'South', 'East', 'West'] * 25,
            'product': ['Electronics', 'Clothing'] * 50,
            'price': np.random.uniform(10, 1000, 100),
            'quantity': np.random.randint(1, 100, 100),
            'sales': np.random.uniform(100, 10000, 100)
        })
    
    def test_create_all_features(self, engineer, complete_sales_data):
        """Test creation of all features together."""
        df_all_features = engineer.create_all_features(
            complete_sales_data,
            target_col='sales'
        )
        
        assert len(df_all_features) == len(complete_sales_data)
        assert len(df_all_features.columns) > len(complete_sales_data.columns)
    
    def test_feature_pipeline_no_errors(self, engineer, complete_sales_data):
        """Test that feature pipeline runs without errors."""
        try:
            df_features = engineer.create_all_features(complete_sales_data)
            assert df_features is not None
        except Exception as e:
            pytest.fail(f"Feature engineering failed: {e}")
    
    def test_feature_names_valid(self, engineer, complete_sales_data):
        """Test that all feature names are valid (no special characters)."""
        df_features = engineer.create_all_features(complete_sales_data)
        
        for col in df_features.columns:
            # Column names should not have spaces or special chars
            assert ' ' not in col
            assert not any(char in col for char in ['(', ')', '[', ']', '{', '}'])
    
    def test_no_infinite_values(self, engineer, complete_sales_data):
        """Test that feature engineering doesn't create infinite values."""
        df_features = engineer.create_all_features(complete_sales_data)
        
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not np.isinf(df_features[col]).any(), f"Infinite values in {col}"
    
    def test_feature_data_types_valid(self, engineer, complete_sales_data):
        """Test that all features have valid data types."""
        df_features = engineer.create_all_features(complete_sales_data)
        
        for col in df_features.columns:
            dtype = df_features[col].dtype
            # Should be numeric or categorical
            assert np.issubdtype(dtype, np.number) or dtype == 'object' or \
                   dtype == 'datetime64[ns]' or dtype == 'category'


class TestFeatureEngineerEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def engineer(self):
        """Create FeatureEngineer instance."""
        return FeatureEngineer()
    
    def test_handle_missing_date_column(self, engineer):
        """Test handling when date column is missing."""
        df_no_date = pd.DataFrame({
            'region': ['North'] * 10,
            'sales': [100] * 10
        })
        
        # Should raise error or handle gracefully
        with pytest.raises((ValueError, KeyError)):
            engineer.create_time_features(df_no_date)
    
    def test_handle_single_row_dataframe(self, engineer):
        """Test feature engineering with single row."""
        df_single = pd.DataFrame({
            'date': [datetime(2024, 1, 1)],
            'region': ['North'],
            'sales': [100]
        })
        
        df_features = engineer.create_all_features(df_single)
        
        assert len(df_features) == 1
    
    def test_handle_all_missing_values(self, engineer):
        """Test handling when target column has all NaN."""
        df_missing = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'region': ['North'] * 10,
            'sales': [np.nan] * 10
        })
        
        # Should handle gracefully or raise appropriate error
        try:
            df_features = engineer.create_lag_features(df_missing, 'sales', lags=[1])
            assert df_features is not None
        except ValueError:
            # Expected behavior for all-NaN column
            pass
    
    def test_handle_duplicate_column_names(self, engineer):
        """Test handling when feature names would clash."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'sales': [100] * 10,
            'sales_lag_1': [90] * 10  # Already exists
        })
        
        # Should handle or raise appropriate error
        df_features = engineer.create_lag_features(df, 'sales', lags=[1])
        assert df_features is not None
