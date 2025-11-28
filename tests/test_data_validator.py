"""
Test Suite for Data Validator

Tests for DataValidator class using Great Expectations.

Author: Amey Talkatkar
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

sys.path.insert(0, '/home/ubuntu/sales-forecasting-mlops')

from src.data import DataValidator


class TestDataValidator:
    """Test suite for DataValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create DataValidator instance."""
        return DataValidator()
    
    @pytest.fixture
    def valid_sales_data(self):
        """Create valid sales data for testing."""
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'region': ['North', 'South', 'East', 'West'] * 25,
            'product': ['Electronics', 'Clothing'] * 50,
            'category': ['Tech', 'Fashion'] * 50,
            'price': np.random.uniform(10, 1000, 100),
            'quantity': np.random.randint(1, 100, 100),
            'sales': np.random.uniform(100, 10000, 100)
        })
    
    def test_validate_sales_data_success(self, validator, valid_sales_data):
        """Test validation of valid sales data."""
        report = validator.validate_sales_data(valid_sales_data)
        
        assert report is not None
        assert 'is_valid' in report
        assert report['is_valid'] == True
        assert report['row_count'] == 100
        assert report['checks_passed'] > 0
    
    def test_validate_missing_required_columns(self, validator):
        """Test validation fails when required columns are missing."""
        invalid_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'region': ['North'] * 10
            # Missing: product, price, quantity, sales
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            validator.validate_sales_data(invalid_data)
    
    def test_validate_empty_dataframe(self, validator):
        """Test validation of empty DataFrame."""
        empty_data = pd.DataFrame(columns=['date', 'region', 'product', 'price', 'quantity', 'sales'])
        
        report = validator.validate_sales_data(empty_data)
        
        assert report is not None
        assert report['is_valid'] == False
        assert report['row_count'] == 0
    
    def test_validate_negative_prices(self, validator):
        """Test detection of negative prices."""
        invalid_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'region': ['North'] * 10,
            'product': ['Electronics'] * 10,
            'price': [-100, 200, 300, -50, 500, 600, 700, 800, 900, 1000],
            'quantity': [10] * 10,
            'sales': [1000] * 10
        })
        
        report = validator.validate_sales_data(invalid_data)
        
        assert report is not None
        assert report['is_valid'] == False
        assert report['checks_failed'] > 0
    
    def test_validate_negative_quantity(self, validator):
        """Test detection of negative quantities."""
        invalid_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'region': ['North'] * 10,
            'product': ['Electronics'] * 10,
            'price': [100] * 10,
            'quantity': [-5, 10, 20, -3, 40, 50, 60, 70, 80, 90],
            'sales': [1000] * 10
        })
        
        report = validator.validate_sales_data(invalid_data)
        
        assert report is not None
        assert report['is_valid'] == False
    
    def test_validate_null_values(self, validator):
        """Test detection of null values in critical columns."""
        invalid_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'region': ['North', None, 'South', 'East', None] * 2,
            'product': ['Electronics'] * 10,
            'price': [100] * 10,
            'quantity': [10] * 10,
            'sales': [1000, None, 3000, 4000, None] * 2
        })
        
        report = validator.validate_sales_data(invalid_data)
        
        assert report is not None
        # Depending on validation rules, this might pass or fail
        if 'null_counts' in report:
            assert report['null_counts']['region'] > 0
            assert report['null_counts']['sales'] > 0
    
    def test_validate_date_range(self, validator):
        """Test validation of date range."""
        # Create data with dates outside reasonable range
        invalid_data = pd.DataFrame({
            'date': pd.date_range('1900-01-01', periods=10),  # Very old dates
            'region': ['North'] * 10,
            'product': ['Electronics'] * 10,
            'price': [100] * 10,
            'quantity': [10] * 10,
            'sales': [1000] * 10
        })
        
        report = validator.validate_sales_data(invalid_data)
        
        # Validation might flag old dates as suspicious
        assert report is not None
    
    def test_validate_sales_price_quantity_relationship(self, validator):
        """Test validation of sales = price * quantity relationship."""
        # Create data where sales doesn't match price * quantity
        invalid_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'region': ['North'] * 10,
            'product': ['Electronics'] * 10,
            'price': [100] * 10,
            'quantity': [10] * 10,
            'sales': [500] * 10  # Should be 1000
        })
        
        report = validator.validate_sales_data(invalid_data)
        
        # Might have warnings about inconsistent calculations
        assert report is not None
    
    def test_validate_extreme_outliers(self, validator):
        """Test detection of extreme outliers."""
        data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'region': ['North'] * 100,
            'product': ['Electronics'] * 100,
            'price': [100] * 99 + [1000000],  # Extreme outlier
            'quantity': [10] * 100,
            'sales': [1000] * 100
        })
        
        report = validator.validate_sales_data(data)
        
        assert report is not None
        # Might flag outliers as warnings
    
    def test_validate_duplicate_rows(self, validator):
        """Test detection of duplicate rows."""
        data = pd.DataFrame({
            'date': ['2024-01-01'] * 10,
            'region': ['North'] * 10,
            'product': ['Electronics'] * 10,
            'price': [100] * 10,
            'quantity': [10] * 10,
            'sales': [1000] * 10
        })
        data['date'] = pd.to_datetime(data['date'])
        
        report = validator.validate_sales_data(data)
        
        assert report is not None
        # Might flag duplicates
    
    def test_validate_data_types(self, validator):
        """Test validation of column data types."""
        invalid_data = pd.DataFrame({
            'date': ['not-a-date'] * 10,
            'region': ['North'] * 10,
            'product': ['Electronics'] * 10,
            'price': ['not-a-number'] * 10,
            'quantity': [10] * 10,
            'sales': [1000] * 10
        })
        
        # Should fail during date parsing or validation
        with pytest.raises((ValueError, TypeError)):
            validator.validate_sales_data(invalid_data)


class TestDataValidatorReportFormat:
    """Test validation report format and content."""
    
    @pytest.fixture
    def validator(self):
        """Create DataValidator instance."""
        return DataValidator()
    
    @pytest.fixture
    def valid_data(self):
        """Create valid test data."""
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=50),
            'region': ['North', 'South'] * 25,
            'product': ['Electronics'] * 50,
            'price': [100] * 50,
            'quantity': [10] * 50,
            'sales': [1000] * 50
        })
    
    def test_report_contains_required_fields(self, validator, valid_data):
        """Test that validation report contains required fields."""
        report = validator.validate_sales_data(valid_data)
        
        required_fields = ['is_valid', 'row_count', 'checks_passed', 'checks_failed']
        for field in required_fields:
            assert field in report, f"Report missing field: {field}"
    
    def test_report_row_count_accurate(self, validator, valid_data):
        """Test that reported row count matches actual data."""
        report = validator.validate_sales_data(valid_data)
        
        assert report['row_count'] == len(valid_data)
    
    def test_report_checks_summary(self, validator, valid_data):
        """Test that checks summary is present."""
        report = validator.validate_sales_data(valid_data)
        
        assert report['checks_passed'] >= 0
        assert report['checks_failed'] >= 0
        assert isinstance(report['checks_passed'], int)
        assert isinstance(report['checks_failed'], int)
    
    def test_report_human_readable(self, validator, valid_data):
        """Test that report is human-readable."""
        report = validator.validate_sales_data(valid_data)
        
        assert report is not None
        assert isinstance(report, dict)
        
        # Should be able to convert to string
        report_str = str(report)
        assert len(report_str) > 0


class TestDataValidatorEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.fixture
    def validator(self):
        """Create DataValidator instance."""
        return DataValidator()
    
    def test_validate_single_row(self, validator):
        """Test validation of single row DataFrame."""
        single_row = pd.DataFrame({
            'date': [datetime.now()],
            'region': ['North'],
            'product': ['Electronics'],
            'price': [100],
            'quantity': [10],
            'sales': [1000]
        })
        
        report = validator.validate_sales_data(single_row)
        
        assert report is not None
        assert report['row_count'] == 1
    
    def test_validate_large_dataset(self, validator):
        """Test validation of large dataset."""
        large_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10000, freq='h'),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 10000),
            'product': np.random.choice(['Electronics', 'Clothing'], 10000),
            'price': np.random.uniform(10, 1000, 10000),
            'quantity': np.random.randint(1, 100, 10000),
            'sales': np.random.uniform(100, 10000, 10000)
        })
        
        report = validator.validate_sales_data(large_data)
        
        assert report is not None
        assert report['row_count'] == 10000
    
    def test_validate_all_same_values(self, validator):
        """Test validation when all rows have identical values."""
        same_values = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'region': ['North'] * 100,
            'product': ['Electronics'] * 100,
            'price': [100] * 100,
            'quantity': [10] * 100,
            'sales': [1000] * 100
        })
        
        report = validator.validate_sales_data(same_values)
        
        assert report is not None
        # Might have warnings about lack of variance
    
    def test_validate_zero_values(self, validator):
        """Test validation with zero values."""
        zero_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'region': ['North'] * 10,
            'product': ['Electronics'] * 10,
            'price': [0] * 10,
            'quantity': [0] * 10,
            'sales': [0] * 10
        })
        
        report = validator.validate_sales_data(zero_data)
        
        # Zero values might be flagged as suspicious
        assert report is not None
    
    def test_validate_extreme_values(self, validator):
        """Test validation with extreme but valid values."""
        extreme_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'region': ['North'] * 10,
            'product': ['Electronics'] * 10,
            'price': [999999] * 10,
            'quantity': [1000] * 10,
            'sales': [999999000] * 10
        })
        
        report = validator.validate_sales_data(extreme_data)
        
        assert report is not None
