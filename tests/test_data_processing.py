"""Data Processing Tests"""
import pytest
import pandas as pd
import sys
sys.path.insert(0, '/home/ubuntu/sales-forecasting-mlops')

from src.data import DataLoader, DataValidator, DataProcessor

def test_data_loader(sample_data, tmp_path):
    """Test data loading."""
    # Save sample data
    file_path = tmp_path / "test_data.csv"
    sample_data.to_csv(file_path, index=False)
    
    # Load data
    loader = DataLoader()
    df = loader.load_csv(file_path)
    
    assert len(df) == 100
    assert 'date' in df.columns

def test_data_validator(sample_data):
    """Test data validation."""
    validator = DataValidator()
    report = validator.validate_sales_data(sample_data)
    
    assert 'is_valid' in report
    assert 'row_count' in report

def test_data_processor(sample_data):
    """Test data processing."""
    processor = DataProcessor()
    
    X_train, X_test, y_train, y_test = processor.split_data(
        sample_data, target_column='sales', test_size=0.2
    )
    
    assert len(X_train) == 80
    assert len(X_test) == 20
