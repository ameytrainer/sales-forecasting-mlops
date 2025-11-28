"""Pytest Configuration and Fixtures"""
import pytest
import pandas as pd
import sys
from pathlib import Path

# sys.path.insert(0, '/home/ubuntu/sales-forecasting-mlops')
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings

@pytest.fixture
def settings():
    """Get application settings."""
    return get_settings()

@pytest.fixture
def sample_data():
    """Generate sample sales data."""
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'region': ['North'] * 100,
        'product': ['Electronics'] * 100,
        'category': ['Technology'] * 100,
        'price': [299.99] * 100,
        'quantity': [50] * 100,
        'sales': [14999.50] * 100
    })

@pytest.fixture
def sample_features():
    """Generate sample feature data."""
    data = {
        f'feature_{i}': [0.5] * 100 
        for i in range(10)
    }
    return pd.DataFrame(data)
