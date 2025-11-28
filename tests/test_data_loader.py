"""
Test Suite for Data Loader

Tests for DataLoader class from src.data.data_loader module.

Author: Amey Talkatkar
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, '/home/ubuntu/sales-forecasting-mlops')

from src.data import DataLoader


class TestDataLoader:
    """Test suite for DataLoader class."""
    
    @pytest.fixture
    def data_loader(self):
        """Create DataLoader instance."""
        return DataLoader()
    
    @pytest.fixture
    def sample_csv_data(self, tmp_path):
        """Create sample CSV file for testing."""
        data = {
            'date': pd.date_range('2024-01-01', periods=100),
            'region': ['North', 'South', 'East', 'West'] * 25,
            'product': ['Electronics', 'Clothing'] * 50,
            'price': np.random.uniform(10, 1000, 100),
            'quantity': np.random.randint(1, 100, 100),
            'sales': np.random.uniform(100, 10000, 100)
        }
        df = pd.DataFrame(data)
        
        file_path = tmp_path / "test_sales_data.csv"
        df.to_csv(file_path, index=False)
        
        return file_path, df
    
    def test_load_csv_success(self, data_loader, sample_csv_data):
        """Test successful CSV loading."""
        file_path, original_df = sample_csv_data
        
        loaded_df = data_loader.load_csv(file_path)
        
        assert loaded_df is not None
        assert len(loaded_df) == len(original_df)
        assert list(loaded_df.columns) == list(original_df.columns)
        assert loaded_df['date'].dtype == 'datetime64[ns]'
    
    def test_load_csv_file_not_found(self, data_loader):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            data_loader.load_csv("non_existent_file.csv")
    
    def test_load_csv_invalid_format(self, data_loader, tmp_path):
        """Test loading invalid CSV format."""
        # Create malformed CSV
        file_path = tmp_path / "malformed.csv"
        with open(file_path, 'w') as f:
            f.write("invalid,csv,data\n")
            f.write("no,matching,columns,here\n")
        
        # Should not raise error, but return DataFrame with unexpected structure
        df = data_loader.load_csv(file_path)
        assert df is not None
        assert len(df) > 0
    
    def test_load_csv_empty_file(self, data_loader, tmp_path):
        """Test loading empty CSV file."""
        file_path = tmp_path / "empty.csv"
        pd.DataFrame().to_csv(file_path, index=False)
        
        df = data_loader.load_csv(file_path)
        assert len(df) == 0
    
    def test_load_csv_with_missing_values(self, data_loader, tmp_path):
        """Test loading CSV with missing values."""
        data = {
            'date': pd.date_range('2024-01-01', periods=10),
            'region': ['North', None, 'South', 'East', None] * 2,
            'product': ['Electronics'] * 10,
            'price': [100, 200, None, 400, 500] * 2,
            'quantity': [10, 20, 30, None, 50] * 2,
            'sales': [1000] * 10
        }
        df = pd.DataFrame(data)
        
        file_path = tmp_path / "missing_values.csv"
        df.to_csv(file_path, index=False)
        
        loaded_df = data_loader.load_csv(file_path)
        
        assert loaded_df is not None
        assert loaded_df.isnull().sum().sum() > 0  # Has missing values
    
    def test_load_csv_large_file(self, data_loader, tmp_path):
        """Test loading large CSV file."""
        # Create large dataset
        n_rows = 10000
        data = {
            'date': pd.date_range('2024-01-01', periods=n_rows, freq='h'),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_rows),
            'product': np.random.choice(['Electronics', 'Clothing', 'Food'], n_rows),
            'price': np.random.uniform(10, 1000, n_rows),
            'quantity': np.random.randint(1, 100, n_rows),
            'sales': np.random.uniform(100, 10000, n_rows)
        }
        df = pd.DataFrame(data)
        
        file_path = tmp_path / "large_data.csv"
        df.to_csv(file_path, index=False)
        
        loaded_df = data_loader.load_csv(file_path)
        
        assert len(loaded_df) == n_rows
        assert loaded_df.memory_usage(deep=True).sum() > 0
    
    def test_load_csv_date_parsing(self, data_loader, tmp_path):
        """Test date column parsing."""
        data = {
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'region': ['North'] * 3,
            'sales': [1000, 2000, 3000]
        }
        df = pd.DataFrame(data)
        
        file_path = tmp_path / "date_test.csv"
        df.to_csv(file_path, index=False)
        
        loaded_df = data_loader.load_csv(file_path)
        
        assert loaded_df['date'].dtype == 'datetime64[ns]'
        assert loaded_df['date'].min() == pd.Timestamp('2024-01-01')
    
    def test_load_csv_column_names(self, data_loader, sample_csv_data):
        """Test that column names are preserved."""
        file_path, original_df = sample_csv_data
        
        loaded_df = data_loader.load_csv(file_path)
        
        assert set(loaded_df.columns) == set(original_df.columns)
    
    def test_load_csv_data_types(self, data_loader, sample_csv_data):
        """Test that data types are correctly inferred."""
        file_path, _ = sample_csv_data
        
        loaded_df = data_loader.load_csv(file_path)
        
        assert loaded_df['date'].dtype == 'datetime64[ns]'
        assert loaded_df['region'].dtype == 'object'
        assert loaded_df['product'].dtype == 'object'
        assert np.issubdtype(loaded_df['price'].dtype, np.number)
        assert np.issubdtype(loaded_df['quantity'].dtype, np.integer)
        assert np.issubdtype(loaded_df['sales'].dtype, np.number)
    
    def test_load_csv_preserves_order(self, data_loader, sample_csv_data):
        """Test that row order is preserved."""
        file_path, original_df = sample_csv_data
        
        loaded_df = data_loader.load_csv(file_path)
        
        # Compare first and last rows
        pd.testing.assert_series_equal(
            loaded_df.iloc[0][['region', 'product']],
            original_df.iloc[0][['region', 'product']],
            check_names=False
        )
    
    def test_load_csv_encoding(self, data_loader, tmp_path):
        """Test loading CSV with different encodings."""
        data = {
            'date': pd.date_range('2024-01-01', periods=5),
            'region': ['North', 'South', 'East', 'West', 'Central'],
            'product': ['Product_α', 'Product_β', 'Product_γ', 'Product_δ', 'Product_ε'],
            'sales': [1000, 2000, 3000, 4000, 5000]
        }
        df = pd.DataFrame(data)
        
        file_path = tmp_path / "utf8_data.csv"
        df.to_csv(file_path, index=False, encoding='utf-8')
        
        loaded_df = data_loader.load_csv(file_path)
        
        assert loaded_df is not None
        assert len(loaded_df) == 5


class TestDataLoaderEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def data_loader(self):
        """Create DataLoader instance."""
        return DataLoader()
    
    def test_load_csv_special_characters(self, data_loader, tmp_path):
        """Test loading CSV with special characters in data."""
        data = {
            'date': pd.date_range('2024-01-01', periods=3),
            'region': ['North-East', 'South & West', 'Central/East'],
            'product': ['Product (A)', 'Product [B]', 'Product {C}'],
            'sales': [1000, 2000, 3000]
        }
        df = pd.DataFrame(data)
        
        file_path = tmp_path / "special_chars.csv"
        df.to_csv(file_path, index=False)
        
        loaded_df = data_loader.load_csv(file_path)
        
        assert loaded_df is not None
        assert 'North-East' in loaded_df['region'].values
    
    def test_load_csv_duplicate_columns(self, data_loader, tmp_path):
        """Test handling of duplicate column names."""
        # Pandas will auto-rename duplicate columns
        file_path = tmp_path / "duplicate_cols.csv"
        with open(file_path, 'w') as f:
            f.write("date,region,region,sales\n")
            f.write("2024-01-01,North,South,1000\n")
        
        loaded_df = data_loader.load_csv(file_path)
        
        assert loaded_df is not None
        # Check that duplicate columns were handled
        assert len(loaded_df.columns) == len(set(loaded_df.columns)) or \
               'region.1' in loaded_df.columns
    
    def test_load_csv_mixed_types(self, data_loader, tmp_path):
        """Test loading CSV with mixed data types in columns."""
        file_path = tmp_path / "mixed_types.csv"
        with open(file_path, 'w') as f:
            f.write("date,value\n")
            f.write("2024-01-01,100\n")
            f.write("2024-01-02,abc\n")
            f.write("2024-01-03,300\n")
        
        loaded_df = data_loader.load_csv(file_path)
        
        assert loaded_df is not None
        # Mixed types should be loaded as object dtype
        assert loaded_df['value'].dtype == 'object'
    
    def test_load_csv_quoted_fields(self, data_loader, tmp_path):
        """Test loading CSV with quoted fields."""
        file_path = tmp_path / "quoted.csv"
        with open(file_path, 'w') as f:
            f.write('date,region,product,sales\n')
            f.write('2024-01-01,"North","Product, with comma",1000\n')
            f.write('2024-01-02,"South","Product ""with quotes""",2000\n')
        
        loaded_df = data_loader.load_csv(file_path)
        
        assert loaded_df is not None
        assert 'Product, with comma' in loaded_df['product'].values
    
    def test_load_csv_whitespace(self, data_loader, tmp_path):
        """Test handling of whitespace in CSV."""
        file_path = tmp_path / "whitespace.csv"
        with open(file_path, 'w') as f:
            f.write("date,region,sales\n")
            f.write("2024-01-01,  North  ,1000\n")
            f.write("2024-01-02,South,  2000  \n")
        
        loaded_df = data_loader.load_csv(file_path)
        
        assert loaded_df is not None
        # Check if whitespace is preserved or stripped (depends on implementation)
        assert loaded_df['region'].iloc[0].strip() == 'North'
