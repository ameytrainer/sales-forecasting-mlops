"""
Data Loader Module

Handles loading data from various sources:
- Local CSV files
- DVC-tracked data
- Databases
- Remote storage

Author: Amey Talkatkar
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import subprocess

from src.config import get_settings
from src.utils import (
    setup_logging,
    timer,
    memory_monitor,
    load_data_chunked,
    validate_dataframe,
    run_command
)


logger = setup_logging(__name__)


class DataLoader:
    """
    Load data from various sources with DVC integration.
    
    Examples:
        >>> loader = DataLoader()
        >>> df = loader.load_csv("data/raw/sales_data.csv")
        >>> df = loader.load_from_dvc("data/raw/sales_data.csv")
    """
    
    def __init__(self):
        """Initialize data loader with settings."""
        self.settings = get_settings()
        logger.info("DataLoader initialized")
    
    @timer
    @memory_monitor
    def load_csv(
        self,
        filepath: Path,
        use_chunking: bool = True,
        validate: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            use_chunking: Use chunked reading for large files
            validate: Run validation checks
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If validation fails
            
        Example:
            >>> df = loader.load_csv("data/raw/sales_data.csv")
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading CSV from {filepath}")
        
        # Check file size
        size_mb = filepath.stat().st_size / 1024 / 1024
        logger.info(f"File size: {size_mb:.2f} MB")
        
        # Load data
        if use_chunking and size_mb > 10:
            logger.info("Using chunked loading for large file")
            df = load_data_chunked(filepath, **kwargs)
        else:
            df = pd.read_csv(filepath, **kwargs)
        
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Validate if requested
        if validate:
            self._validate_basic(df)
        
        return df
    
    @timer
    def load_from_dvc(
        self,
        filepath: Path,
        pull: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data tracked by DVC.
        
        Args:
            filepath: Path to DVC-tracked file
            pull: Pull from DVC remote before loading
            **kwargs: Additional arguments for load_csv
            
        Returns:
            Loaded DataFrame
            
        Raises:
            RuntimeError: If DVC pull fails
            
        Example:
            >>> df = loader.load_from_dvc("data/raw/sales_data.csv")
        """
        filepath = Path(filepath)
        dvc_file = filepath.with_suffix(filepath.suffix + '.dvc')
        
        # Check if DVC file exists
        if not dvc_file.exists():
            logger.warning(f"DVC file not found: {dvc_file}")
            logger.info("Attempting to load file directly")
        elif pull:
            # Pull from DVC remote
            logger.info(f"Pulling data from DVC: {filepath}")
            result = run_command(
                ["dvc", "pull", str(dvc_file)],
                cwd=self.settings.project_root
            )
            
            if result["returncode"] != 0:
                raise RuntimeError(f"DVC pull failed: {result['stderr']}")
            
            logger.info("DVC pull successful")
        
        # Load the actual data file
        return self.load_csv(filepath, **kwargs)
    
    @timer
    def load_processed_data(
        self,
        feature_name: str = "features",
        **kwargs
    ) -> pd.DataFrame:
        """
        Load processed/feature-engineered data.
        
        Args:
            feature_name: Name of processed dataset
            **kwargs: Additional arguments for load_csv
            
        Returns:
            Processed DataFrame
            
        Example:
            >>> df = loader.load_processed_data("features")
        """
        filepath = self.settings.processed_data_dir / f"{feature_name}.csv"
        logger.info(f"Loading processed data: {feature_name}")
        return self.load_csv(filepath, validate=False, **kwargs)
    
    @timer
    def load_train_test_split(
        self,
        prefix: str = ""
    ) -> Dict[str, pd.DataFrame]:
        """
        Load train/test split data.
        
        Args:
            prefix: Optional prefix for filenames
            
        Returns:
            Dictionary with keys: X_train, X_test, y_train, y_test
            
        Example:
            >>> data = loader.load_train_test_split()
            >>> X_train = data['X_train']
        """
        logger.info("Loading train/test split data")
        
        data = {}
        for name in ['X_train', 'X_test', 'y_train', 'y_test']:
            filename = f"{prefix}{name}.csv" if prefix else f"{name}.csv"
            filepath = self.settings.processed_data_dir / filename
            
            if not filepath.exists():
                raise FileNotFoundError(f"Split file not found: {filepath}")
            
            data[name] = pd.read_csv(filepath)
            logger.info(f"Loaded {name}: {data[name].shape}")
        
        return data
    
    @timer
    def load_predictions(
        self,
        prediction_file: str = "latest.csv"
    ) -> pd.DataFrame:
        """
        Load prediction results.
        
        Args:
            prediction_file: Name of prediction file
            
        Returns:
            Predictions DataFrame
            
        Example:
            >>> preds = loader.load_predictions("latest.csv")
        """
        filepath = self.settings.predictions_dir / prediction_file
        logger.info(f"Loading predictions from {filepath}")
        return self.load_csv(filepath, validate=False)
    
    def _validate_basic(self, df: pd.DataFrame) -> None:
        """
        Run basic validation checks on DataFrame.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If validation fails
        """
        logger.info("Running basic validation checks")
        
        # Check for empty DataFrame
        if len(df) == 0:
            raise ValueError("DataFrame is empty")
        
        # Check for duplicate columns
        duplicates = df.columns[df.columns.duplicated()].tolist()
        if duplicates:
            raise ValueError(f"Duplicate columns found: {duplicates}")
        
        # Check memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        logger.info(f"DataFrame memory usage: {memory_mb:.2f} MB")
        
        if memory_mb > self.settings.max_memory_mb:
            logger.warning(
                f"DataFrame uses {memory_mb:.2f} MB, "
                f"exceeds limit of {self.settings.max_memory_mb} MB"
            )
        
        # Log data info
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Missing values: {df.isnull().sum().sum()}")
        
        # Check for high null percentage
        null_pct = (df.isnull().sum() / len(df) * 100)
        high_null_cols = null_pct[null_pct > 50].index.tolist()
        if high_null_cols:
            logger.warning(f"Columns with >50% null values: {high_null_cols}")
    
    def get_data_info(self, filepath: Path) -> Dict[str, Any]:
        """
        Get information about a data file without loading it.
        
        Args:
            filepath: Path to data file
            
        Returns:
            Dictionary with file information
            
        Example:
            >>> info = loader.get_data_info("data/raw/sales_data.csv")
            >>> print(f"Rows: {info['row_count']}")
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Get basic file info
        size_bytes = filepath.stat().st_size
        size_mb = size_bytes / 1024 / 1024
        
        # Count rows (efficient)
        with open(filepath, 'r') as f:
            row_count = sum(1 for _ in f) - 1  # Subtract header
        
        # Get column names
        df_sample = pd.read_csv(filepath, nrows=0)
        columns = df_sample.columns.tolist()
        
        return {
            "filepath": str(filepath),
            "size_bytes": size_bytes,
            "size_mb": round(size_mb, 2),
            "row_count": row_count,
            "column_count": len(columns),
            "columns": columns,
        }
    
    @staticmethod
    def check_dvc_status(filepath: Path) -> Dict[str, Any]:
        """
        Check DVC status for a file.
        
        Args:
            filepath: Path to check
            
        Returns:
            Dictionary with DVC status information
            
        Example:
            >>> status = DataLoader.check_dvc_status("data/raw/sales_data.csv")
            >>> print(f"Tracked: {status['is_tracked']}")
        """
        filepath = Path(filepath)
        dvc_file = filepath.with_suffix(filepath.suffix + '.dvc')
        
        status = {
            "filepath": str(filepath),
            "is_tracked": dvc_file.exists(),
            "dvc_file": str(dvc_file) if dvc_file.exists() else None,
            "file_exists": filepath.exists(),
        }
        
        if dvc_file.exists():
            # Read DVC metadata
            import yaml
            with open(dvc_file, 'r') as f:
                dvc_metadata = yaml.safe_load(f)
            
            status["md5"] = dvc_metadata.get("outs", [{}])[0].get("md5")
            status["size"] = dvc_metadata.get("outs", [{}])[0].get("size")
        
        return status


if __name__ == "__main__":
    # Test data loader
    loader = DataLoader()
    
    # Test loading CSV (if exists)
    test_file = Path("data/raw/sales_data.csv")
    if test_file.exists():
        logger.info("Testing CSV loading...")
        df = loader.load_csv(test_file)
        logger.info(f"Successfully loaded {len(df)} rows")
        
        # Get file info
        info = loader.get_data_info(test_file)
        logger.info(f"File info: {info}")
        
        # Check DVC status
        status = loader.check_dvc_status(test_file)
        logger.info(f"DVC status: {status}")
    else:
        logger.info(f"Test file not found: {test_file}")
    
    logger.info("Data loader tests completed")
