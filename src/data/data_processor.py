"""
Data Processor Module

Handles data preprocessing and transformation:
- Train/test splitting
- Scaling and normalization
- Encoding categorical variables
- Data cleaning

Author: Amey Talkatkar
"""

from pathlib import Path
from typing import Tuple, Dict, Optional, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

from src.config import get_settings
from src.utils import setup_logging, timer, memory_monitor, save_data_efficiently


logger = setup_logging(__name__)


class DataProcessor:
    """
    Process and transform data for ML pipeline.
    
    Examples:
        >>> processor = DataProcessor()
        >>> X_train, X_test, y_train, y_test = processor.split_data(df, 'sales')
        >>> X_train_scaled = processor.scale_features(X_train)
    """
    
    def __init__(self):
        """Initialize data processor with settings."""
        self.settings = get_settings()
        self.scaler = None
        self.label_encoders = {}
        logger.info("DataProcessor initialized")
    
    @timer
    @memory_monitor
    def split_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None,
        shuffle: bool = False,
        save: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            test_size: Proportion of test set (default from settings)
            random_state: Random seed (default from settings)
            shuffle: Whether to shuffle data (False for time series)
            save: Save splits to disk
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
            
        Example:
            >>> X_train, X_test, y_train, y_test = processor.split_data(df, 'sales')
        """
        test_size = test_size or self.settings.model_test_size
        random_state = random_state or self.settings.model_random_state
        
        logger.info(f"Splitting data: test_size={test_size}, shuffle={shuffle}")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle
        )
        
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Save if requested
        if save:
            self._save_splits(X_train, X_test, y_train, y_test)
        
        return X_train, X_test, y_train, y_test
    
    @timer
    def scale_features(
        self,
        X: pd.DataFrame,
        fit: bool = True,
        save_scaler: bool = False
    ) -> pd.DataFrame:
        """
        Scale numeric features using StandardScaler.
        
        Args:
            X: Features DataFrame
            fit: Fit scaler (True for train, False for test)
            save_scaler: Save fitted scaler to disk
            
        Returns:
            Scaled DataFrame
            
        Example:
            >>> X_train_scaled = processor.scale_features(X_train, fit=True)
            >>> X_test_scaled = processor.scale_features(X_test, fit=False)
        """
        logger.info(f"Scaling features (fit={fit})")
        
        # Get numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns to scale")
            return X.copy()
        
        logger.info(f"Scaling {len(numeric_cols)} numeric columns")
        
        # Initialize scaler if needed
        if fit or self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X[numeric_cols]),
                columns=numeric_cols,
                index=X.index
            )
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X[numeric_cols]),
                columns=numeric_cols,
                index=X.index
            )
        
        # Add back non-numeric columns
        non_numeric_cols = [col for col in X.columns if col not in numeric_cols]
        for col in non_numeric_cols:
            X_scaled[col] = X[col]
        
        # Reorder columns to match original
        X_scaled = X_scaled[X.columns]
        
        # Save scaler if requested
        if save_scaler and fit:
            scaler_path = self.settings.processed_data_dir / "scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Saved scaler to {scaler_path}")
        
        return X_scaled
    
    @timer
    def encode_categorical(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "onehot"
    ) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            df: Input DataFrame
            columns: Columns to encode (auto-detect if None)
            method: Encoding method ('onehot' or 'label')
            
        Returns:
            DataFrame with encoded variables
            
        Example:
            >>> df_encoded = processor.encode_categorical(df, ['region', 'product'])
        """
        logger.info(f"Encoding categorical variables (method={method})")
        
        df = df.copy()
        
        # Auto-detect categorical columns if not specified
        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(columns) == 0:
            logger.info("No categorical columns to encode")
            return df
        
        logger.info(f"Encoding columns: {columns}")
        
        if method == "onehot":
            # One-hot encoding
            df = pd.get_dummies(df, columns=columns, prefix=columns, drop_first=True)
            logger.info(f"Created {len(df.columns)} columns after one-hot encoding")
            
        elif method == "label":
            # Label encoding
            for col in columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col])
                else:
                    df[col] = self.label_encoders[col].transform(df[col])
            
            logger.info(f"Label encoded {len(columns)} columns")
        else:
            raise ValueError(f"Unknown encoding method: {method}")
        
        return df
    
    @timer
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = "drop",
        fill_value: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in DataFrame.
        
        Args:
            df: Input DataFrame
            strategy: Strategy ('drop', 'mean', 'median', 'mode', 'constant')
            fill_value: Value for constant strategy
            
        Returns:
            DataFrame with missing values handled
            
        Example:
            >>> df_clean = processor.handle_missing_values(df, strategy='mean')
        """
        logger.info(f"Handling missing values (strategy={strategy})")
        
        df = df.copy()
        null_counts = df.isnull().sum()
        
        if null_counts.sum() == 0:
            logger.info("No missing values found")
            return df
        
        logger.info(f"Missing values: {null_counts[null_counts > 0].to_dict()}")
        
        if strategy == "drop":
            initial_rows = len(df)
            df = df.dropna()
            logger.info(f"Dropped {initial_rows - len(df)} rows with missing values")
            
        elif strategy == "mean":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            logger.info("Filled numeric columns with mean")
            
        elif strategy == "median":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            logger.info("Filled numeric columns with median")
            
        elif strategy == "mode":
            for col in df.columns:
                if df[col].isnull().any():
                    df[col].fillna(df[col].mode()[0], inplace=True)
            logger.info("Filled columns with mode")
            
        elif strategy == "constant":
            if fill_value is None:
                raise ValueError("fill_value required for constant strategy")
            df = df.fillna(fill_value)
            logger.info(f"Filled with constant value: {fill_value}")
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return df
    
    @timer
    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Remove outliers from DataFrame.
        
        Args:
            df: Input DataFrame
            columns: Columns to check (all numeric if None)
            method: Method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers removed
            
        Example:
            >>> df_clean = processor.remove_outliers(df, columns=['price', 'sales'])
        """
        logger.info(f"Removing outliers (method={method}, threshold={threshold})")
        
        df = df.copy()
        initial_rows = len(df)
        
        # Auto-detect numeric columns if not specified
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == "iqr":
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        elif method == "zscore":
            from scipy import stats
            for col in columns:
                z_scores = np.abs(stats.zscore(df[col]))
                df = df[z_scores < threshold]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        removed_rows = initial_rows - len(df)
        logger.info(f"Removed {removed_rows} outlier rows ({removed_rows/initial_rows*100:.2f}%)")
        
        return df
    
    def load_scaler(self, scaler_path: Optional[Path] = None) -> StandardScaler:
        """
        Load saved scaler from disk.
        
        Args:
            scaler_path: Path to scaler file
            
        Returns:
            Loaded scaler
            
        Example:
            >>> scaler = processor.load_scaler()
        """
        if scaler_path is None:
            scaler_path = self.settings.processed_data_dir / "scaler.joblib"
        
        logger.info(f"Loading scaler from {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        return self.scaler
    
    def _save_splits(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> None:
        """Save train/test splits to disk."""
        logger.info("Saving train/test splits")
        
        output_dir = self.settings.processed_data_dir
        
        X_train.to_csv(output_dir / "X_train.csv", index=False)
        X_test.to_csv(output_dir / "X_test.csv", index=False)
        y_train.to_csv(output_dir / "y_train.csv", index=False)
        y_test.to_csv(output_dir / "y_test.csv", index=False)
        
        logger.info(f"Saved splits to {output_dir}")
    
    def get_processing_pipeline(self) -> Dict[str, any]:
        """
        Get complete processing pipeline configuration.
        
        Returns:
            Dictionary with processing steps
        """
        return {
            "missing_values": "drop",
            "encoding": "onehot",
            "scaling": "standard",
            "outlier_removal": "iqr",
            "test_size": self.settings.model_test_size,
            "random_state": self.settings.model_random_state,
        }


if __name__ == "__main__":
    # Test data processor
    processor = DataProcessor()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'region': np.random.choice(['North', 'South'], 100),
        'product': np.random.choice(['A', 'B', 'C'], 100),
        'price': np.random.uniform(10, 500, 100),
        'quantity': np.random.randint(1, 100, 100),
        'sales': np.random.uniform(100, 5000, 100),
    })
    
    logger.info("Testing data splitting...")
    X_train, X_test, y_train, y_test = processor.split_data(
        sample_data, 'sales', shuffle=False
    )
    
    logger.info("Testing encoding...")
    X_train_encoded = processor.encode_categorical(X_train, ['region', 'product'])
    
    logger.info("Testing scaling...")
    X_train_scaled = processor.scale_features(X_train_encoded, fit=True)
    
    logger.info("Data processor tests completed")
