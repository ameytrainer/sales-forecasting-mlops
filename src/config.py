"""
Configuration Management for MLOps Pipeline

Uses Pydantic Settings for type-safe configuration management.
All settings can be overridden via environment variables.

Author: Amey Talkatkar
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings have sensible defaults for development.
    Override via .env file or environment variables.
    
    Example:
        >>> settings = get_settings()
        >>> print(settings.mlflow_tracking_uri)
    """
    
    # ==================== Project Settings ====================
    project_name: str = Field(
        default="Sales Forecasting MLOps",
        description="Project name"
    )
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent,
        description="Project root directory"
    )
    environment: str = Field(
        default="development",
        description="Environment: development, staging, production"
    )
    debug: bool = Field(
        default=True,
        description="Enable debug mode"
    )
    
    # ==================== Data Settings ====================
    data_dir: Path = Field(
        default_factory=lambda: Path("data"),
        description="Base data directory"
    )
    raw_data_dir: Path = Field(
        default_factory=lambda: Path("data/raw"),
        description="Raw data directory"
    )
    processed_data_dir: Path = Field(
        default_factory=lambda: Path("data/processed"),
        description="Processed data directory"
    )
    predictions_dir: Path = Field(
        default_factory=lambda: Path("data/predictions"),
        description="Predictions output directory"
    )
    
    # ==================== MLflow Settings ====================
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI"
    )
    mlflow_tracking_username: Optional[str] = Field(
        default=None,
        description="MLflow tracking username (for DagsHub)"
    )
    mlflow_tracking_password: Optional[str] = Field(
        default=None,
        description="MLflow tracking password (for DagsHub)"
    )
    mlflow_experiment_name: str = Field(
        default="sales_forecasting",
        description="Default MLflow experiment name"
    )
    mlflow_model_name: str = Field(
        default="sales_forecasting_model",
        description="Registered model name in MLflow"
    )
    
    # ==================== DVC Settings ====================
    dvc_remote_name: str = Field(
        default="origin",
        description="DVC remote name"
    )
    dvc_remote_url: Optional[str] = Field(
        default=None,
        description="DVC remote URL (DagsHub or S3)"
    )
    
    # ==================== Database Settings ====================
    database_url: str = Field(
        default="postgresql://airflow:airflow123@localhost/airflow",
        description="Database connection URL"
    )
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_user: str = Field(default="airflow")
    postgres_password: str = Field(default="airflow123")
    postgres_db: str = Field(default="airflow")
    
    # ==================== Airflow Settings ====================
    airflow_home: Path = Field(
        default_factory=lambda: Path.home() / "airflow",
        description="Airflow home directory"
    )
    airflow_dags_folder: Optional[Path] = Field(
        default=None,
        description="Airflow DAGs folder"
    )
    
    # ==================== API Settings ====================
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_reload: bool = Field(default=True)
    api_workers: int = Field(default=1)
    
    # ==================== Model Settings ====================
    model_random_state: int = Field(default=42)
    model_test_size: float = Field(default=0.2)
    model_cv_folds: int = Field(default=5)
    
    # Linear Regression
    lr_fit_intercept: bool = Field(default=True)
    
    # Random Forest
    rf_n_estimators: int = Field(default=100)
    rf_max_depth: int = Field(default=10)
    rf_min_samples_split: int = Field(default=20)
    rf_min_samples_leaf: int = Field(default=10)
    rf_n_jobs: int = Field(default=-1)
    
    # XGBoost
    xgb_n_estimators: int = Field(default=100)
    xgb_learning_rate: float = Field(default=0.1)
    xgb_max_depth: int = Field(default=5)
    xgb_min_child_weight: int = Field(default=3)
    xgb_subsample: float = Field(default=0.8)
    xgb_colsample_bytree: float = Field(default=0.8)
    xgb_n_jobs: int = Field(default=-1)
    
    # ==================== Feature Engineering Settings ====================
    lag_features: List[int] = Field(
        default=[1, 7, 30],
        description="Lag periods for feature engineering"
    )
    rolling_windows: List[int] = Field(
        default=[7, 30],
        description="Rolling window sizes"
    )
    
    # ==================== Monitoring Settings ====================
    drift_detection_threshold: float = Field(
        default=0.05,
        description="P-value threshold for drift detection (KS test)"
    )
    performance_degradation_threshold: float = Field(
        default=0.1,
        description="Threshold for model performance degradation (10%)"
    )
    monitoring_window_days: int = Field(
        default=7,
        description="Days of data to use for monitoring"
    )
    
    # ==================== Logging Settings ====================
    log_level: str = Field(default="INFO")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_file: Optional[Path] = Field(default=None)
    
    # ==================== Resource Settings ====================
    max_memory_mb: int = Field(
        default=1500,
        description="Maximum memory usage in MB (for 2GB instance)"
    )
    chunk_size: int = Field(
        default=5000,
        description="Chunk size for processing large datasets"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment value."""
        valid_envs = ["development", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v
    
    @field_validator("model_test_size")
    @classmethod
    def validate_test_size(cls, v: float) -> float:
        """Validate test size is between 0 and 1."""
        if not 0 < v < 1:
            raise ValueError("Test size must be between 0 and 1")
        return v
    
    def get_database_url(self) -> str:
        """Get formatted database URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    def setup_mlflow_auth(self) -> None:
        """Setup MLflow authentication environment variables."""
        if self.mlflow_tracking_username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = self.mlflow_tracking_username
        if self.mlflow_tracking_password:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = self.mlflow_tracking_password
    
    def create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.raw_data_dir,
            self.processed_data_dir,
            self.predictions_dir,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_model_params(self, model_type: str) -> dict:
        """
        Get model-specific parameters.
        
        Args:
            model_type: Type of model ('lr', 'rf', 'xgb')
            
        Returns:
            Dictionary of model parameters
            
        Raises:
            ValueError: If model_type is invalid
        """
        if model_type == "lr":
            return {
                "fit_intercept": self.lr_fit_intercept,
            }
        elif model_type == "rf":
            return {
                "n_estimators": self.rf_n_estimators,
                "max_depth": self.rf_max_depth,
                "min_samples_split": self.rf_min_samples_split,
                "min_samples_leaf": self.rf_min_samples_leaf,
                "random_state": self.model_random_state,
                "n_jobs": self.rf_n_jobs,
            }
        elif model_type == "xgb":
            return {
                "n_estimators": self.xgb_n_estimators,
                "learning_rate": self.xgb_learning_rate,
                "max_depth": self.xgb_max_depth,
                "min_child_weight": self.xgb_min_child_weight,
                "subsample": self.xgb_subsample,
                "colsample_bytree": self.xgb_colsample_bytree,
                "random_state": self.model_random_state,
                "n_jobs": self.xgb_n_jobs,
                "verbosity": 0,
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses LRU cache to ensure only one instance is created.
    
    Returns:
        Settings instance
        
    Example:
        >>> settings = get_settings()
        >>> print(settings.mlflow_tracking_uri)
    """
    settings = Settings()
    settings.create_directories()
    return settings


# Convenience function to get settings
def load_config() -> Settings:
    """Load and return configuration settings."""
    return get_settings()


if __name__ == "__main__":
    # Test configuration
    settings = get_settings()
    print(f"Project: {settings.project_name}")
    print(f"Environment: {settings.environment}")
    print(f"MLflow URI: {settings.mlflow_tracking_uri}")
    print(f"Database URL: {settings.get_database_url()}")
    print(f"RF Params: {settings.get_model_params('rf')}")
