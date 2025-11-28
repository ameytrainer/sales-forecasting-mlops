"""
Utility Functions for MLOps Pipeline

Common utilities for logging, memory management, file operations, etc.

Author: Amey Talkatkar
"""

import logging
import os
import sys
import time
import psutil
import subprocess
from pathlib import Path
from typing import Optional, Any, Dict, List
from functools import wraps
from datetime import datetime

import pandas as pd
import numpy as np

from src.config import get_settings


def setup_logging(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Setup logger with consistent formatting.
    
    Args:
        name: Logger name (usually __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logging(__name__)
        >>> logger.info("Pipeline started")
    """
    settings = get_settings()
    level = level or settings.log_level
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    formatter = logging.Formatter(settings.log_format)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory statistics in MB
        
    Example:
        >>> memory = get_memory_usage()
        >>> print(f"Used: {memory['used_mb']:.2f} MB")
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    virtual_memory = psutil.virtual_memory()
    
    return {
        "process_mb": memory_info.rss / 1024 / 1024,
        "available_mb": virtual_memory.available / 1024 / 1024,
        "used_mb": virtual_memory.used / 1024 / 1024,
        "percent": virtual_memory.percent,
        "total_mb": virtual_memory.total / 1024 / 1024,
    }


def check_memory_limit(threshold_mb: Optional[int] = None) -> bool:
    """
    Check if memory usage is within acceptable limits.
    
    Args:
        threshold_mb: Memory threshold in MB (default from settings)
        
    Returns:
        True if within limits, False otherwise
        
    Raises:
        MemoryError: If memory usage exceeds threshold
    """
    settings = get_settings()
    threshold_mb = threshold_mb or settings.max_memory_mb
    
    memory = get_memory_usage()
    
    if memory["process_mb"] > threshold_mb:
        raise MemoryError(
            f"Memory usage ({memory['process_mb']:.2f} MB) "
            f"exceeds threshold ({threshold_mb} MB)"
        )
    
    return True


def timer(func):
    """
    Decorator to measure function execution time.
    
    Example:
        >>> @timer
        >>> def train_model():
        >>>     # training code
        >>>     pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        logger.info(f"Starting {func.__name__}...")
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(
                f"Completed {func.__name__} in {elapsed:.2f}s"
            )
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"Failed {func.__name__} after {elapsed:.2f}s: {e}"
            )
            raise
    
    return wrapper


def memory_monitor(func):
    """
    Decorator to monitor memory usage during function execution.
    
    Example:
        >>> @memory_monitor
        >>> def process_data():
        >>>     # processing code
        >>>     pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        
        # Memory before
        mem_before = get_memory_usage()
        logger.debug(
            f"Memory before {func.__name__}: {mem_before['process_mb']:.2f} MB"
        )
        
        # Execute function
        try:
            result = func(*args, **kwargs)
            
            # Memory after
            mem_after = get_memory_usage()
            mem_diff = mem_after["process_mb"] - mem_before["process_mb"]
            logger.debug(
                f"Memory after {func.__name__}: {mem_after['process_mb']:.2f} MB "
                f"(Δ {mem_diff:+.2f} MB)"
            )
            
            # Check if within limits
            check_memory_limit()
            
            return result
            
        except MemoryError:
            logger.error(f"Memory limit exceeded in {func.__name__}")
            raise
    
    return wrapper


def load_data_chunked(
    filepath: Path,
    chunk_size: Optional[int] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Load CSV data in chunks to manage memory.
    
    Args:
        filepath: Path to CSV file
        chunk_size: Rows per chunk (default from settings)
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        Complete DataFrame
        
    Example:
        >>> df = load_data_chunked("data/raw/sales_data.csv")
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading data from {filepath} in chunks of {chunk_size}")
    
    chunks = []
    for chunk in pd.read_csv(filepath, chunksize=chunk_size, **kwargs):
        chunks.append(chunk)
        check_memory_limit()
    
    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"Loaded {len(df):,} rows")
    
    return df


def save_data_efficiently(
    df: pd.DataFrame,
    filepath: Path,
    compress: bool = False
) -> None:
    """
    Save DataFrame efficiently with optional compression.
    
    Args:
        df: DataFrame to save
        filepath: Output file path
        compress: Whether to compress (gzip)
        
    Example:
        >>> save_data_efficiently(df, "data/processed/features.csv", compress=True)
    """
    logger = logging.getLogger(__name__)
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if compress:
        filepath = filepath.with_suffix('.csv.gz')
        compression = 'gzip'
    else:
        compression = None
    
    df.to_csv(filepath, index=False, compression=compression)
    
    size_mb = filepath.stat().st_size / 1024 / 1024
    logger.info(f"Saved {len(df):,} rows to {filepath} ({size_mb:.2f} MB)")


def run_command(
    command: List[str],
    cwd: Optional[Path] = None,
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run shell command and return results.
    
    Args:
        command: Command as list of strings
        cwd: Working directory
        timeout: Timeout in seconds
        
    Returns:
        Dictionary with returncode, stdout, stderr
        
    Raises:
        subprocess.TimeoutExpired: If timeout is exceeded
        
    Example:
        >>> result = run_command(["dvc", "pull"])
        >>> if result['returncode'] == 0:
        >>>     print("Success!")
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running command: {' '.join(command)}")
    
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=timeout
    )
    
    if result.returncode != 0:
        logger.error(f"Command failed with code {result.returncode}")
        logger.error(f"stderr: {result.stderr}")
    else:
        logger.info("Command completed successfully")
    
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 1
) -> None:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        
    Raises:
        ValueError: If validation fails
        
    Example:
        >>> validate_dataframe(df, required_columns=['date', 'sales'], min_rows=100)
    """
    # Check if DataFrame is empty
    if len(df) < min_rows:
        raise ValueError(f"DataFrame has {len(df)} rows, need at least {min_rows}")
    
    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for all-null columns
    null_cols = df.columns[df.isnull().all()].tolist()
    if null_cols:
        raise ValueError(f"Columns with all null values: {null_cols}")


def create_run_id() -> str:
    """
    Create unique run ID based on timestamp.
    
    Returns:
        Run ID string
        
    Example:
        >>> run_id = create_run_id()
        >>> print(run_id)  # '20241124_153045'
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Value to return if denominator is zero
        
    Returns:
        Result of division or default
        
    Example:
        >>> result = safe_divide(10, 2)  # Returns 5.0
        >>> result = safe_divide(10, 0)  # Returns 0.0
    """
    if denominator == 0:
        return default
    return numerator / denominator


def get_git_commit_hash() -> Optional[str]:
    """
    Get current git commit hash.
    
    Returns:
        Commit hash or None if not in git repo
        
    Example:
        >>> commit = get_git_commit_hash()
        >>> print(f"Commit: {commit}")
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes as human-readable string.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
        
    Example:
        >>> print(format_bytes(1536000))  # "1.46 MB"
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def get_project_root() -> Path:
    """
    Get project root directory.
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent


if __name__ == "__main__":
    # Test utilities
    logger = setup_logging(__name__)
    logger.info("Testing utilities...")
    
    memory = get_memory_usage()
    logger.info(f"Memory usage: {memory['process_mb']:.2f} MB")
    
    run_id = create_run_id()
    logger.info(f"Run ID: {run_id}")
    
    commit = get_git_commit_hash()
    logger.info(f"Git commit: {commit}")
    
    logger.info("All tests passed!")
