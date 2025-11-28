"""
Sales Forecasting MLOps Pipeline - Source Package

This package contains all core modules for the MLOps pipeline:
- data: Data loading, validation, and processing
- features: Feature engineering and selection
- models: Model training, evaluation, and registry operations
- monitoring: Drift detection and performance monitoring

Author: Amey Talkatkar
Email: ameytalkatkar169@gmail.com
GitHub: https://github.com/ameytrainer
Course: MLOps with Agentic AI - Advanced Certification
"""

__version__ = "1.0.0"
__author__ = "Amey Talkatkar"
__email__ = "ameytalkatkar169@gmail.com"

# Package-level imports for convenience
from src.config import Settings, get_settings

__all__ = ["Settings", "get_settings", "__version__"]
