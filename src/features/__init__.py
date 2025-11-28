"""
Feature Engineering Package

Contains modules for creating and selecting features.

Modules:
- feature_engineering: Create lag, rolling, and time-based features
- feature_selector: Select important features using various methods

Author: Amey Talkatkar
"""

from src.features.feature_engineering import FeatureEngineer
from src.features.feature_selector import FeatureSelector

__all__ = [
    "FeatureEngineer",
    "FeatureSelector",
]
