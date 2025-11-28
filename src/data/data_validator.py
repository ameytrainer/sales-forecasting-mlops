"""
Data Validation Module

Validates data quality using Great Expectations and custom checks.

Author: Amey Talkatkar
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime

from src.config import get_settings
from src.utils import setup_logging, timer


logger = setup_logging(__name__)


class DataValidator:
    """
    Validate data quality with comprehensive checks.
    
    Implements validation rules for:
    - Schema validation (column names, types)
    - Completeness (missing values)
    - Consistency (value ranges, formats)
    - Uniqueness (duplicate detection)
    - Timeliness (date range checks)
    
    Examples:
        >>> validator = DataValidator()
        >>> report = validator.validate_sales_data(df)
        >>> if report['is_valid']:
        >>>     print("Data is valid!")
    """
    
    def __init__(self):
        """Initialize validator with settings."""
        self.settings = get_settings()
        logger.info("DataValidator initialized")
    
    @timer
    def validate_sales_data(
        self,
        df: pd.DataFrame,
        strict: bool = False
    ) -> Dict[str, Any]:
        """
        Validate sales data with domain-specific rules.
        
        Args:
            df: DataFrame to validate
            strict: Fail on warnings if True
            
        Returns:
            Validation report dictionary
            
        Example:
            >>> report = validator.validate_sales_data(df)
            >>> print(f"Valid: {report['is_valid']}")
        """
        logger.info("Validating sales data...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "row_count": len(df),
            "column_count": len(df.columns),
            "checks_passed": [],
            "checks_failed": [],
            "warnings": [],
            "is_valid": True,
        }
        
        # Run all validation checks
        self._check_schema(df, report)
        self._check_completeness(df, report)
        self._check_data_types(df, report)
        self._check_value_ranges(df, report)
        self._check_dates(df, report)
        self._check_duplicates(df, report)
        self._check_statistical_properties(df, report)
        
        # Determine overall validity
        report["is_valid"] = len(report["checks_failed"]) == 0
        
        if strict and len(report["warnings"]) > 0:
            report["is_valid"] = False
        
        # Log summary
        logger.info(
            f"Validation complete: "
            f"Passed={len(report['checks_passed'])}, "
            f"Failed={len(report['checks_failed'])}, "
            f"Warnings={len(report['warnings'])}"
        )
        
        return report
    
    def _check_schema(self, df: pd.DataFrame, report: Dict) -> None:
        """Check if required columns exist."""
        required_columns = [
            'date', 'region', 'product', 'price', 'quantity', 'sales'
        ]
        
        missing_cols = set(required_columns) - set(df.columns)
        
        if missing_cols:
            report["checks_failed"].append({
                "check": "schema_validation",
                "message": f"Missing required columns: {missing_cols}"
            })
        else:
            report["checks_passed"].append("schema_validation")
    
    def _check_completeness(self, df: pd.DataFrame, report: Dict) -> None:
        """Check for missing values."""
        null_counts = df.isnull().sum()
        null_pct = (null_counts / len(df) * 100).round(2)
        
        # Critical columns should have no nulls
        critical_cols = ['date', 'sales']
        critical_nulls = null_counts[critical_cols]
        
        if critical_nulls.any():
            report["checks_failed"].append({
                "check": "completeness_critical",
                "message": f"Null values in critical columns: {critical_nulls[critical_nulls > 0].to_dict()}"
            })
        else:
            report["checks_passed"].append("completeness_critical")
        
        # Warn about high null percentages
        high_null_cols = null_pct[null_pct > 10].index.tolist()
        if high_null_cols:
            report["warnings"].append({
                "check": "completeness_warning",
                "message": f"Columns with >10% nulls: {high_null_cols}"
            })
        else:
            report["checks_passed"].append("completeness_general")
    
    def _check_data_types(self, df: pd.DataFrame, report: Dict) -> None:
        """Check if data types are correct."""
        type_checks = {
            'price': (float, int),
            'quantity': (int,),
            'sales': (float, int),
        }
        
        type_errors = []
        for col, expected_types in type_checks.items():
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    type_errors.append(f"{col} (expected numeric, got {df[col].dtype})")
        
        if type_errors:
            report["checks_failed"].append({
                "check": "data_types",
                "message": f"Type mismatches: {type_errors}"
            })
        else:
            report["checks_passed"].append("data_types")
    
    def _check_value_ranges(self, df: pd.DataFrame, report: Dict) -> None:
        """Check if values are within expected ranges."""
        range_violations = []
        
        # Price should be positive
        if 'price' in df.columns:
            if (df['price'] <= 0).any():
                count = (df['price'] <= 0).sum()
                range_violations.append(f"price: {count} non-positive values")
        
        # Quantity should be positive
        if 'quantity' in df.columns:
            if (df['quantity'] <= 0).any():
                count = (df['quantity'] <= 0).sum()
                range_violations.append(f"quantity: {count} non-positive values")
        
        # Sales should be non-negative
        if 'sales' in df.columns:
            if (df['sales'] < 0).any():
                count = (df['sales'] < 0).sum()
                range_violations.append(f"sales: {count} negative values")
        
        if range_violations:
            report["checks_failed"].append({
                "check": "value_ranges",
                "message": f"Range violations: {range_violations}"
            })
        else:
            report["checks_passed"].append("value_ranges")
    
    def _check_dates(self, df: pd.DataFrame, report: Dict) -> None:
        """Check date column validity."""
        if 'date' not in df.columns:
            return
        
        try:
            # Convert to datetime
            dates = pd.to_datetime(df['date'])
            
            # Check for future dates
            today = pd.Timestamp.now()
            future_dates = (dates > today).sum()
            
            if future_dates > 0:
                report["warnings"].append({
                    "check": "date_validity",
                    "message": f"{future_dates} future dates found"
                })
            
            # Check date range
            date_range = (dates.max() - dates.min()).days
            logger.info(f"Date range: {date_range} days")
            
            if date_range < 30:
                report["warnings"].append({
                    "check": "date_range",
                    "message": f"Short date range: {date_range} days"
                })
            
            report["checks_passed"].append("date_validity")
            
        except Exception as e:
            report["checks_failed"].append({
                "check": "date_parsing",
                "message": f"Failed to parse dates: {str(e)}"
            })
    
    def _check_duplicates(self, df: pd.DataFrame, report: Dict) -> None:
        """Check for duplicate rows."""
        # Check for exact duplicates
        duplicate_count = df.duplicated().sum()
        
        if duplicate_count > 0:
            duplicate_pct = (duplicate_count / len(df) * 100).round(2)
            report["warnings"].append({
                "check": "duplicates",
                "message": f"{duplicate_count} duplicate rows ({duplicate_pct}%)"
            })
        else:
            report["checks_passed"].append("no_duplicates")
        
        # Check for duplicates on key columns
        if all(col in df.columns for col in ['date', 'region', 'product']):
            key_duplicates = df.duplicated(subset=['date', 'region', 'product']).sum()
            
            if key_duplicates > 0:
                report["checks_failed"].append({
                    "check": "key_duplicates",
                    "message": f"{key_duplicates} duplicate date-region-product combinations"
                })
            else:
                report["checks_passed"].append("no_key_duplicates")
    
    def _check_statistical_properties(self, df: pd.DataFrame, report: Dict) -> None:
        """Check statistical properties for anomalies."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        outlier_warnings = []
        for col in numeric_cols:
            if col in ['price', 'quantity', 'sales']:
                # Check for outliers using IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_pct = (outliers / len(df) * 100).round(2)
                
                if outlier_pct > 5:
                    outlier_warnings.append(f"{col}: {outliers} outliers ({outlier_pct}%)")
        
        if outlier_warnings:
            report["warnings"].append({
                "check": "statistical_outliers",
                "message": f"Potential outliers: {outlier_warnings}"
            })
        else:
            report["checks_passed"].append("no_extreme_outliers")
    
    def validate_features(
        self,
        df: pd.DataFrame,
        expected_features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate feature-engineered data.
        
        Args:
            df: Feature DataFrame
            expected_features: List of expected feature names
            
        Returns:
            Validation report
            
        Example:
            >>> report = validator.validate_features(X_train, expected_features)
        """
        logger.info("Validating feature data...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "row_count": len(df),
            "column_count": len(df.columns),
            "checks_passed": [],
            "checks_failed": [],
            "warnings": [],
            "is_valid": True,
        }
        
        # Check expected features
        if expected_features:
            missing = set(expected_features) - set(df.columns)
            extra = set(df.columns) - set(expected_features)
            
            if missing:
                report["checks_failed"].append({
                    "check": "expected_features",
                    "message": f"Missing features: {missing}"
                })
            
            if extra:
                report["warnings"].append({
                    "check": "extra_features",
                    "message": f"Unexpected features: {extra}"
                })
        
        # Check for infinite values
        if df.select_dtypes(include=[np.number]).apply(lambda x: np.isinf(x).any()).any():
            report["checks_failed"].append({
                "check": "infinite_values",
                "message": "Infinite values detected"
            })
        else:
            report["checks_passed"].append("no_infinite_values")
        
        # Check for all-zero columns
        zero_cols = df.columns[(df == 0).all()].tolist()
        if zero_cols:
            report["warnings"].append({
                "check": "zero_columns",
                "message": f"All-zero columns: {zero_cols}"
            })
        
        report["is_valid"] = len(report["checks_failed"]) == 0
        
        return report
    
    def generate_validation_report(self, report: Dict) -> str:
        """
        Generate human-readable validation report.
        
        Args:
            report: Validation report dictionary
            
        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "DATA VALIDATION REPORT",
            "=" * 60,
            f"Timestamp: {report['timestamp']}",
            f"Rows: {report['row_count']:,}",
            f"Columns: {report['column_count']}",
            f"Status: {'✅ VALID' if report['is_valid'] else '❌ INVALID'}",
            "",
            f"Checks Passed: {len(report['checks_passed'])}",
            f"Checks Failed: {len(report['checks_failed'])}",
            f"Warnings: {len(report['warnings'])}",
        ]
        
        if report['checks_failed']:
            lines.extend([
                "",
                "FAILED CHECKS:",
                "-" * 60,
            ])
            for i, check in enumerate(report['checks_failed'], 1):
                lines.append(f"{i}. {check['check']}: {check['message']}")
        
        if report['warnings']:
            lines.extend([
                "",
                "WARNINGS:",
                "-" * 60,
            ])
            for i, warning in enumerate(report['warnings'], 1):
                lines.append(f"{i}. {warning['check']}: {warning['message']}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Test validator
    validator = DataValidator()
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
        'product': np.random.choice(['Electronics', 'Clothing', 'Food'], 100),
        'price': np.random.uniform(10, 500, 100),
        'quantity': np.random.randint(1, 100, 100),
        'sales': np.random.uniform(100, 5000, 100),
    })
    
    # Run validation
    report = validator.validate_sales_data(sample_data)
    
    # Print report
    print(validator.generate_validation_report(report))
    
    logger.info("Validator tests completed")
