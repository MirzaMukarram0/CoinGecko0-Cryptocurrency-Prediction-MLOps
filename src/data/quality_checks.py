"""
Data Quality Checks Module
Comprehensive data validation and quality assurance
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityException(Exception):
    """Custom exception for data quality violations"""
    pass


class DataQualityChecker:
    """
    Comprehensive data quality checker for cryptocurrency data
    """
    
    def __init__(self, max_null_percentage: float = 1.0):
        """
        Initialize quality checker
        
        Args:
            max_null_percentage: Maximum allowed null percentage before failing
        """
        self.max_null_percentage = max_null_percentage
        
    def check_null_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for null values in the dataset
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary with null value statistics
            
        Raises:
            DataQualityException: If null percentage exceeds threshold
        """
        logger.info("Checking for null values...")
        
        null_counts = df.isnull().sum()
        null_percentages = (null_counts / len(df)) * 100
        
        result = {
            "null_counts": null_counts.to_dict(),
            "null_percentages": null_percentages.to_dict(),
            "total_rows": len(df),
            "failed_columns": []
        }
        
        # Check if any column exceeds threshold
        failed_columns = null_percentages[null_percentages > self.max_null_percentage]
        if not failed_columns.empty:
            result["failed_columns"] = failed_columns.to_dict()
            raise DataQualityException(
                f"Null percentage exceeds threshold ({self.max_null_percentage}%): {failed_columns.to_dict()}"
            )
        
        logger.info("✓ Null value check passed")
        return result
    
    def check_schema(self, df: pd.DataFrame, expected_columns: List[str]) -> Dict[str, Any]:
        """
        Validate DataFrame schema
        
        Args:
            df: DataFrame to validate
            expected_columns: List of required column names
            
        Returns:
            Schema validation results
            
        Raises:
            DataQualityException: If schema validation fails
        """
        logger.info("Validating schema...")
        
        actual_columns = set(df.columns)
        expected_columns_set = set(expected_columns)
        
        missing_columns = expected_columns_set - actual_columns
        extra_columns = actual_columns - expected_columns_set
        
        result = {
            "expected_columns": expected_columns,
            "actual_columns": list(actual_columns),
            "missing_columns": list(missing_columns),
            "extra_columns": list(extra_columns),
            "schema_valid": len(missing_columns) == 0
        }
        
        if missing_columns:
            raise DataQualityException(f"Missing required columns: {missing_columns}")
        
        logger.info("✓ Schema validation passed")
        return result
    
    def check_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data types for cryptocurrency data
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Data type validation results
        """
        logger.info("Checking data types...")
        
        # Expected data types for crypto data
        numeric_columns = ['price', 'market_cap', 'total_volume']
        
        type_issues = {}
        
        for col in numeric_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    type_issues[col] = f"Expected numeric, got {df[col].dtype}"
        
        # Check if index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            type_issues['index'] = f"Expected DatetimeIndex, got {type(df.index)}"
        
        result = {
            "data_types": df.dtypes.astype(str).to_dict(),
            "type_issues": type_issues,
            "types_valid": len(type_issues) == 0
        }
        
        if type_issues:
            logger.warning(f"Data type issues found: {type_issues}")
        else:
            logger.info("✓ Data type check passed")
        
        return result
    
    def check_value_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check if values are within expected ranges
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Value range validation results
        """
        logger.info("Checking value ranges...")
        
        range_issues = {}
        
        # Check for negative prices (should not happen)
        if 'price' in df.columns:
            negative_prices = df[df['price'] < 0]
            if not negative_prices.empty:
                range_issues['price'] = f"Found {len(negative_prices)} negative price values"
        
        # Check for zero or negative volumes
        if 'total_volume' in df.columns:
            invalid_volumes = df[df['total_volume'] <= 0]
            if not invalid_volumes.empty:
                range_issues['total_volume'] = f"Found {len(invalid_volumes)} zero/negative volume values"
        
        # Check for extremely high price changes (potential data errors)
        if 'price' in df.columns and len(df) > 1:
            price_changes = df['price'].pct_change().abs()
            extreme_changes = price_changes[price_changes > 0.5]  # 50% change
            if not extreme_changes.empty:
                range_issues['price_volatility'] = f"Found {len(extreme_changes)} extreme price changes (>50%)"
        
        result = {
            "range_issues": range_issues,
            "ranges_valid": len(range_issues) == 0
        }
        
        if range_issues:
            logger.warning(f"Value range issues found: {range_issues}")
        else:
            logger.info("✓ Value range check passed")
        
        return result
    
    def check_data_freshness(self, df: pd.DataFrame, max_age_hours: int = 2) -> Dict[str, Any]:
        """
        Check if data is fresh (recent)
        
        Args:
            df: DataFrame to check
            max_age_hours: Maximum allowed age in hours
            
        Returns:
            Data freshness validation results
        """
        logger.info("Checking data freshness...")
        
        if df.empty:
            return {"fresh": False, "reason": "Empty dataset"}
        
        latest_timestamp = df.index.max()
        current_time = pd.Timestamp.now(tz=latest_timestamp.tz)
        age_hours = (current_time - latest_timestamp).total_seconds() / 3600
        
        is_fresh = age_hours <= max_age_hours
        
        result = {
            "latest_timestamp": latest_timestamp.isoformat(),
            "current_time": current_time.isoformat(),
            "age_hours": age_hours,
            "max_age_hours": max_age_hours,
            "fresh": is_fresh
        }
        
        if is_fresh:
            logger.info(f"✓ Data is fresh (age: {age_hours:.1f} hours)")
        else:
            logger.warning(f"⚠ Data is stale (age: {age_hours:.1f} hours)")
        
        return result
    
    def check_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data completeness (no gaps in time series)
        
        Args:
            df: DataFrame to check
            
        Returns:
            Completeness validation results
        """
        logger.info("Checking data completeness...")
        
        if df.empty:
            return {"complete": False, "reason": "Empty dataset"}
        
        # For hourly data, check if we have expected number of records
        time_diff = (df.index.max() - df.index.min()).total_seconds() / 3600
        expected_records = int(time_diff) + 1
        actual_records = len(df)
        
        completeness_ratio = actual_records / expected_records if expected_records > 0 else 0
        
        result = {
            "expected_records": expected_records,
            "actual_records": actual_records,
            "completeness_ratio": completeness_ratio,
            "complete": completeness_ratio >= 0.9  # 90% completeness threshold
        }
        
        if result["complete"]:
            logger.info(f"✓ Data is complete ({completeness_ratio:.1%})")
        else:
            logger.warning(f"⚠ Data is incomplete ({completeness_ratio:.1%})")
        
        return result
    
    def run_all_checks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run all quality checks
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Comprehensive quality report
            
        Raises:
            DataQualityException: If any critical check fails
        """
        logger.info("=" * 60)
        logger.info("STARTING DATA QUALITY CHECKS")
        logger.info("=" * 60)
        
        results = {
            "dataset_info": {
                "rows": len(df),
                "columns": len(df.columns),
                "memory_usage": df.memory_usage(deep=True).sum()
            }
        }
        
        try:
            # Required columns for crypto data
            expected_columns = ['price', 'market_cap', 'total_volume']
            
            # Run all checks
            results["null_check"] = self.check_null_values(df)
            results["schema_check"] = self.check_schema(df, expected_columns)
            results["type_check"] = self.check_data_types(df)
            results["range_check"] = self.check_value_ranges(df)
            results["freshness_check"] = self.check_data_freshness(df)
            results["completeness_check"] = self.check_completeness(df)
            
            # Determine overall quality status
            all_passed = all([
                results["null_check"].get("failed_columns", []) == [],
                results["schema_check"]["schema_valid"],
                results["type_check"]["types_valid"],
                results["range_check"]["ranges_valid"]
            ])
            
            results["overall_status"] = "PASSED" if all_passed else "FAILED"
            
            logger.info("=" * 60)
            if all_passed:
                logger.info("✓ ALL QUALITY CHECKS PASSED")
            else:
                logger.error("✗ SOME QUALITY CHECKS FAILED")
            logger.info("=" * 60)
            
            return results
            
        except DataQualityException:
            logger.error("✗ CRITICAL QUALITY CHECK FAILED")
            logger.info("=" * 60)
            raise


def check_data_quality(filepath: str, 
                      max_null_percentage: float = 1.0) -> Dict[str, Any]:
    """
    Main quality check function for Airflow tasks
    
    Args:
        filepath: Path to CSV file to check
        max_null_percentage: Maximum allowed null percentage
        
    Returns:
        Quality check results
        
    Raises:
        DataQualityException: If quality checks fail
    """
    logger.info(f"Loading data from: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    # Load data
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    # Initialize checker
    checker = DataQualityChecker(max_null_percentage=max_null_percentage)
    
    # Run checks
    results = checker.run_all_checks(df)
    
    return results


if __name__ == "__main__":
    # Test the quality checker
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        try:
            results = check_data_quality(filepath)
            print(f"Quality check results: {results['overall_status']}")
        except Exception as e:
            print(f"Quality check failed: {e}")
    else:
        print("Usage: python quality_checks.py <filepath>")