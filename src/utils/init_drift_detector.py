"""
Initialize Data Drift Detector with Training Statistics
This script should be run after model training to set up drift detection
"""
import pandas as pd
import numpy as np
import logging
import os
import sys
import joblib
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.metrics import drift_detector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_training_stats(df: pd.DataFrame, feature_columns: list) -> dict:
    """
    Calculate statistics for training data features
    
    Args:
        df: Training DataFrame
        feature_columns: List of feature column names
        
    Returns:
        Dictionary with feature statistics
    """
    stats = {}
    
    for feature in feature_columns:
        if feature not in df.columns:
            logger.warning(f"Feature '{feature}' not found in DataFrame")
            continue
        
        feature_data = df[feature].dropna()
        
        if len(feature_data) == 0:
            logger.warning(f"Feature '{feature}' has no valid data")
            continue
        
        stats[feature] = {
            'mean': float(feature_data.mean()),
            'std': float(feature_data.std()),
            'min': float(feature_data.min()),
            'max': float(feature_data.max()),
            'median': float(feature_data.median()),
            'q25': float(feature_data.quantile(0.25)),
            'q75': float(feature_data.quantile(0.75))
        }
    
    return stats


def initialize_drift_detector_from_data(data_file: str, feature_columns_file: str = None):
    """
    Initialize drift detector from training data
    
    Args:
        data_file: Path to training data CSV
        feature_columns_file: Path to saved feature columns (optional)
    """
    logger.info("=" * 60)
    logger.info("INITIALIZING DATA DRIFT DETECTOR")
    logger.info("=" * 60)
    
    # Load training data
    logger.info(f"Loading training data from: {data_file}")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Get feature columns
    if feature_columns_file and os.path.exists(feature_columns_file):
        logger.info(f"Loading feature columns from: {feature_columns_file}")
        feature_columns = joblib.load(feature_columns_file)
    else:
        # Use all numeric columns except target
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'price' in numeric_cols:
            numeric_cols.remove('price')
        feature_columns = numeric_cols
        logger.info(f"Using {len(feature_columns)} numeric columns as features")
    
    # Calculate statistics
    logger.info("Calculating training statistics...")
    training_stats = calculate_training_stats(df, feature_columns)
    logger.info(f"Calculated stats for {len(training_stats)} features")
    
    # Set training stats in drift detector
    drift_detector.set_training_stats(training_stats)
    
    logger.info("=" * 60)
    logger.info("✓ DRIFT DETECTOR INITIALIZED")
    logger.info("=" * 60)
    
    return training_stats


def save_training_stats(training_stats: dict, output_file: str):
    """
    Save training statistics to file
    
    Args:
        training_stats: Dictionary of feature statistics
        output_file: Output file path
    """
    import json
    
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    logger.info(f"Training statistics saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize Data Drift Detector")
    parser.add_argument("data_file", help="Path to training data CSV")
    parser.add_argument("--features", help="Path to feature columns file (optional)")
    parser.add_argument("--output", help="Path to save training stats JSON (optional)")
    
    args = parser.parse_args()
    
    # Initialize drift detector
    training_stats = initialize_drift_detector_from_data(args.data_file, args.features)
    
    # Save stats if output path provided
    if args.output:
        save_training_stats(training_stats, args.output)

