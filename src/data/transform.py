"""
Data Transformation and Feature Engineering Module
Processes raw cryptocurrency data and creates ML-ready features
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CryptoDataTransformer:
    """
    Cryptocurrency data transformer with comprehensive feature engineering
    """
    
    def __init__(self):
        """Initialize the transformer"""
        pass
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare raw cryptocurrency data
        
        Args:
            df: Raw DataFrame with timestamp index
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning raw data...")
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Remove duplicates
        df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
        
        # Sort by timestamp
        df_clean.sort_index(inplace=True)
        
        # Forward fill missing values (reasonable for time series)
        df_clean.fillna(method='ffill', inplace=True)
        
        # Remove any remaining NaN rows
        df_clean.dropna(inplace=True)
        
        logger.info(f"✓ Data cleaned: {len(df)} -> {len(df_clean)} rows")
        return df_clean
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features
        
        Args:
            df: DataFrame with timestamp index
            
        Returns:
            DataFrame with temporal features
        """
        logger.info("Adding temporal features...")
        
        df_temporal = df.copy()
        
        # Extract time components
        df_temporal['hour'] = df_temporal.index.hour
        df_temporal['day'] = df_temporal.index.day
        df_temporal['day_of_week'] = df_temporal.index.dayofweek
        df_temporal['month'] = df_temporal.index.month
        df_temporal['quarter'] = df_temporal.index.quarter
        df_temporal['year'] = df_temporal.index.year
        
        # Cyclical encoding for time features
        df_temporal['hour_sin'] = np.sin(2 * np.pi * df_temporal['hour'] / 24)
        df_temporal['hour_cos'] = np.cos(2 * np.pi * df_temporal['hour'] / 24)
        df_temporal['day_sin'] = np.sin(2 * np.pi * df_temporal['day_of_week'] / 7)
        df_temporal['day_cos'] = np.cos(2 * np.pi * df_temporal['day_of_week'] / 7)
        df_temporal['month_sin'] = np.sin(2 * np.pi * df_temporal['month'] / 12)
        df_temporal['month_cos'] = np.cos(2 * np.pi * df_temporal['month'] / 12)
        
        # Time since start (in hours)
        df_temporal['hours_since_start'] = (df_temporal.index - df_temporal.index.min()).total_seconds() / 3600
        
        logger.info("✓ Added temporal features")
        return df_temporal
    
    def add_lag_features(self, df: pd.DataFrame, columns: List[str] = None, lags: List[int] = None) -> pd.DataFrame:
        """
        Add lagged versions of specified columns
        
        Args:
            df: Input DataFrame
            columns: Columns to create lags for (default: price, market_cap, total_volume)
            lags: List of lag periods (default: [1, 3, 6, 12, 24])
            
        Returns:
            DataFrame with lag features
        """
        logger.info("Adding lag features...")
        
        if columns is None:
            columns = ['price', 'market_cap', 'total_volume']
        
        if lags is None:
            lags = [1, 3, 6, 12, 24]  # 1h, 3h, 6h, 12h, 24h
        
        df_lags = df.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    lag_col_name = f"{col}_lag_{lag}h"
                    df_lags[lag_col_name] = df_lags[col].shift(lag)
        
        logger.info(f"✓ Added {len(columns) * len(lags)} lag features")
        return df_lags
    
    def add_rolling_features(self, df: pd.DataFrame, columns: List[str] = None, windows: List[int] = None) -> pd.DataFrame:
        """
        Add rolling window statistics
        
        Args:
            df: Input DataFrame
            columns: Columns to calculate rolling stats for
            windows: Rolling window sizes in hours
            
        Returns:
            DataFrame with rolling features
        """
        logger.info("Adding rolling window features...")
        
        if columns is None:
            columns = ['price', 'market_cap', 'total_volume']
        
        if windows is None:
            windows = [3, 6, 12, 24, 48]  # 3h, 6h, 12h, 24h, 48h
        
        df_rolling = df.copy()
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    # Rolling mean
                    df_rolling[f"{col}_rolling_mean_{window}h"] = df_rolling[col].rolling(window=window).mean()
                    
                    # Rolling standard deviation
                    df_rolling[f"{col}_rolling_std_{window}h"] = df_rolling[col].rolling(window=window).std()
                    
                    # Rolling min and max
                    df_rolling[f"{col}_rolling_min_{window}h"] = df_rolling[col].rolling(window=window).min()
                    df_rolling[f"{col}_rolling_max_{window}h"] = df_rolling[col].rolling(window=window).max()
        
        logger.info(f"✓ Added {len(columns) * len(windows) * 4} rolling features")
        return df_rolling
    
    def add_price_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price change and momentum features
        
        Args:
            df: DataFrame with price column
            
        Returns:
            DataFrame with price change features
        """
        logger.info("Adding price change features...")
        
        df_changes = df.copy()
        
        if 'price' in df.columns:
            # Price changes (absolute)
            df_changes['price_change_1h'] = df_changes['price'].diff()
            df_changes['price_change_3h'] = df_changes['price'].diff(3)
            df_changes['price_change_6h'] = df_changes['price'].diff(6)
            df_changes['price_change_24h'] = df_changes['price'].diff(24)
            
            # Price changes (percentage)
            df_changes['price_pct_change_1h'] = df_changes['price'].pct_change()
            df_changes['price_pct_change_3h'] = df_changes['price'].pct_change(3)
            df_changes['price_pct_change_6h'] = df_changes['price'].pct_change(6)
            df_changes['price_pct_change_24h'] = df_changes['price'].pct_change(24)
            
            # Volatility (rolling standard deviation of returns)
            df_changes['volatility_6h'] = df_changes['price_pct_change_1h'].rolling(window=6).std()
            df_changes['volatility_24h'] = df_changes['price_pct_change_1h'].rolling(window=24).std()
            
            # Momentum indicators
            df_changes['momentum_6h'] = df_changes['price'] / df_changes['price'].shift(6) - 1
            df_changes['momentum_24h'] = df_changes['price'] / df_changes['price'].shift(24) - 1
            
            # Price position within recent range
            df_changes['price_position_24h'] = (df_changes['price'] - df_changes['price'].rolling(24).min()) / (df_changes['price'].rolling(24).max() - df_changes['price'].rolling(24).min())
        
        logger.info("✓ Added price change and momentum features")
        return df_changes
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-related features
        
        Args:
            df: DataFrame with total_volume column
            
        Returns:
            DataFrame with volume features
        """
        logger.info("Adding volume features...")
        
        df_volume = df.copy()
        
        if 'total_volume' in df.columns:
            # Volume changes
            df_volume['volume_change_1h'] = df_volume['total_volume'].pct_change()
            df_volume['volume_change_6h'] = df_volume['total_volume'].pct_change(6)
            df_volume['volume_change_24h'] = df_volume['total_volume'].pct_change(24)
            
            # Volume moving averages
            df_volume['volume_ma_6h'] = df_volume['total_volume'].rolling(window=6).mean()
            df_volume['volume_ma_24h'] = df_volume['total_volume'].rolling(window=24).mean()
            
            # Volume ratios
            df_volume['volume_ratio_6h'] = df_volume['total_volume'] / df_volume['volume_ma_6h']
            df_volume['volume_ratio_24h'] = df_volume['total_volume'] / df_volume['volume_ma_24h']
            
            # Price-volume relationship
            if 'price' in df_volume.columns:
                df_volume['price_volume_corr_24h'] = df_volume['price'].rolling(window=24).corr(df_volume['total_volume'])
        
        logger.info("✓ Added volume features")
        return df_volume
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical analysis indicators
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with technical indicators
        """
        logger.info("Adding technical indicators...")
        
        df_tech = df.copy()
        
        if 'price' in df.columns:
            # Simple Moving Averages
            df_tech['sma_6h'] = df_tech['price'].rolling(window=6).mean()
            df_tech['sma_12h'] = df_tech['price'].rolling(window=12).mean()
            df_tech['sma_24h'] = df_tech['price'].rolling(window=24).mean()
            
            # Exponential Moving Averages
            df_tech['ema_6h'] = df_tech['price'].ewm(span=6).mean()
            df_tech['ema_12h'] = df_tech['price'].ewm(span=12).mean()
            df_tech['ema_24h'] = df_tech['price'].ewm(span=24).mean()
            
            # Bollinger Bands
            sma_20 = df_tech['price'].rolling(window=20).mean()
            std_20 = df_tech['price'].rolling(window=20).std()
            df_tech['bollinger_upper'] = sma_20 + (std_20 * 2)
            df_tech['bollinger_lower'] = sma_20 - (std_20 * 2)
            df_tech['bollinger_width'] = df_tech['bollinger_upper'] - df_tech['bollinger_lower']
            
            # Price position relative to moving averages
            df_tech['price_above_sma_6h'] = (df_tech['price'] > df_tech['sma_6h']).astype(int)
            df_tech['price_above_sma_24h'] = (df_tech['price'] > df_tech['sma_24h']).astype(int)
        
        logger.info("✓ Added technical indicators")
        return df_tech
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply complete transformation pipeline
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Fully transformed DataFrame with all features
        """
        logger.info("=" * 60)
        logger.info("STARTING DATA TRANSFORMATION")
        logger.info("=" * 60)
        
        # Clean data
        df_transformed = self.clean_data(df)
        
        # Add all feature types
        df_transformed = self.add_temporal_features(df_transformed)
        df_transformed = self.add_lag_features(df_transformed)
        df_transformed = self.add_rolling_features(df_transformed)
        df_transformed = self.add_price_change_features(df_transformed)
        df_transformed = self.add_volume_features(df_transformed)
        df_transformed = self.add_technical_indicators(df_transformed)
        
        # Remove rows with NaN values (due to lags and rolling windows)
        initial_rows = len(df_transformed)
        df_transformed.dropna(inplace=True)
        final_rows = len(df_transformed)
        
        logger.info("=" * 60)
        logger.info("✓ DATA TRANSFORMATION COMPLETE")
        logger.info(f"  Input rows: {len(df)}")
        logger.info(f"  Output rows: {final_rows}")
        logger.info(f"  Features created: {len(df_transformed.columns) - len(df.columns)}")
        logger.info(f"  Total features: {len(df_transformed.columns)}")
        logger.info("=" * 60)
        
        return df_transformed


def transform_data(input_filepath: str,
                  output_dir: str = "data/processed",
                  output_filename: str = None) -> str:
    """
    Main transformation function for Airflow tasks
    
    Args:
        input_filepath: Path to raw CSV file
        output_dir: Output directory for processed data
        output_filename: Custom output filename
        
    Returns:
        Path to processed data file
    """
    logger.info(f"Loading data from: {input_filepath}")
    
    # Load raw data
    df = pd.read_csv(input_filepath, index_col=0, parse_dates=True)
    
    # Initialize transformer
    transformer = CryptoDataTransformer()
    
    # Transform data
    df_transformed = transformer.transform(df)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename if not provided
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"processed_{timestamp}.csv"
    
    # Save processed data
    output_filepath = os.path.join(output_dir, output_filename)
    df_transformed.to_csv(output_filepath)
    
    logger.info(f"✓ Processed data saved to: {output_filepath}")
    return output_filepath


if __name__ == "__main__":
    # Test the transformer
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = transform_data(input_file)
        print(f"Transformed data saved to: {output_file}")
    else:
        print("Usage: python transform.py <input_filepath>")