"""
CoinGecko Data Extraction Module
Handles cryptocurrency data extraction from CoinGecko API
"""
import os
import requests
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoinGeckoExtractor:
    """
    Cryptocurrency data extractor using CoinGecko API
    """
    
    def __init__(self, base_url: str = "https://api.coingecko.com/api/v3", api_key: Optional[str] = None):
        """
        Initialize the extractor
        
        Args:
            base_url: CoinGecko API base URL
            api_key: CoinGecko API key (optional, for higher rate limits)
        """
        self.base_url = base_url
        self.api_key = api_key or os.getenv('COINGECKO_API_KEY')
        self.session = requests.Session()
        
    def fetch_market_chart(self, 
                          coin_id: str,
                          vs_currency: str = "usd",
                          days: int = 1) -> Dict:
        """
        Fetch market chart data from CoinGecko API
        
        Args:
            coin_id: Cryptocurrency ID (e.g., 'bitcoin')
            vs_currency: Target currency (default: 'usd')
            days: Number of days (1-365)
            
        Returns:
            Dictionary with price, market_cap, and total_volume data
        """
        logger.info(f"Fetching {coin_id} data for {days} days...")
        
        url = f"{self.base_url}/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": vs_currency,
            "days": days
        }
        
        # Prepare headers with API key if available
        headers = {}
        if self.api_key:
            headers['x-cg-pro-api-key'] = self.api_key
        
        try:
            response = self.session.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"✓ Successfully fetched {len(data['prices'])} data points")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    def transform_to_dataframe(self, market_data: Dict) -> pd.DataFrame:
        """
        Transform market chart data to pandas DataFrame
        
        Args:
            market_data: Raw market data from API
            
        Returns:
            Cleaned and structured DataFrame
        """
        logger.info("Transforming data to DataFrame...")
        
        # Extract data arrays
        prices = market_data.get('prices', [])
        market_caps = market_data.get('market_caps', [])
        volumes = market_data.get('total_volumes', [])
        
        if not prices:
            raise ValueError("No price data found in API response")
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': [pd.to_datetime(item[0], unit='ms') for item in prices],
            'price': [item[1] for item in prices],
            'market_cap': [item[1] if item else None for item in market_caps],
            'total_volume': [item[1] if item else None for item in volumes]
        })
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        logger.info(f"✓ Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def save_raw_data(self, 
                     df: pd.DataFrame, 
                     output_dir: str = "data/raw",
                     filename: str = None) -> str:
        """
        Save raw data to CSV file
        
        Args:
            df: DataFrame to save
            output_dir: Output directory
            filename: Custom filename (optional)
            
        Returns:
            Path to saved file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"raw_{timestamp}.csv"
        
        filepath = os.path.join(output_dir, filename)
        
        # Save to CSV
        df.to_csv(filepath)
        logger.info(f"✓ Saved raw data to: {filepath}")
        
        return filepath


def extract_coingecko_data(coin_id: str = "bitcoin",
                          vs_currency: str = "usd", 
                          days: int = 1,
                          output_dir: str = "data/raw",
                          api_key: Optional[str] = None) -> str:
    """
    Main extraction function for Airflow tasks
    
    Args:
        coin_id: Cryptocurrency to extract
        vs_currency: Target currency
        days: Number of days to fetch
        output_dir: Output directory for raw data
        api_key: CoinGecko API key (optional)
        
    Returns:
        Path to saved data file
    """
    logger.info("=" * 60)
    logger.info("STARTING DATA EXTRACTION")
    logger.info("=" * 60)
    
    try:
        # Initialize extractor with API key
        extractor = CoinGeckoExtractor(api_key=api_key)
        
        # Fetch data
        market_data = extractor.fetch_market_chart(
            coin_id=coin_id,
            vs_currency=vs_currency,
            days=days
        )
        
        # Transform to DataFrame
        df = extractor.transform_to_dataframe(market_data)
        
        # Save raw data
        filepath = extractor.save_raw_data(df, output_dir)
        
        logger.info("=" * 60)
        logger.info("✓ DATA EXTRACTION COMPLETE")
        logger.info(f"  File: {filepath}")
        logger.info(f"  Rows: {len(df)}")
        logger.info(f"  Columns: {list(df.columns)}")
        logger.info("=" * 60)
        
        return filepath
        
    except Exception as e:
        logger.error(f"✗ DATA EXTRACTION FAILED: {e}")
        raise


if __name__ == "__main__":
    # Test the extractor
    filepath = extract_coingecko_data(
        coin_id="bitcoin",
        vs_currency="usd",
        days=1,
        interval="hourly"
    )
    print(f"Data saved to: {filepath}")