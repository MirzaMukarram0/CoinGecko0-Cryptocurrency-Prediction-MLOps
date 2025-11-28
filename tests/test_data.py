"""
Test Data Module
Unit tests for data extraction, transformation, and quality checks
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Import modules to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.extract import CoinGeckoExtractor, extract_coingecko_data
from data.quality_checks import DataQualityChecker, check_data_quality, DataQualityException
from data.transform import CryptoDataTransformer, transform_data


class TestCoinGeckoExtractor:
    """Test cases for CoinGecko data extraction"""
    
    @pytest.fixture
    def extractor(self):
        return CoinGeckoExtractor()
    
    @pytest.fixture
    def sample_api_response(self):
        """Sample API response data"""
        base_time = 1640995200000  # 2022-01-01 00:00:00 UTC in milliseconds
        return {
            'prices': [
                [base_time + i * 3600000, 47000 + i * 100] for i in range(24)
            ],
            'market_caps': [
                [base_time + i * 3600000, 900000000000 + i * 1000000] for i in range(24)
            ],
            'total_volumes': [
                [base_time + i * 3600000, 20000000000 + i * 1000000] for i in range(24)
            ]
        }
    
    @patch('data.extract.requests.Session.get')
    def test_fetch_market_chart_success(self, mock_get, extractor, sample_api_response):
        """Test successful API data fetching"""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = sample_api_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test the method (3 args: coin_id, vs_currency, days)
        result = extractor.fetch_market_chart('bitcoin', 'usd', 1)
        
        # Assertions
        assert 'prices' in result
        assert 'market_caps' in result
        assert 'total_volumes' in result
        assert len(result['prices']) == 24
        mock_get.assert_called_once()
    
    def test_transform_to_dataframe(self, extractor, sample_api_response):
        """Test data transformation to DataFrame"""
        df = extractor.transform_to_dataframe(sample_api_response)
        
        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 24
        assert list(df.columns) == ['price', 'market_cap', 'total_volume']
        assert isinstance(df.index, pd.DatetimeIndex)
        
        # Check data types are numeric (int or float are both valid)
        assert np.issubdtype(df['price'].dtype, np.number)
        assert np.issubdtype(df['market_cap'].dtype, np.number)
        assert np.issubdtype(df['total_volume'].dtype, np.number)
    
    def test_save_raw_data(self, extractor, sample_api_response):
        """Test saving data to CSV"""
        df = extractor.transform_to_dataframe(sample_api_response)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = extractor.save_raw_data(df, temp_dir, 'test_data.csv')
            
            # Check file was created
            assert os.path.exists(filepath)
            
            # Check file contents
            loaded_df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            assert len(loaded_df) == len(df)
            assert list(loaded_df.columns) == list(df.columns)
    
    def test_transform_to_dataframe_empty_data(self, extractor):
        """Test handling of empty API response"""
        empty_data = {'prices': [], 'market_caps': [], 'total_volumes': []}
        
        with pytest.raises(ValueError, match="No price data found"):
            extractor.transform_to_dataframe(empty_data)


class TestDataQualityChecker:
    """Test cases for data quality checking"""
    
    @pytest.fixture
    def quality_checker(self):
        return DataQualityChecker(max_null_percentage=1.0)
    
    @pytest.fixture
    def good_data(self):
        """Create good quality test data"""
        dates = pd.date_range('2022-01-01', periods=100, freq='h')
        return pd.DataFrame({
            'price': np.random.uniform(45000, 50000, 100),
            'market_cap': np.random.uniform(900000000000, 950000000000, 100),
            'total_volume': np.random.uniform(20000000000, 25000000000, 100)
        }, index=dates)
    
    @pytest.fixture
    def bad_data_nulls(self):
        """Create data with too many null values"""
        dates = pd.date_range('2022-01-01', periods=100, freq='h')
        data = pd.DataFrame({
            'price': np.random.uniform(45000, 50000, 100),
            'market_cap': np.random.uniform(900000000000, 950000000000, 100),
            'total_volume': np.random.uniform(20000000000, 25000000000, 100)
        }, index=dates)
        
        # Introduce too many nulls (>1%) - use iloc for positional indexing
        data.iloc[:6, data.columns.get_loc('price')] = np.nan  # 6% null values
        return data
    
    def test_check_null_values_pass(self, quality_checker, good_data):
        """Test null value check with good data"""
        result = quality_checker.check_null_values(good_data)
        
        assert result['total_rows'] == 100
        assert all(count == 0 for count in result['null_counts'].values())
        assert all(pct == 0 for pct in result['null_percentages'].values())
        assert result['failed_columns'] == []
    
    def test_check_null_values_fail(self, quality_checker, bad_data_nulls):
        """Test null value check with bad data"""
        with pytest.raises(DataQualityException):
            quality_checker.check_null_values(bad_data_nulls)
    
    def test_check_schema_pass(self, quality_checker, good_data):
        """Test schema validation with correct columns"""
        expected_columns = ['price', 'market_cap', 'total_volume']
        result = quality_checker.check_schema(good_data, expected_columns)
        
        assert result['schema_valid'] is True
        assert result['missing_columns'] == []
    
    def test_check_schema_fail(self, quality_checker, good_data):
        """Test schema validation with missing columns"""
        expected_columns = ['price', 'market_cap', 'total_volume', 'missing_column']
        
        with pytest.raises(DataQualityException):
            quality_checker.check_schema(good_data, expected_columns)
    
    def test_check_data_types(self, quality_checker, good_data):
        """Test data type validation"""
        result = quality_checker.check_data_types(good_data)
        
        assert result['types_valid'] is True
        assert 'type_issues' in result
    
    def test_check_value_ranges(self, quality_checker, good_data):
        """Test value range validation"""
        result = quality_checker.check_value_ranges(good_data)
        
        assert result['ranges_valid'] is True
        assert result['range_issues'] == {}
    
    def test_run_all_checks_pass(self, quality_checker, good_data):
        """Test complete quality check suite with good data"""
        result = quality_checker.run_all_checks(good_data)
        
        assert result['overall_status'] == 'PASSED'
        assert 'dataset_info' in result
        assert result['dataset_info']['rows'] == 100


class TestCryptoDataTransformer:
    """Test cases for data transformation"""
    
    @pytest.fixture
    def transformer(self):
        return CryptoDataTransformer()
    
    @pytest.fixture
    def raw_data(self):
        """Create raw cryptocurrency data for testing"""
        dates = pd.date_range('2022-01-01', periods=100, freq='h')
        np.random.seed(42)  # For reproducible tests
        
        data = pd.DataFrame({
            'price': 45000 + np.cumsum(np.random.randn(100) * 100),
            'market_cap': 900000000000 + np.cumsum(np.random.randn(100) * 1000000000),
            'total_volume': 20000000000 + np.random.uniform(-1000000000, 1000000000, 100)
        }, index=dates)
        
        return data
    
    def test_clean_data(self, transformer, raw_data):
        """Test data cleaning functionality"""
        # Add some duplicates and nulls
        dirty_data = raw_data.copy()
        dirty_data.loc[dirty_data.index[0]] = np.nan  # Add null row
        dirty_data = pd.concat([dirty_data, dirty_data.iloc[:1]])  # Add duplicate
        
        cleaned = transformer.clean_data(dirty_data)
        
        # Check cleaning worked
        assert len(cleaned) <= len(dirty_data)
        assert not cleaned.isnull().any().any()
        assert not cleaned.index.duplicated().any()
    
    def test_add_temporal_features(self, transformer, raw_data):
        """Test temporal feature engineering"""
        result = transformer.add_temporal_features(raw_data)
        
        # Check new temporal columns exist
        temporal_cols = ['hour', 'day', 'day_of_week', 'month', 'quarter', 'year']
        cyclical_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
        
        for col in temporal_cols + cyclical_cols + ['hours_since_start']:
            assert col in result.columns
        
        # Check cyclical encoding ranges
        assert all(-1 <= result['hour_sin']) and all(result['hour_sin'] <= 1)
        assert all(-1 <= result['hour_cos']) and all(result['hour_cos'] <= 1)
    
    def test_add_lag_features(self, transformer, raw_data):
        """Test lag feature creation"""
        result = transformer.add_lag_features(raw_data)
        
        # Check lag columns exist
        lag_cols = [f'price_lag_{lag}h' for lag in [1, 3, 6, 12, 24]]
        for col in lag_cols:
            assert col in result.columns
        
        # Check lag values are correct
        assert result['price_lag_1h'].iloc[1] == result['price'].iloc[0]
    
    def test_add_rolling_features(self, transformer, raw_data):
        """Test rolling window feature creation"""
        result = transformer.add_rolling_features(raw_data)
        
        # Check rolling columns exist
        rolling_cols = [f'price_rolling_mean_{window}h' for window in [3, 6, 12, 24, 48]]
        for col in rolling_cols:
            assert col in result.columns
        
        # Check rolling calculations
        assert not result['price_rolling_mean_3h'].iloc[2:].isnull().any()
    
    def test_transform_complete_pipeline(self, transformer, raw_data):
        """Test complete transformation pipeline"""
        result = transformer.transform(raw_data)
        
        # Check that we have many more features
        assert len(result.columns) > len(raw_data.columns)
        
        # Check no null values in final result
        assert not result.isnull().any().any()
        
        # Check we still have reasonable amount of data
        assert len(result) > len(raw_data) * 0.5  # At least 50% of original data


class TestIntegrationFunctions:
    """Test integration of main functions"""
    
    @pytest.fixture
    def sample_csv_file(self):
        """Create a temporary CSV file for testing"""
        dates = pd.date_range('2022-01-01', periods=50, freq='h')
        data = pd.DataFrame({
            'price': np.random.uniform(45000, 50000, 50),
            'market_cap': np.random.uniform(900000000000, 950000000000, 50),
            'total_volume': np.random.uniform(20000000000, 25000000000, 50)
        }, index=dates)
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        data.to_csv(temp_file.name)
        temp_file.close()
        
        yield temp_file.name
        
        # Cleanup
        os.unlink(temp_file.name)
    
    def test_check_data_quality_function(self, sample_csv_file):
        """Test the main check_data_quality function"""
        result = check_data_quality(sample_csv_file, max_null_percentage=1.0)
        
        assert 'overall_status' in result
        assert 'dataset_info' in result
        assert result['dataset_info']['rows'] == 50
    
    def test_transform_data_function(self, sample_csv_file):
        """Test the main transform_data function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = transform_data(sample_csv_file, output_dir=temp_dir)
            
            # Check file was created
            assert os.path.exists(result_path)
            
            # Check transformed data
            df = pd.read_csv(result_path, index_col=0, parse_dates=True)
            assert len(df.columns) > 3  # Should have many more features


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__])
