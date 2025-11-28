"""
Pytest Configuration and Shared Fixtures
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(scope="session")
def sample_raw_data():
    """Create sample raw cryptocurrency data"""
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=100, freq='H')
    
    return pd.DataFrame({
        'price': 45000 + np.cumsum(np.random.randn(100) * 100),
        'market_cap': 900000000000 + np.cumsum(np.random.randn(100) * 1000000000),
        'total_volume': 20000000000 + np.random.uniform(-1000000000, 1000000000, 100),
    }, index=dates)


@pytest.fixture(scope="session")
def sample_processed_data():
    """Create sample processed data with all features"""
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=200, freq='H')
    
    data = pd.DataFrame({
        'price': 45000 + np.cumsum(np.random.randn(200) * 100),
        'market_cap': 900000000000 + np.cumsum(np.random.randn(200) * 1000000000),
        'total_volume': 20000000000 + np.random.uniform(-1000000000, 1000000000, 200),
    }, index=dates)
    
    # Add temporal features
    data['hour'] = dates.hour
    data['day_of_week'] = dates.dayofweek
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
    
    # Add lag features
    for lag in [1, 3, 6, 12, 24]:
        data[f'price_lag_{lag}h'] = data['price'].shift(lag)
    
    # Add rolling features
    for window in [6, 12, 24]:
        data[f'price_rolling_mean_{window}h'] = data['price'].rolling(window=window).mean()
        data[f'price_rolling_std_{window}h'] = data['price'].rolling(window=window).std()
    
    # Add price change features
    data['price_change_1h'] = data['price'].pct_change(1) * 100
    data['price_change_24h'] = data['price'].pct_change(24) * 100
    
    # Drop NaN rows
    data.dropna(inplace=True)
    
    return data


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_api_response():
    """Sample CoinGecko API response"""
    base_time = 1640995200000  # 2022-01-01 00:00:00 UTC
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


@pytest.fixture
def quality_issues_data():
    """Create data with various quality issues for testing"""
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=50, freq='H')
    
    data = pd.DataFrame({
        'price': 45000 + np.cumsum(np.random.randn(50) * 100),
        'market_cap': 900000000000 + np.cumsum(np.random.randn(50) * 1000000000),
        'total_volume': 20000000000 + np.random.uniform(-1000000000, 1000000000, 50),
    }, index=dates)
    
    # Introduce quality issues
    data.loc[data.index[5], 'price'] = np.nan  # Missing value
    data.loc[data.index[10], 'price'] = -1000  # Negative value (invalid)
    data.loc[data.index[15], 'price'] = data['price'].iloc[14]  # Duplicate value
    
    return data


# Markers
def pytest_configure(config):
    """Configure custom markers"""
    config.addinivalue_line("markers", "unit: Unit tests (fast, no external dependencies)")
    config.addinivalue_line("markers", "integration: Integration tests (may require external services)")
    config.addinivalue_line("markers", "slow: Slow tests (skip with -m 'not slow')")


# Skip slow tests by default
def pytest_addoption(parser):
    """Add command line options"""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers"""
    if config.getoption("--run-slow"):
        return
    
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
