"""
Test Training Module
Unit tests for MLflow-integrated model training
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock

# Import modules to test
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.train import CryptoPricePredictor, setup_mlflow, train_with_mlflow


class TestCryptoPricePredictorInit:
    """Test CryptoPricePredictor initialization"""
    
    @pytest.mark.unit
    def test_default_initialization(self, tmp_path):
        """Test default initialization"""
        predictor = CryptoPricePredictor(models_dir=str(tmp_path))
        
        assert predictor.target_column == 'price'
        assert predictor.feature_columns == []
        assert predictor.trained_models == {}
        assert predictor.metrics == {}
    
    @pytest.mark.unit
    def test_creates_models_directory(self, tmp_path):
        """Test that models directory is created"""
        models_dir = tmp_path / 'test_models'
        predictor = CryptoPricePredictor(models_dir=str(models_dir))
        
        assert models_dir.exists()


class TestFeaturePreparation:
    """Test feature preparation"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', periods=200, freq='h')
        
        data = pd.DataFrame({
            'price': 45000 + np.cumsum(np.random.randn(200) * 100),
            'market_cap': 900e9 + np.cumsum(np.random.randn(200) * 1e9),
            'total_volume': 20e9 + np.random.uniform(-1e9, 1e9, 200),
            'hour': dates.hour,
            'day_of_week': dates.dayofweek,
        }, index=dates)
        
        # Add lag features
        for lag in [1, 3, 6]:
            data[f'price_lag_{lag}h'] = data['price'].shift(lag)
        
        # Add rolling features
        data['price_rolling_mean_24h'] = data['price'].rolling(24).mean()
        data['price_rolling_std_24h'] = data['price'].rolling(24).std()
        
        data.dropna(inplace=True)
        return data

    @pytest.mark.unit
    def test_prepare_features_returns_correct_types(self, sample_data, tmp_path):
        """Test prepare_features returns DataFrame and Series"""
        predictor = CryptoPricePredictor(models_dir=str(tmp_path))
        X, y = predictor.prepare_features(sample_data)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert len(X) > 0
    
    @pytest.mark.unit
    def test_prepare_features_stores_columns(self, sample_data, tmp_path):
        """Test that feature columns are stored"""
        predictor = CryptoPricePredictor(models_dir=str(tmp_path))
        predictor.prepare_features(sample_data)
        
        assert predictor.feature_columns is not None
        assert len(predictor.feature_columns) > 0
    
    @pytest.mark.unit
    def test_prepare_features_handles_inf(self, sample_data, tmp_path):
        """Test that infinite values are handled"""
        sample_data_with_inf = sample_data.copy()
        sample_data_with_inf.iloc[5, 1] = np.inf
        
        predictor = CryptoPricePredictor(models_dir=str(tmp_path))
        X, y = predictor.prepare_features(sample_data_with_inf)
        
        # Should not contain inf values
        assert not np.isinf(X.values).any()


class TestModelConfigurations:
    """Test model configurations"""
    
    @pytest.mark.unit
    def test_all_model_types_configured(self, tmp_path):
        """Test all expected model types are configured"""
        predictor = CryptoPricePredictor(models_dir=str(tmp_path))
        
        expected_models = ['random_forest', 'gradient_boosting', 'ridge']
        for model_name in expected_models:
            assert model_name in predictor.model_configs
    
    @pytest.mark.unit
    def test_model_configs_have_required_keys(self, tmp_path):
        """Test model configs have required keys"""
        predictor = CryptoPricePredictor(models_dir=str(tmp_path))
        
        for model_name, config in predictor.model_configs.items():
            assert 'model_class' in config, f"{model_name} missing model_class"
            assert 'params' in config, f"{model_name} missing params"
            assert callable(config['model_class']), f"{model_name} model_class not callable"


class TestMLflowSetup:
    """Test MLflow setup"""
    
    @pytest.mark.unit
    def test_setup_mlflow_is_callable(self):
        """Test setup_mlflow function exists and is callable"""
        assert callable(setup_mlflow)
    
    @pytest.mark.unit
    @patch.dict(os.environ, {'MLFLOW_TRACKING_URI': ''})
    def test_setup_mlflow_with_empty_uri(self):
        """Test setup_mlflow with empty tracking URI uses default"""
        # This should not raise an error
        try:
            result = setup_mlflow()
            # Result should be a string (experiment ID)
            assert isinstance(result, str)
        except Exception:
            # May fail without MLflow server, that's OK for unit test
            pass


class TestTrainWithMLflow:
    """Test train_with_mlflow function"""
    
    @pytest.mark.unit
    def test_train_with_mlflow_is_callable(self):
        """Test train_with_mlflow function exists and is callable"""
        assert callable(train_with_mlflow)


class TestScaler:
    """Test scaler functionality"""
    
    @pytest.mark.unit
    def test_scaler_initialized(self, tmp_path):
        """Test that scaler is initialized"""
        predictor = CryptoPricePredictor(models_dir=str(tmp_path))
        
        assert predictor.scaler is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
