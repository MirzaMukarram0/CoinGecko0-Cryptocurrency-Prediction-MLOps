"""
Test Model Module
Unit tests for model training functionality matching actual train.py implementation
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys

# Import modules to test
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.train import CryptoPricePredictor, setup_mlflow, train_with_mlflow


class TestCryptoPricePredictor:
    """Test cases for CryptoPricePredictor class"""
    
    @pytest.fixture
    def predictor(self, tmp_path):
        """Create a predictor instance with temp directory"""
        return CryptoPricePredictor(models_dir=str(tmp_path / 'models'))
    
    @pytest.fixture
    def sample_data(self):
        """Create sample processed data with features"""
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
        data['price_rolling_mean_24h'] = data['price'].rolling(window=24).mean()
        data['price_rolling_std_24h'] = data['price'].rolling(window=24).std()
        
        # Drop NaN rows
        data.dropna(inplace=True)
        
        return data
    
    @pytest.mark.unit
    def test_predictor_initialization(self, predictor):
        """Test predictor initializes correctly"""
        assert predictor.target_column == 'price'
        assert predictor.feature_columns == []
        assert predictor.trained_models == {}
        assert predictor.models_dir.exists()
    
    @pytest.mark.unit
    def test_prepare_features(self, predictor, sample_data):
        """Test feature preparation for training"""
        X, y = predictor.prepare_features(sample_data)
        
        # Check output types
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        
        # Check feature columns were stored
        assert predictor.feature_columns is not None
        assert len(predictor.feature_columns) > 0
    
    @pytest.mark.unit
    def test_model_configs_exist(self, predictor):
        """Test that model configurations are defined"""
        assert 'random_forest' in predictor.model_configs
        assert 'gradient_boosting' in predictor.model_configs
        assert 'ridge' in predictor.model_configs
        
        for config in predictor.model_configs.values():
            assert 'model_class' in config
            assert 'params' in config
    
    @pytest.mark.unit
    def test_prepare_features_handles_missing_values(self, predictor, sample_data):
        """Test that prepare_features handles missing values"""
        # Add some NaN values
        sample_data_with_nan = sample_data.copy()
        sample_data_with_nan.iloc[0, 0] = np.nan
        
        X, y = predictor.prepare_features(sample_data_with_nan)
        
        # Should have handled NaN values
        assert not X.isnull().any().any()


class TestTrainWithMLflow:
    """Test the train_with_mlflow function"""
    
    @pytest.mark.unit
    def test_train_with_mlflow_callable(self):
        """Test that train_with_mlflow is callable"""
        assert callable(train_with_mlflow)


class TestSetupMLflow:
    """Test MLflow setup function"""
    
    @pytest.mark.unit
    def test_setup_mlflow_callable(self):
        """Test that setup_mlflow is callable"""
        assert callable(setup_mlflow)


class TestModelConfigs:
    """Test model configuration"""
    
    @pytest.mark.unit
    def test_random_forest_config(self, tmp_path):
        """Test Random Forest model config"""
        predictor = CryptoPricePredictor(models_dir=str(tmp_path))
        config = predictor.model_configs['random_forest']
        
        assert 'n_estimators' in config['params']
        assert 'max_depth' in config['params']
        assert 'random_state' in config['params']
    
    @pytest.mark.unit
    def test_gradient_boosting_config(self, tmp_path):
        """Test Gradient Boosting model config"""
        predictor = CryptoPricePredictor(models_dir=str(tmp_path))
        config = predictor.model_configs['gradient_boosting']
        
        assert 'n_estimators' in config['params']
        assert 'learning_rate' in config['params']
        assert 'random_state' in config['params']
    
    @pytest.mark.unit
    def test_ridge_config(self, tmp_path):
        """Test Ridge model config"""
        predictor = CryptoPricePredictor(models_dir=str(tmp_path))
        config = predictor.model_configs['ridge']
        
        assert 'alpha' in config['params']


class TestErrorHandling:
    """Test error handling"""
    
    @pytest.mark.unit
    def test_predictor_creates_directory(self, tmp_path):
        """Test predictor creates models directory if not exists"""
        new_dir = tmp_path / 'new_models_dir'
        predictor = CryptoPricePredictor(models_dir=str(new_dir))
        
        assert new_dir.exists()
    
    @pytest.mark.unit
    def test_empty_dataframe_handling(self, tmp_path):
        """Test handling of empty dataframe"""
        predictor = CryptoPricePredictor(models_dir=str(tmp_path / 'models'))
        empty_df = pd.DataFrame()
        
        # Should handle empty dataframe gracefully
        try:
            X, y = predictor.prepare_features(empty_df)
        except (KeyError, ValueError):
            # Expected to fail with empty data
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
