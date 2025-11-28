"""
Test Training Module
Unit tests specifically for MLflow-integrated model training
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock

# Import modules to test
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.train import CryptoPricePredictor, setup_mlflow


class TestCryptoPricePredictor:
    """Test cases for CryptoPricePredictor class"""
    
    @pytest.fixture
    def predictor(self):
        """Create a predictor instance"""
        return CryptoPricePredictor(target_column='price', lookback_hours=24)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', periods=200, freq='H')
        
        data = pd.DataFrame({
            'price': 45000 + np.cumsum(np.random.randn(200) * 100),
            'market_cap': 900e9 + np.cumsum(np.random.randn(200) * 1e9),
            'total_volume': 20e9 + np.random.uniform(-1e9, 1e9, 200),
        }, index=dates)
        
        # Add temporal features
        data['hour'] = dates.hour
        data['day_of_week'] = dates.dayofweek
        
        # Add lag features
        for lag in [1, 3, 6]:
            data[f'price_lag_{lag}h'] = data['price'].shift(lag)
        
        # Add rolling features
        data['price_rolling_mean_24h'] = data['price'].rolling(24).mean()
        data['price_rolling_std_24h'] = data['price'].rolling(24).std()
        
        data.dropna(inplace=True)
        return data

    @pytest.mark.unit
    def test_predictor_initialization(self, predictor):
        """Test predictor initializes correctly"""
        assert predictor.target_column == 'price'
        assert predictor.lookback_hours == 24
        assert predictor.models == {}
        assert predictor.scalers == {}
    
    @pytest.mark.unit
    def test_prepare_features_returns_arrays(self, predictor, sample_data):
        """Test prepare_features returns numpy arrays"""
        X, y = predictor.prepare_features(sample_data)
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(X) == len(y)
        assert len(X) > 0
    
    @pytest.mark.unit
    def test_prepare_features_excludes_target(self, predictor, sample_data):
        """Test that target column is not in features"""
        X, y = predictor.prepare_features(sample_data)
        
        # Feature columns should be stored
        assert predictor.feature_columns is not None
        assert 'price' not in predictor.feature_columns
    
    @pytest.mark.unit
    def test_train_random_forest(self, predictor, sample_data):
        """Test Random Forest model training"""
        X, y = predictor.prepare_features(sample_data)
        result = predictor.train_model(X, y, 'random_forest')
        
        assert result['model_type'] == 'random_forest'
        assert 'final_metrics' in result
        assert 'random_forest' in predictor.models
    
    @pytest.mark.unit
    def test_train_gradient_boosting(self, predictor, sample_data):
        """Test Gradient Boosting model training"""
        X, y = predictor.prepare_features(sample_data)
        result = predictor.train_model(X, y, 'gradient_boosting')
        
        assert result['model_type'] == 'gradient_boosting'
        assert 'gradient_boosting' in predictor.models
    
    @pytest.mark.unit
    def test_train_ridge(self, predictor, sample_data):
        """Test Ridge regression model training"""
        X, y = predictor.prepare_features(sample_data)
        result = predictor.train_model(X, y, 'ridge')
        
        assert result['model_type'] == 'ridge'
        assert 'ridge' in predictor.models
    
    @pytest.mark.unit
    def test_metrics_calculated(self, predictor, sample_data):
        """Test that all metrics are calculated"""
        X, y = predictor.prepare_features(sample_data)
        result = predictor.train_model(X, y, 'random_forest')
        
        metrics = result['final_metrics']
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'rmse' in metrics
        
        # Check metrics are reasonable
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['mse'] >= 0
    
    @pytest.mark.unit
    def test_cross_validation_performed(self, predictor, sample_data):
        """Test that cross-validation is performed"""
        X, y = predictor.prepare_features(sample_data)
        result = predictor.train_model(X, y, 'random_forest')
        
        cv_metrics = result['cv_metrics']
        assert 'cv_rmse_mean' in cv_metrics
        assert 'cv_rmse_std' in cv_metrics
    
    @pytest.mark.unit
    def test_predict_after_training(self, predictor, sample_data):
        """Test prediction after training"""
        X, y = predictor.prepare_features(sample_data)
        predictor.train_model(X, y, 'random_forest')
        
        # Predict on a subset
        predictions = predictor.predict(X[:10], 'random_forest')
        
        assert len(predictions) == 10
        assert all(pred > 0 for pred in predictions)  # Price should be positive
    
    @pytest.mark.unit  
    def test_invalid_model_type(self, predictor, sample_data):
        """Test that invalid model type raises error"""
        X, y = predictor.prepare_features(sample_data)
        
        with pytest.raises((ValueError, KeyError)):
            predictor.train_model(X, y, 'invalid_model_type')


class TestMLflowSetup:
    """Test cases for MLflow setup function"""
    
    @pytest.mark.unit
    @patch.dict(os.environ, {'MLFLOW_TRACKING_URI': 'http://test-server'})
    def test_setup_mlflow_with_env_var(self):
        """Test MLflow setup reads from environment"""
        # This should not raise an error
        setup_mlflow()
    
    @pytest.mark.unit
    @patch.dict(os.environ, {}, clear=True)
    def test_setup_mlflow_without_env_var(self):
        """Test MLflow setup handles missing env var"""
        # Should handle gracefully
        setup_mlflow()


class TestFeatureImportance:
    """Test feature importance extraction"""
    
    @pytest.fixture
    def trained_predictor(self, sample_data_fixture):
        """Create a trained predictor"""
        predictor = CryptoPricePredictor(target_column='price')
        X, y = predictor.prepare_features(sample_data_fixture)
        predictor.train_model(X, y, 'random_forest')
        return predictor
    
    @pytest.fixture
    def sample_data_fixture(self):
        """Sample data for feature importance tests"""
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', periods=200, freq='H')
        
        data = pd.DataFrame({
            'price': 45000 + np.cumsum(np.random.randn(200) * 100),
            'market_cap': 900e9 + np.cumsum(np.random.randn(200) * 1e9),
            'total_volume': 20e9 + np.random.uniform(-1e9, 1e9, 200),
            'hour': dates.hour,
            'day_of_week': dates.dayofweek,
        }, index=dates)
        
        for lag in [1, 3, 6]:
            data[f'price_lag_{lag}h'] = data['price'].shift(lag)
        
        data['price_rolling_mean_24h'] = data['price'].rolling(24).mean()
        data.dropna(inplace=True)
        return data
    
    @pytest.mark.unit
    def test_feature_importance_exists(self, trained_predictor):
        """Test that feature importance can be extracted"""
        model = trained_predictor.models.get('random_forest')
        assert model is not None
        
        # Random Forest has feature_importances_
        assert hasattr(model, 'feature_importances_')
        importances = model.feature_importances_
        
        assert len(importances) == len(trained_predictor.feature_columns)
        assert sum(importances) > 0  # Importances should be positive
