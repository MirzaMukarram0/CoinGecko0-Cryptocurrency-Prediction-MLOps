"""
Test Model Module
Unit tests for model training and prediction functionality
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import joblib
from unittest.mock import patch, MagicMock

# Import modules to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.train import CryptoPricePredictor as TrainPredictor, train_prediction_model
from models.predict import CryptoPricePredictor as PredictPredictor, make_predictions


class TestModelTraining:
    """Test cases for model training functionality"""
    
    @pytest.fixture
    def predictor(self):
        return TrainPredictor(target_column='price', lookback_hours=24)
    
    @pytest.fixture
    def sample_processed_data(self):
        """Create sample processed data with features"""
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', periods=200, freq='H')
        
        data = pd.DataFrame({
            'price': 45000 + np.cumsum(np.random.randn(200) * 100),
            'market_cap': 900000000000 + np.cumsum(np.random.randn(200) * 1000000000),
            'total_volume': 20000000000 + np.random.uniform(-1000000000, 1000000000, 200),
            # Add some lag features
            'price_lag_1h': np.nan,
            'price_lag_3h': np.nan,
            'price_lag_6h': np.nan,
            # Add temporal features
            'hour': dates.hour,
            'day_of_week': dates.dayofweek,
            # Add rolling features
            'price_rolling_mean_24h': np.nan,
            'price_rolling_std_24h': np.nan,
        }, index=dates)
        
        # Fill lag features properly
        for lag in [1, 3, 6]:
            data[f'price_lag_{lag}h'] = data['price'].shift(lag)
        
        # Fill rolling features
        data['price_rolling_mean_24h'] = data['price'].rolling(window=24).mean()
        data['price_rolling_std_24h'] = data['price'].rolling(window=24).std()
        
        # Drop NaN rows
        data.dropna(inplace=True)
        
        return data
    
    def test_prepare_features(self, predictor, sample_processed_data):
        """Test feature preparation for training"""
        features, targets = predictor.prepare_features(sample_processed_data)
        
        # Check output shapes
        assert isinstance(features, np.ndarray)
        assert isinstance(targets, np.ndarray)
        assert len(features) == len(targets)
        assert len(features) < len(sample_processed_data)  # Should be less due to target shift
        
        # Check feature columns were stored
        assert predictor.feature_columns is not None
        assert len(predictor.feature_columns) == features.shape[1]
    
    def test_train_random_forest_model(self, predictor, sample_processed_data):
        """Test Random Forest model training"""
        features, targets = predictor.prepare_features(sample_processed_data)
        
        result = predictor.train_model(features, targets, 'random_forest')
        
        # Check training results
        assert 'model_type' in result
        assert result['model_type'] == 'random_forest'
        assert 'final_metrics' in result
        assert 'cv_metrics' in result
        
        # Check metrics exist
        assert 'mse' in result['final_metrics']
        assert 'mae' in result['final_metrics']
        assert 'r2' in result['final_metrics']
        assert 'rmse' in result['final_metrics']
        
        # Check model was stored
        assert 'random_forest' in predictor.models
        assert 'random_forest' in predictor.scalers
    
    def test_train_gradient_boosting_model(self, predictor, sample_processed_data):
        """Test Gradient Boosting model training"""
        features, targets = predictor.prepare_features(sample_processed_data)
        
        result = predictor.train_model(features, targets, 'gradient_boosting')
        
        # Check training results
        assert result['model_type'] == 'gradient_boosting'
        assert 'gradient_boosting' in predictor.models
    
    def test_get_feature_importance(self, predictor, sample_processed_data):
        """Test feature importance extraction"""
        features, targets = predictor.prepare_features(sample_processed_data)
        predictor.train_model(features, targets, 'random_forest')
        
        importance_df = predictor.get_feature_importance('random_forest')
        
        assert isinstance(importance_df, pd.DataFrame)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert len(importance_df) > 0
    
    def test_save_and_load_model(self, predictor, sample_processed_data):
        """Test model saving and loading"""
        features, targets = predictor.prepare_features(sample_processed_data)
        predictor.train_model(features, targets, 'random_forest')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model
            model_path = predictor.save_model('random_forest', temp_dir)
            
            # Check files were created
            assert os.path.exists(model_path)
            
            # Check corresponding scaler and features files exist
            timestamp = model_path.split('_')[-1].replace('.joblib', '')
            scaler_path = os.path.join(temp_dir, f"random_forest_scaler_{timestamp}.joblib")
            features_path = os.path.join(temp_dir, f"random_forest_features_{timestamp}.joblib")
            
            assert os.path.exists(scaler_path)
            assert os.path.exists(features_path)
            
            # Test loading
            new_predictor = TrainPredictor()
            new_predictor.load_model(model_path, scaler_path, features_path)
            
            assert 'random_forest' in new_predictor.models
            assert 'random_forest' in new_predictor.scalers
            assert new_predictor.feature_columns is not None


class TestModelPrediction:
    """Test cases for model prediction functionality"""
    
    @pytest.fixture
    def predictor(self):
        return PredictPredictor()
    
    @pytest.fixture
    def trained_model_files(self, sample_processed_data):
        """Create trained model files for testing"""
        trainer = TrainPredictor(target_column='price', lookback_hours=24)
        features, targets = trainer.prepare_features(sample_processed_data)
        trainer.train_model(features, targets, 'random_forest')
        
        temp_dir = tempfile.mkdtemp()
        model_path = trainer.save_model('random_forest', temp_dir)
        
        timestamp = model_path.split('_')[-1].replace('.joblib', '')
        scaler_path = os.path.join(temp_dir, f"random_forest_scaler_{timestamp}.joblib")
        features_path = os.path.join(temp_dir, f"random_forest_features_{timestamp}.joblib")
        
        return model_path, scaler_path, features_path, temp_dir
    
    @pytest.fixture
    def sample_processed_data(self):
        """Create sample processed data with features (same as training)"""
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', periods=200, freq='H')
        
        data = pd.DataFrame({
            'price': 45000 + np.cumsum(np.random.randn(200) * 100),
            'market_cap': 900000000000 + np.cumsum(np.random.randn(200) * 1000000000),
            'total_volume': 20000000000 + np.random.uniform(-1000000000, 1000000000, 200),
            'price_lag_1h': np.nan,
            'price_lag_3h': np.nan,
            'price_lag_6h': np.nan,
            'hour': dates.hour,
            'day_of_week': dates.dayofweek,
            'price_rolling_mean_24h': np.nan,
            'price_rolling_std_24h': np.nan,
        }, index=dates)
        
        # Fill lag features
        for lag in [1, 3, 6]:
            data[f'price_lag_{lag}h'] = data['price'].shift(lag)
        
        # Fill rolling features
        data['price_rolling_mean_24h'] = data['price'].rolling(window=24).mean()
        data['price_rolling_std_24h'] = data['price'].rolling(window=24).std()
        
        data.dropna(inplace=True)
        return data
    
    def test_load_model(self, predictor, trained_model_files):
        """Test model loading"""
        model_path, scaler_path, features_path, temp_dir = trained_model_files
        
        predictor.load_model(model_path, scaler_path, features_path)
        
        assert predictor.model is not None
        assert predictor.scaler is not None
        assert predictor.feature_columns is not None
        assert predictor.model_type == 'random_forest'
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_predict_single(self, predictor, trained_model_files, sample_processed_data):
        """Test single timestamp prediction"""
        model_path, scaler_path, features_path, temp_dir = trained_model_files
        predictor.load_model(model_path, scaler_path, features_path)
        
        # Make prediction
        result = predictor.predict_single(sample_processed_data)
        
        # Check result structure
        assert 'timestamp' in result
        assert 'predicted_price' in result
        assert 'current_price' in result
        assert 'model_type' in result
        assert result['model_type'] == 'random_forest'
        
        # Check prediction is reasonable
        assert isinstance(result['predicted_price'], float)
        assert result['predicted_price'] > 0
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_predict_batch(self, predictor, trained_model_files, sample_processed_data):
        """Test batch prediction"""
        model_path, scaler_path, features_path, temp_dir = trained_model_files
        predictor.load_model(model_path, scaler_path, features_path)
        
        # Make batch predictions
        results_df = predictor.predict_batch(sample_processed_data)
        
        # Check results structure
        assert isinstance(results_df, pd.DataFrame)
        assert 'predicted_price' in results_df.columns
        assert 'model_type' in results_df.columns
        assert len(results_df) > 0
        
        # Check all predictions are reasonable
        assert all(results_df['predicted_price'] > 0)
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_evaluate_predictions(self, predictor, trained_model_files, sample_processed_data):
        """Test prediction evaluation"""
        model_path, scaler_path, features_path, temp_dir = trained_model_files
        predictor.load_model(model_path, scaler_path, features_path)
        
        # Make predictions
        results_df = predictor.predict_batch(sample_processed_data)
        
        # Evaluate predictions
        metrics = predictor.evaluate_predictions(results_df, 'current_price')
        
        # Check metrics
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert 'sample_count' in metrics
        
        # Check metric values are reasonable
        assert metrics['sample_count'] > 0
        assert metrics['rmse'] > 0
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)


class TestIntegrationFunctions:
    """Test integration functions"""
    
    @pytest.fixture
    def sample_csv_file(self):
        """Create a temporary CSV file for testing"""
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', periods=100, freq='H')
        
        data = pd.DataFrame({
            'price': 45000 + np.cumsum(np.random.randn(100) * 100),
            'market_cap': 900000000000 + np.cumsum(np.random.randn(100) * 1000000000),
            'total_volume': 20000000000 + np.random.uniform(-1000000000, 1000000000, 100),
            'price_lag_1h': np.nan,
            'hour': dates.hour,
            'day_of_week': dates.dayofweek,
            'price_rolling_mean_24h': np.nan,
        }, index=dates)
        
        # Fill some features
        data['price_lag_1h'] = data['price'].shift(1)
        data['price_rolling_mean_24h'] = data['price'].rolling(window=24).mean()
        data.dropna(inplace=True)
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        data.to_csv(temp_file.name)
        temp_file.close()
        
        yield temp_file.name
        
        # Cleanup
        os.unlink(temp_file.name)
    
    def test_train_prediction_model_function(self, sample_csv_file):
        """Test the main train_prediction_model function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = train_prediction_model(
                data_filepath=sample_csv_file,
                model_types=['random_forest'],
                output_dir=temp_dir
            )
            
            # Check results structure
            assert 'random_forest' in result
            assert 'final_metrics' in result['random_forest']
            assert 'model_path' in result['random_forest']
            
            # Check model files were created
            model_path = result['random_forest']['model_path']
            assert os.path.exists(model_path)


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_model_type(self):
        """Test training with invalid model type"""
        predictor = TrainPredictor()
        features = np.random.randn(100, 5)
        targets = np.random.randn(100)
        
        with pytest.raises(ValueError, match="Unsupported model type"):
            predictor.train_model(features, targets, 'invalid_model')
    
    def test_missing_target_column(self):
        """Test feature preparation with missing target column"""
        predictor = TrainPredictor(target_column='missing_column')
        data = pd.DataFrame({'price': [1, 2, 3], 'volume': [100, 200, 300]})
        
        with pytest.raises(ValueError, match="Target column"):
            predictor.prepare_features(data)
    
    def test_prediction_without_loaded_model(self):
        """Test prediction without loading model"""
        predictor = PredictPredictor()
        data = pd.DataFrame({'price': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Model not loaded"):
            predictor.predict_single(data)


if __name__ == "__main__":
    pytest.main([__file__])