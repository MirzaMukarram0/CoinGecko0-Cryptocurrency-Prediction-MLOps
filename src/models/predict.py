"""
Model Prediction Module
Handles cryptocurrency price predictions using trained models
"""
import pandas as pd
import numpy as np
import logging
import joblib
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CryptoPricePredictor:
    """
    Cryptocurrency price prediction using trained models
    """
    
    def __init__(self):
        """Initialize the predictor"""
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.model_type = None
        
    def load_model(self, model_path: str, scaler_path: str, features_path: str):
        """
        Load trained model, scaler, and feature columns
        
        Args:
            model_path: Path to saved model
            scaler_path: Path to saved scaler
            features_path: Path to saved feature columns
        """
        logger.info("Loading model components...")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_columns = joblib.load(features_path)
        
        # Extract model type from filename
        self.model_type = os.path.basename(model_path).split('_')[0]
        
        logger.info(f"✓ Model loaded: {self.model_type}")
        logger.info(f"  Features: {len(self.feature_columns)}")
        
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for prediction
        
        Args:
            df: DataFrame with processed features
            
        Returns:
            Scaled feature array
        """
        # Ensure we have the required features
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select and order features
        features_df = df[self.feature_columns]
        
        # Handle any remaining NaN values
        features_df = features_df.fillna(method='ffill').fillna(method='bfill')
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        return features_scaled
    
    def predict_single(self, df: pd.DataFrame, timestamp: pd.Timestamp = None) -> Dict[str, Any]:
        """
        Make prediction for a single timestamp
        
        Args:
            df: DataFrame with features
            timestamp: Specific timestamp to predict (latest if None)
            
        Returns:
            Prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if timestamp is None:
            timestamp = df.index[-1]
        
        if timestamp not in df.index:
            raise ValueError(f"Timestamp {timestamp} not found in data")
        
        # Get features for the timestamp
        features_row = df.loc[[timestamp]]
        features_scaled = self.prepare_features(features_row)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        
        # Get current price for comparison
        current_price = df.loc[timestamp, 'price'] if 'price' in df.columns else None
        
        result = {
            'timestamp': timestamp.isoformat(),
            'predicted_price': float(prediction),
            'current_price': float(current_price) if current_price is not None else None,
            'model_type': self.model_type
        }
        
        if current_price is not None:
            result['predicted_change'] = float(prediction - current_price)
            result['predicted_change_pct'] = float((prediction - current_price) / current_price * 100)
        
        return result
    
    def predict_batch(self, df: pd.DataFrame, start_time: pd.Timestamp = None, end_time: pd.Timestamp = None) -> pd.DataFrame:
        """
        Make predictions for a batch of timestamps
        
        Args:
            df: DataFrame with features
            start_time: Start timestamp (first available if None)
            end_time: End timestamp (last available if None)
            
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Filter dataframe by time range
        if start_time is not None:
            df = df[df.index >= start_time]
        if end_time is not None:
            df = df[df.index <= end_time]
        
        if len(df) == 0:
            raise ValueError("No data available for the specified time range")
        
        # Prepare features
        features_scaled = self.prepare_features(df)
        
        # Make predictions
        predictions = self.model.predict(features_scaled)
        
        # Create results dataframe
        results_df = pd.DataFrame(index=df.index)
        results_df['predicted_price'] = predictions
        results_df['model_type'] = self.model_type
        
        # Add current prices and changes if available
        if 'price' in df.columns:
            results_df['current_price'] = df['price']
            results_df['predicted_change'] = results_df['predicted_price'] - results_df['current_price']
            results_df['predicted_change_pct'] = (results_df['predicted_change'] / results_df['current_price']) * 100
        
        return results_df
    
    def predict_future(self, df: pd.DataFrame, hours_ahead: int = 1) -> List[Dict[str, Any]]:
        """
        Predict future prices (experimental - recursive prediction)
        
        Args:
            df: DataFrame with features
            hours_ahead: Number of hours to predict ahead
            
        Returns:
            List of future predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.warning("Future prediction is experimental and may be inaccurate for long horizons")
        
        # Start with the latest data point
        current_df = df.copy()
        future_predictions = []
        
        for hour in range(1, hours_ahead + 1):
            # Get latest features
            latest_timestamp = current_df.index[-1]
            features_scaled = self.prepare_features(current_df.tail(1))
            
            # Predict next hour
            prediction = self.model.predict(features_scaled)[0]
            
            # Create timestamp for prediction
            future_timestamp = latest_timestamp + timedelta(hours=1)
            
            # Store prediction
            prediction_result = {
                'timestamp': future_timestamp.isoformat(),
                'predicted_price': float(prediction),
                'hours_ahead': hour,
                'model_type': self.model_type
            }
            future_predictions.append(prediction_result)
            
            # Create new row for next iteration (this is highly experimental)
            # In practice, you would need actual feature values, not predictions
            new_row = current_df.iloc[-1:].copy()
            new_row.index = [future_timestamp]
            new_row['price'] = prediction  # Use prediction as new "current" price
            
            # Update other features based on the new price (simplified approach)
            if 'price_change_1h' in new_row.columns:
                new_row['price_change_1h'] = prediction - current_df['price'].iloc[-1]
            if 'price_pct_change_1h' in new_row.columns:
                new_row['price_pct_change_1h'] = (prediction - current_df['price'].iloc[-1]) / current_df['price'].iloc[-1]
            
            # Append to dataframe
            current_df = pd.concat([current_df, new_row])
        
        return future_predictions
    
    def evaluate_predictions(self, df: pd.DataFrame, actual_column: str = 'price') -> Dict[str, float]:
        """
        Evaluate prediction accuracy against actual values
        
        Args:
            df: DataFrame with predictions and actual values
            actual_column: Column name with actual values
            
        Returns:
            Dictionary with evaluation metrics
        """
        if 'predicted_price' not in df.columns:
            raise ValueError("DataFrame must contain 'predicted_price' column")
        
        if actual_column not in df.columns:
            raise ValueError(f"DataFrame must contain '{actual_column}' column")
        
        # Remove rows with missing values
        evaluation_df = df[['predicted_price', actual_column]].dropna()
        
        if len(evaluation_df) == 0:
            raise ValueError("No valid prediction-actual pairs found")
        
        actual = evaluation_df[actual_column]
        predicted = evaluation_df['predicted_price']
        
        # Calculate metrics
        mse = np.mean((actual - predicted) ** 2)
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # R-squared
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Directional accuracy (for price change predictions)
        if len(evaluation_df) > 1:
            actual_direction = np.sign(actual.diff().dropna())
            predicted_direction = np.sign(predicted.diff().dropna())
            directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
        else:
            directional_accuracy = None
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2),
            'sample_count': len(evaluation_df)
        }
        
        if directional_accuracy is not None:
            metrics['directional_accuracy'] = float(directional_accuracy)
        
        return metrics


def make_predictions(data_filepath: str,
                    model_path: str,
                    scaler_path: str,
                    features_path: str,
                    output_dir: str = "predictions") -> str:
    """
    Main prediction function for Airflow tasks
    
    Args:
        data_filepath: Path to processed data CSV
        model_path: Path to trained model
        scaler_path: Path to model scaler
        features_path: Path to feature columns
        output_dir: Output directory for predictions
        
    Returns:
        Path to predictions file
    """
    logger.info("=" * 60)
    logger.info("STARTING PREDICTION GENERATION")
    logger.info("=" * 60)
    
    # Load data
    df = pd.read_csv(data_filepath, index_col=0, parse_dates=True)
    logger.info(f"Loaded data: {df.shape}")
    
    # Initialize predictor
    predictor = CryptoPricePredictor()
    
    # Load model
    predictor.load_model(model_path, scaler_path, features_path)
    
    # Make batch predictions
    predictions_df = predictor.predict_batch(df)
    
    # Evaluate if we have actual prices
    if 'current_price' in predictions_df.columns:
        metrics = predictor.evaluate_predictions(predictions_df, 'current_price')
        logger.info(f"Prediction metrics: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save predictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"predictions_{timestamp}.csv"
    output_filepath = os.path.join(output_dir, output_filename)
    
    predictions_df.to_csv(output_filepath)
    
    logger.info("=" * 60)
    logger.info("✓ PREDICTION GENERATION COMPLETE")
    logger.info(f"  Predictions saved to: {output_filepath}")
    logger.info("=" * 60)
    
    return output_filepath


if __name__ == "__main__":
    # Test the predictor
    import sys
    
    if len(sys.argv) >= 5:
        data_file = sys.argv[1]
        model_file = sys.argv[2]
        scaler_file = sys.argv[3]
        features_file = sys.argv[4]
        
        predictions_file = make_predictions(data_file, model_file, scaler_file, features_file)
        print(f"Predictions saved to: {predictions_file}")
    else:
        print("Usage: python predict.py <data_filepath> <model_path> <scaler_path> <features_path>")