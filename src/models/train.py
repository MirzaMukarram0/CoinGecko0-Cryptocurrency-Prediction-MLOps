"""
Model Training Module with MLflow & DagHub Integration
Phase 2: MLflow Experiment Tracking and Model Management
"""
import os
import sys
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore')


def setup_mlflow() -> str:
    """
    Setup MLflow tracking with DagHub.
    Falls back to local tracking if remote connection fails.
    
    Returns:
        str: Experiment ID
    """
    # Load environment variables
    load_dotenv()
    
    # Get DagHub/MLflow credentials
    mlflow_tracking_uri = os.getenv(
        'MLFLOW_TRACKING_URI',
        'https://dagshub.com/MirzaMukarram0/CoinGecko0-Cryptocurrency-Prediction-MLOps.mlflow/'
    )
    mlflow_username = os.getenv('MLFLOW_TRACKING_USERNAME', 'MirzaMukarram0')
    mlflow_password = os.getenv('MLFLOW_TRACKING_PASSWORD')
    
    # Set or create experiment
    experiment_name = "crypto-price-prediction"
    
    # Try remote MLflow first, fall back to local if it fails
    use_local = False
    
    if mlflow_password and 'dagshub.com' in mlflow_tracking_uri:
        # Try to connect to DagHub
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_password
        
        try:
            # Test connection by getting experiments
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                print(f"Created new experiment on DagHub: {experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                print(f"Using existing experiment on DagHub: {experiment_name} (ID: {experiment_id})")
        except Exception as e:
            print(f"Warning: Could not connect to DagHub MLflow: {e}")
            print("Falling back to local MLflow tracking...")
            use_local = True
    else:
        print("No DagHub credentials found, using local MLflow tracking")
        use_local = True
    
    if use_local:
        # Use local file-based MLflow tracking
        local_tracking_uri = '/opt/airflow/mlruns'
        mlflow.set_tracking_uri(f'file://{local_tracking_uri}')
        
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                print(f"Created new local experiment: {experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                print(f"Using existing local experiment: {experiment_name} (ID: {experiment_id})")
        except Exception as e:
            print(f"Warning: Could not setup local experiment: {e}")
            experiment_id = "0"  # Default experiment
    
    mlflow.set_experiment(experiment_name)
    
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"MLflow experiment: {experiment_name}")
    
    return experiment_id


class CryptoPricePredictor:
    """
    Cryptocurrency Price Predictor with MLflow tracking.
    Trains multiple models and tracks experiments on DagHub.
    """
    
    def __init__(self, models_dir: str = '/opt/airflow/data/models'):
        """Initialize predictor with models directory."""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.target_column = 'price'  # Target column from processed data
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'model_class': RandomForestRegressor,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'gradient_boosting': {
                'model_class': GradientBoostingRegressor,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                }
            },
            'ridge': {
                'model_class': Ridge,
                'params': {
                    'alpha': 1.0,
                    'random_state': 42
                }
            }
        }
        
        self.trained_models: Dict[str, Any] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for training.
        
        Args:
            df: Input DataFrame with processed data
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        print(f"Preparing features from {len(df)} records...")
        
        # Define feature columns (exclude non-feature columns)
        exclude_cols = ['id', 'symbol', 'name', 'last_updated', 'timestamp', 
                        'extraction_timestamp', 'current_price', 'roi']
        
        self.feature_columns = [col for col in df.columns 
                                if col not in exclude_cols 
                                and df[col].dtype in ['int64', 'float64']]
        
        print(f"Feature columns: {self.feature_columns}")
        
        # Handle missing values
        X = df[self.feature_columns].copy()
        X = X.fillna(X.median())
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Target variable
        y = df[self.target_column].copy()
        
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")
        
        return X, y
    
    def train_single_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        run_name: str
    ) -> Dict[str, Any]:
        """
        Train a single model with MLflow tracking.
        
        Args:
            model_name: Name of the model to train
            X_train, X_test: Training and test features
            y_train, y_test: Training and test targets
            run_name: Name for the MLflow run
            
        Returns:
            Dictionary with model and metrics
        """
        config = self.model_configs[model_name]
        
        with mlflow.start_run(run_name=f"{run_name}_{model_name}", nested=True):
            start_time = datetime.now()
            
            # Log parameters
            mlflow.log_params(config['params'])
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("n_features", len(self.feature_columns))
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_test_samples", len(X_test))
            mlflow.log_param("feature_columns", str(self.feature_columns[:10]) + "...")  # Log first 10
            
            # Create and train model
            model = config['model_class'](**config['params'])
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # MAPE (handle zero values)
            try:
                train_mape = mean_absolute_percentage_error(y_train, y_pred_train) * 100
                test_mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100
            except:
                train_mape = 0
                test_mape = 0
            
            # Cross-validation scores
            cv = TimeSeriesSplit(n_splits=3) if len(X_train) > 30 else 3
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
            cv_rmse_mean = np.sqrt(-cv_scores.mean())
            cv_rmse_std = np.sqrt(cv_scores.std())
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Log metrics to MLflow
            metrics = {
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "train_mae": train_mae,
                "test_mae": test_mae,
                "train_r2": train_r2,
                "test_r2": test_r2,
                "train_mape": train_mape,
                "test_mape": test_mape,
                "cv_rmse_mean": cv_rmse_mean,
                "cv_rmse_std": cv_rmse_std,
                "training_time_seconds": training_time
            }
            
            mlflow.log_metrics(metrics)
            
            # Log feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Save and log feature importance
                importance_path = self.models_dir / f'{model_name}_feature_importance.csv'
                importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(str(importance_path), "feature_importance")
                
                # Create feature importance plot
                plt.figure(figsize=(10, 8))
                top_features = importance_df.head(15)
                plt.barh(top_features['feature'], top_features['importance'])
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                plt.title(f'{model_name} - Top 15 Feature Importances')
                plt.tight_layout()
                
                plot_path = self.models_dir / f'{model_name}_feature_importance.png'
                plt.savefig(plot_path, dpi=100, bbox_inches='tight')
                plt.close()
                mlflow.log_artifact(str(plot_path), "plots")
            
            # Create predictions vs actual plot
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred_test, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual Price')
            plt.ylabel('Predicted Price')
            plt.title(f'{model_name} - Predicted vs Actual')
            plt.tight_layout()
            
            pred_plot_path = self.models_dir / f'{model_name}_predictions.png'
            plt.savefig(pred_plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact(str(pred_plot_path), "plots")
            
            # Log model to MLflow
            mlflow.sklearn.log_model(
                model,
                artifact_path=model_name,
                registered_model_name=f"crypto_price_{model_name}"
            )
            
            # Save model locally as well
            model_path = self.models_dir / f'{model_name}_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            print(f"\n{model_name.upper()} Results:")
            print(f"  Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
            print(f"  Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
            print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
            print(f"  CV RMSE: {cv_rmse_mean:.4f} (+/- {cv_rmse_std:.4f})")
            print(f"  Training time: {training_time:.2f}s")
            
            return {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred_test
            }
    
    def train_all_models(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train all configured models with MLflow tracking.
        
        Args:
            df: Processed DataFrame
            test_size: Fraction of data for testing
            
        Returns:
            Dictionary with training results
        """
        print("\n" + "="*60)
        print("STARTING MODEL TRAINING WITH MLFLOW TRACKING")
        print("="*60)
        
        # Setup MLflow
        experiment_id = setup_mlflow()
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Train/test split (preserve order for time series)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, shuffle=False
        )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Create parent run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"training_run_{timestamp}"
        
        results = {}
        best_model = None
        best_rmse = float('inf')
        
        # Try to start MLflow run, fall back to local if remote fails
        try:
            parent_run_context = mlflow.start_run(run_name=run_name)
        except Exception as e:
            print(f"Warning: Failed to start MLflow run on remote server: {e}")
            print("Switching to local MLflow tracking...")
            # Switch to local tracking
            mlflow.set_tracking_uri('file:///opt/airflow/mlruns')
            mlflow.set_experiment("crypto-price-prediction")
            parent_run_context = mlflow.start_run(run_name=run_name)
        
        with parent_run_context as parent_run:
            # Log dataset info
            mlflow.log_param("dataset_size", len(df))
            mlflow.log_param("n_features", len(self.feature_columns))
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("timestamp", timestamp)
            
            # Train each model
            for model_name in self.model_configs.keys():
                print(f"\n--- Training {model_name} ---")
                
                try:
                    result = self.train_single_model(
                        model_name=model_name,
                        X_train=X_train,
                        X_test=X_test,
                        y_train=y_train,
                        y_test=y_test,
                        run_name=run_name
                    )
                    
                    results[model_name] = result
                    self.trained_models[model_name] = result['model']
                    self.metrics[model_name] = result['metrics']
                    
                    # Track best model
                    if result['metrics']['test_rmse'] < best_rmse:
                        best_rmse = result['metrics']['test_rmse']
                        best_model = model_name
                        
                except Exception as e:
                    print(f"Error training {model_name}: {e}")
                    continue
            
            # Log best model info
            if best_model:
                mlflow.log_param("best_model", best_model)
                mlflow.log_metric("best_test_rmse", best_rmse)
                
                print(f"\n{'='*60}")
                print(f"BEST MODEL: {best_model} with Test RMSE: {best_rmse:.4f}")
                print(f"{'='*60}")
            
            # Save scaler
            scaler_path = self.models_dir / 'scaler.pkl'
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            mlflow.log_artifact(str(scaler_path), "preprocessing")
            
            # Save feature columns
            features_path = self.models_dir / 'feature_columns.json'
            with open(features_path, 'w') as f:
                json.dump(self.feature_columns, f)
            mlflow.log_artifact(str(features_path), "preprocessing")
            
            # Create model comparison plot
            if results:
                self._create_comparison_plot(results)
            
            # Log training summary
            summary = {
                'timestamp': timestamp,
                'experiment_id': experiment_id,
                'parent_run_id': parent_run.info.run_id,
                'best_model': best_model,
                'best_rmse': best_rmse,
                'models_trained': list(results.keys()),
                'all_metrics': self.metrics
            }
            
            summary_path = self.models_dir / 'training_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            mlflow.log_artifact(str(summary_path))
            
            print(f"\nMLflow run ID: {parent_run.info.run_id}")
            print(f"Models saved to: {self.models_dir}")
        
        return {
            'best_model': best_model,
            'best_rmse': best_rmse,
            'results': results,
            'summary': summary
        }
    
    def _create_comparison_plot(self, results: Dict[str, Any]):
        """Create model comparison visualization."""
        model_names = list(results.keys())
        
        # Metrics to compare
        metrics_to_plot = ['test_rmse', 'test_mae', 'test_r2']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, metric in enumerate(metrics_to_plot):
            values = [results[m]['metrics'][metric] for m in model_names]
            axes[idx].bar(model_names, values, color=['#2ecc71', '#3498db', '#9b59b6'])
            axes[idx].set_title(metric.replace('_', ' ').upper())
            axes[idx].set_ylabel('Value')
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for i, v in enumerate(values):
                axes[idx].text(i, v, f'{v:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        comparison_path = self.models_dir / 'model_comparison.png'
        plt.savefig(comparison_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        mlflow.log_artifact(str(comparison_path), "plots")
        print(f"Model comparison plot saved: {comparison_path}")


def train_with_mlflow(data_path: str = None, **kwargs) -> Dict[str, Any]:
    """
    Main training function to be called from Airflow DAG.
    
    Args:
        data_path: Path to processed data CSV file
        **kwargs: Additional arguments from Airflow
        
    Returns:
        Dictionary with training results
    """
    print("\n" + "="*70)
    print("CRYPTOCURRENCY PRICE PREDICTION - MODEL TRAINING")
    print("Phase 2: MLflow & DagHub Integration")
    print("="*70)
    
    # Find latest processed data if not specified
    if data_path is None:
        processed_dir = Path('/opt/airflow/data/processed')
        processed_files = sorted(processed_dir.glob('processed_*.csv'))
        
        if not processed_files:
            raise ValueError("No processed data files found!")
        
        data_path = str(processed_files[-1])
        print(f"Using latest processed data: {data_path}")
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    # Check if we have enough data
    if len(df) < 10:
        print(f"Warning: Only {len(df)} records available. Need more data for robust training.")
        if len(df) < 3:
            raise ValueError("Not enough data for training (minimum 3 records required)")
    
    # Initialize predictor and train
    predictor = CryptoPricePredictor()
    results = predictor.train_all_models(df)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best Model: {results['best_model']}")
    print(f"Best Test RMSE: {results['best_rmse']:.4f}")
    print(f"View experiments at: https://dagshub.com/MirzaMukarram0/CoinGecko0-Cryptocurrency-Prediction-MLOps.mlflow/")
    
    return results


if __name__ == "__main__":
    # For local testing
    import argparse
    
    parser = argparse.ArgumentParser(description='Train cryptocurrency price prediction models')
    parser.add_argument('--data-path', type=str, help='Path to processed data CSV')
    args = parser.parse_args()
    
    results = train_with_mlflow(data_path=args.data_path)
    print(f"\nTraining completed. Best model: {results['best_model']}")
