"""
FastAPI Application for Cryptocurrency Price Prediction
Provides REST API endpoints for real-time price predictions
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging
import os
import json
from datetime import datetime, timedelta
import asyncio

# Import our modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.extract import extract_coingecko_data
from src.data.transform import transform_data
from src.models.predict import CryptoPricePredictor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Cryptocurrency Price Prediction API",
    description="Real-time cryptocurrency price prediction using machine learning",
    version="1.0.0"
)


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    coin_id: str = Field(default="bitcoin", description="Cryptocurrency ID (e.g., 'bitcoin')")
    hours_ahead: int = Field(default=1, ge=1, le=24, description="Hours to predict ahead (1-24)")
    model_type: str = Field(default="random_forest", description="Model type to use")


class PredictionResponse(BaseModel):
    coin_id: str
    timestamp: str
    predicted_price: float
    confidence_interval: Optional[Dict[str, float]] = None
    model_type: str
    prediction_horizon_hours: int


class BatchPredictionRequest(BaseModel):
    coin_id: str = Field(default="bitcoin")
    start_date: str = Field(description="Start date (YYYY-MM-DD)")
    end_date: str = Field(description="End date (YYYY-MM-DD)")
    model_type: str = Field(default="random_forest")


class ModelStatus(BaseModel):
    model_type: str
    status: str
    last_updated: str
    accuracy_metrics: Optional[Dict[str, float]] = None


# Global variables for models
loaded_models = {}
model_metadata = {}


# Utility functions
def load_model_if_needed(model_type: str = "random_forest"):
    """Load model if not already loaded"""
    if model_type not in loaded_models:
        try:
            # Find the latest model files
            models_dir = "models"
            if not os.path.exists(models_dir):
                raise FileNotFoundError(f"Models directory not found: {models_dir}")
            
            # Look for model files
            model_files = [f for f in os.listdir(models_dir) if f.startswith(f"{model_type}_model_") and f.endswith('.joblib')]
            
            if not model_files:
                raise FileNotFoundError(f"No model files found for {model_type}")
            
            # Get the latest model
            latest_model = sorted(model_files)[-1]
            timestamp = latest_model.split('_')[-1].replace('.joblib', '')
            
            model_path = os.path.join(models_dir, latest_model)
            scaler_path = os.path.join(models_dir, f"{model_type}_scaler_{timestamp}.joblib")
            features_path = os.path.join(models_dir, f"{model_type}_features_{timestamp}.joblib")
            
            # Load model
            predictor = CryptoPricePredictor()
            predictor.load_model(model_path, scaler_path, features_path)
            
            loaded_models[model_type] = predictor
            model_metadata[model_type] = {
                "last_updated": timestamp,
                "model_path": model_path,
                "status": "loaded"
            }
            
            logger.info(f"✓ Model {model_type} loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model {model_type}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


async def get_latest_data(coin_id: str = "bitcoin", hours: int = 48):
    """Get latest cryptocurrency data"""
    try:
        # Extract fresh data
        raw_file = extract_coingecko_data(
            coin_id=coin_id,
            vs_currency="usd",
            days=max(2, hours // 24 + 1),  # Get enough data
            interval="hourly"
        )
        
        # Transform data
        processed_file = transform_data(raw_file)
        
        # Load processed data
        df = pd.read_csv(processed_file, index_col=0, parse_dates=True)
        
        return df
        
    except Exception as e:
        logger.error(f"Error getting latest data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get latest data: {str(e)}")


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Cryptocurrency Price Prediction API",
        "version": "1.0.0",
        "endpoints": [
            "/predict",
            "/predict/batch", 
            "/models/status",
            "/health"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "loaded_models": list(loaded_models.keys())
    }


@app.get("/models/status")
async def get_models_status() -> List[ModelStatus]:
    """Get status of all available models"""
    models_dir = "models"
    available_models = []
    
    if os.path.exists(models_dir):
        # Find all model types
        model_files = [f for f in os.listdir(models_dir) if f.endswith('_model_*.joblib')]
        model_types = list(set([f.split('_')[0] for f in model_files]))
        
        for model_type in model_types:
            status_info = ModelStatus(
                model_type=model_type,
                status="available" if model_type in loaded_models else "not_loaded",
                last_updated=model_metadata.get(model_type, {}).get("last_updated", "unknown")
            )
            available_models.append(status_info)
    
    return available_models


@app.post("/predict")
async def predict_price(request: PredictionRequest) -> PredictionResponse:
    """
    Predict cryptocurrency price for the next N hours
    """
    try:
        # Load model if needed
        load_model_if_needed(request.model_type)
        
        # Get latest data
        df = await get_latest_data(request.coin_id, hours=48)
        
        # Get predictor
        predictor = loaded_models[request.model_type]
        
        # Make prediction
        if request.hours_ahead == 1:
            # Single step prediction
            latest_timestamp = df.index[-1]
            prediction_result = predictor.predict_single(df, latest_timestamp)
            predicted_price = prediction_result['predicted_price']
        else:
            # Multi-step prediction
            future_predictions = predictor.predict_future(df, request.hours_ahead)
            predicted_price = future_predictions[-1]['predicted_price']
        
        # Create response
        response = PredictionResponse(
            coin_id=request.coin_id,
            timestamp=(datetime.now() + timedelta(hours=request.hours_ahead)).isoformat(),
            predicted_price=predicted_price,
            model_type=request.model_type,
            prediction_horizon_hours=request.hours_ahead
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest) -> List[PredictionResponse]:
    """
    Generate batch predictions for a date range
    """
    try:
        # Load model if needed
        load_model_if_needed(request.model_type)
        
        # Parse dates
        start_date = pd.to_datetime(request.start_date)
        end_date = pd.to_datetime(request.end_date)
        
        # Validate date range
        if start_date >= end_date:
            raise HTTPException(status_code=400, detail="Start date must be before end date")
        
        # Get data for the period (with some buffer for features)
        buffer_days = 3
        extended_start = start_date - timedelta(days=buffer_days)
        
        # For historical predictions, we would load historical data
        # For now, we'll use the latest data as an example
        df = await get_latest_data(request.coin_id, hours=168)  # 7 days
        
        # Filter data to requested range
        mask = (df.index >= start_date) & (df.index <= end_date)
        filtered_df = df[mask]
        
        if len(filtered_df) == 0:
            raise HTTPException(status_code=404, detail="No data available for the requested date range")
        
        # Get predictor
        predictor = loaded_models[request.model_type]
        
        # Make batch predictions
        predictions_df = predictor.predict_batch(filtered_df, start_date, end_date)
        
        # Convert to response format
        responses = []
        for timestamp, row in predictions_df.iterrows():
            response = PredictionResponse(
                coin_id=request.coin_id,
                timestamp=timestamp.isoformat(),
                predicted_price=float(row['predicted_price']),
                model_type=request.model_type,
                prediction_horizon_hours=1  # Batch predictions are 1-hour ahead
            )
            responses.append(response)
        
        return responses
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/models/{model_type}/load")
async def load_model(model_type: str):
    """
    Manually load a specific model
    """
    try:
        load_model_if_needed(model_type)
        return {
            "message": f"Model {model_type} loaded successfully",
            "status": "loaded",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.delete("/models/{model_type}/unload")
async def unload_model(model_type: str):
    """
    Unload a specific model from memory
    """
    if model_type in loaded_models:
        del loaded_models[model_type]
        if model_type in model_metadata:
            del model_metadata[model_type]
        
        return {
            "message": f"Model {model_type} unloaded successfully",
            "status": "unloaded",
            "timestamp": datetime.now().isoformat()
        }
    else:
        raise HTTPException(status_code=404, detail=f"Model {model_type} not found in memory")


# Background task for periodic model updates
@app.on_event("startup")
async def startup_event():
    """Initialize API on startup"""
    logger.info("Starting Cryptocurrency Prediction API...")
    
    # Try to load default model
    try:
        load_model_if_needed("random_forest")
        logger.info("✓ Default model loaded")
    except Exception as e:
        logger.warning(f"Could not load default model: {e}")
    
    logger.info("API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Cryptocurrency Prediction API...")
    
    # Clear loaded models
    loaded_models.clear()
    model_metadata.clear()
    
    logger.info("API shutdown complete")


if __name__ == "__main__":
    import uvicorn
    
    # Run the API
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )