"""
Prometheus Metrics Module
Exposes service and model health metrics for monitoring
"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.core import CollectorRegistry
import time
import logging
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Create a custom registry for our metrics
registry = CollectorRegistry()

# Service Metrics
# ===============

# Total number of API requests
api_request_count = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

# API request latency (in seconds)
api_request_latency = Histogram(
    'api_request_latency_seconds',
    'API request latency in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=registry
)

# Current number of active requests
api_active_requests = Gauge(
    'api_active_requests',
    'Number of currently active API requests',
    registry=registry
)

# Model/Data Drift Metrics
# =========================

# Data drift ratio (out-of-distribution feature values)
data_drift_ratio = Gauge(
    'data_drift_ratio',
    'Ratio of prediction requests with out-of-distribution feature values (0.0-1.0)',
    ['model_type'],
    registry=registry
)

# Number of drift detections
data_drift_detections = Counter(
    'data_drift_detections_total',
    'Total number of data drift detections',
    ['model_type', 'feature'],
    registry=registry
)

# Model prediction count
model_predictions_total = Counter(
    'model_predictions_total',
    'Total number of predictions made',
    ['model_type', 'coin_id'],
    registry=registry
)

# Model prediction latency
model_prediction_latency = Histogram(
    'model_prediction_latency_seconds',
    'Model prediction latency in seconds',
    ['model_type'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=registry
)

# Model health status
model_health_status = Gauge(
    'model_health_status',
    'Model health status (1=healthy, 0=unhealthy)',
    ['model_type'],
    registry=registry
)


class DataDriftDetector:
    """
    Detects data drift by comparing incoming features to training distribution
    """
    
    def __init__(self):
        """Initialize the drift detector"""
        self.feature_bounds: Dict[str, Dict[str, float]] = {}
        self.training_stats: Dict[str, Dict[str, float]] = {}
        
    def set_training_stats(self, feature_stats: Dict[str, Dict[str, float]]):
        """
        Set training data statistics for drift detection
        
        Args:
            feature_stats: Dictionary with feature names as keys and stats as values
                          Each stat dict should contain: 'mean', 'std', 'min', 'max'
        """
        self.training_stats = feature_stats
        
        # Calculate bounds (mean ± 3*std for normal distribution)
        for feature, stats in feature_stats.items():
            self.feature_bounds[feature] = {
                'min': stats.get('min', stats['mean'] - 3 * stats['std']),
                'max': stats.get('max', stats['mean'] + 3 * stats['std']),
                'mean': stats['mean'],
                'std': stats['std']
            }
        
        logger.info(f"Set training stats for {len(self.feature_bounds)} features")
    
    def detect_drift(self, features: Dict[str, float], model_type: str = "unknown") -> Dict[str, bool]:
        """
        Detect if features are out-of-distribution
        
        Args:
            features: Dictionary of feature name -> value
            model_type: Type of model being used
            
        Returns:
            Dictionary of feature name -> is_drift (boolean)
        """
        drift_results = {}
        
        if not self.feature_bounds:
            # No training stats available, assume no drift
            logger.warning("No training stats available for drift detection")
            return {feature: False for feature in features.keys()}
        
        for feature, value in features.items():
            if feature not in self.feature_bounds:
                # Unknown feature, assume no drift
                drift_results[feature] = False
                continue
            
            bounds = self.feature_bounds[feature]
            
            # Check if value is outside 3-sigma bounds
            is_drift = value < bounds['min'] or value > bounds['max']
            drift_results[feature] = is_drift
            
            if is_drift:
                # Increment drift counter
                data_drift_detections.labels(model_type=model_type, feature=feature).inc()
                logger.debug(f"Drift detected in feature '{feature}': value={value:.4f}, "
                           f"bounds=[{bounds['min']:.4f}, {bounds['max']:.4f}]")
        
        return drift_results
    
    def calculate_drift_ratio(self, features: Dict[str, float], model_type: str = "unknown") -> float:
        """
        Calculate the ratio of features that are out-of-distribution
        
        Args:
            features: Dictionary of feature name -> value
            model_type: Type of model being used
            
        Returns:
            Ratio of drifted features (0.0 to 1.0)
        """
        drift_results = self.detect_drift(features, model_type)
        
        if not drift_results:
            return 0.0
        
        drifted_count = sum(1 for is_drift in drift_results.values() if is_drift)
        drift_ratio = drifted_count / len(drift_results)
        
        return drift_ratio


# Global drift detector instance
drift_detector = DataDriftDetector()


def get_metrics():
    """
    Get Prometheus metrics in text format
    
    Returns:
        Metrics in Prometheus text format
    """
    return generate_latest(registry)


def record_api_request(method: str, endpoint: str, status: str, duration: float):
    """
    Record an API request metric
    
    Args:
        method: HTTP method (GET, POST, etc.)
        endpoint: API endpoint path
        status: Response status (success, error, etc.)
        duration: Request duration in seconds
    """
    api_request_count.labels(method=method, endpoint=endpoint, status=status).inc()
    api_request_latency.labels(method=method, endpoint=endpoint).observe(duration)


def record_prediction(model_type: str, coin_id: str, duration: float, drift_ratio: float = 0.0):
    """
    Record a model prediction metric
    
    Args:
        model_type: Type of model used
        coin_id: Cryptocurrency ID
        duration: Prediction duration in seconds
        drift_ratio: Data drift ratio (0.0-1.0)
    """
    model_predictions_total.labels(model_type=model_type, coin_id=coin_id).inc()
    model_prediction_latency.labels(model_type=model_type).observe(duration)
    data_drift_ratio.labels(model_type=model_type).set(drift_ratio)


def set_model_health(model_type: str, is_healthy: bool):
    """
    Set model health status
    
    Args:
        model_type: Type of model
        is_healthy: Whether the model is healthy
    """
    model_health_status.labels(model_type=model_type).set(1.0 if is_healthy else 0.0)

