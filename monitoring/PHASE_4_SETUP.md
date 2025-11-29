# Phase IV: Monitoring and Observability - Quick Setup Guide

## 🎯 Overview

This guide will help you set up **Prometheus** and **Grafana** for monitoring your Cryptocurrency Prediction API with:
- ✅ Service metrics (API latency, request count)
- ✅ Model metrics (prediction latency, prediction count)
- ✅ Data drift detection (out-of-distribution feature ratio)
- ✅ Real-time dashboards
- ✅ Alerting (latency > 500ms, drift spikes)

## 📋 Prerequisites

- Docker and Docker Compose installed
- All Phase I services running (optional, but recommended)
- At least 4GB RAM available

## 🚀 Quick Start (5 Minutes)

### Step 1: Start All Services

```bash
# Navigate to project root
cd CoinGecko0-Cryptocurrency-Prediction-MLOps

# Start all services (including Prometheus and Grafana)
docker-compose up -d

# Wait for services to be healthy (about 30 seconds)
docker-compose ps
```

### Step 2: Verify Services

Check that all services are running:

```bash
# Check service status
docker-compose ps

# You should see:
# - api (crypto-prediction-api) - Running
# - prometheus - Running  
# - grafana - Running
# - airflow services - Running
```

### Step 3: Access Dashboards

| Service | URL | Default Credentials |
|---------|-----|---------------------|
| **Grafana** | http://localhost:3000 | admin / admin |
| **Prometheus** | http://localhost:9090 | None |
| **FastAPI** | http://localhost:8000 | None |
| **API Metrics** | http://localhost:8000/metrics | None |

### Step 4: View Grafana Dashboard

1. Open http://localhost:3000
2. Login with `admin` / `admin`
3. Go to **Dashboards** → **Browse**
4. Find **"Cryptocurrency Prediction API - Monitoring Dashboard"**
5. Click to open

The dashboard will show:
- 📊 API Request Rate
- ⏱️ API Latency (P50, P95, P99)
- 📈 Total Requests
- 🔄 Active Requests
- 🤖 Model Predictions
- 🎯 Data Drift Ratio
- ✅ Model Health Status

## 🔧 Configuration

### Initialize Data Drift Detection

Data drift detection requires training statistics. Initialize it:

```bash
# Option 1: Using the helper script
python src/utils/init_drift_detector.py \
    data/processed/your_training_data.csv \
    --features models/your_model_features.joblib \
    --output monitoring/training_stats.json

# Option 2: Set environment variable for API to auto-load
export TRAINING_STATS_FILE=monitoring/training_stats.json
```

The API will automatically load training stats on startup if the file exists.

### Configure Alerts

#### Option 1: Grafana UI Alerts (Recommended)

1. Open Grafana → **Alerting** → **Alert rules**
2. Find the pre-configured alerts:
   - **HighAPILatency**: Latency > 500ms
   - **DataDriftSpike**: Drift ratio > 30%
3. Edit alerts to add notification channels (Slack, email, etc.)

#### Option 2: Alertmanager (Advanced)

1. Add Alertmanager to `docker-compose.yml`
2. Configure notification channels in Alertmanager
3. Prometheus will send alerts to Alertmanager

### Customize Metrics

All metrics are exposed at: `http://localhost:8000/metrics`

Key metrics:
- `api_requests_total` - Total API requests
- `api_request_latency_seconds` - Request latency histogram
- `model_predictions_total` - Total predictions
- `model_prediction_latency_seconds` - Prediction latency
- `data_drift_ratio` - Data drift ratio (0.0-1.0)
- `model_health_status` - Model health (1=healthy, 0=unhealthy)

## 🧪 Testing

### Test API and Metrics

```bash
# 1. Make a prediction request
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "coin_id": "bitcoin",
    "hours_ahead": 1,
    "model_type": "random_forest"
  }'

# 2. Check metrics endpoint
curl http://localhost:8000/metrics | grep api_requests_total

# 3. View in Prometheus
# Go to http://localhost:9090
# Try query: rate(api_requests_total[5m])
```

### Test Data Drift

```bash
# Make multiple requests with different data patterns
# Watch the data_drift_ratio metric in Grafana
# It should update in real-time
```

## 📊 Dashboard Panels Explained

1. **API Request Rate**: Requests per second by endpoint and status
2. **API Latency**: P50, P95, P99 latency percentiles
3. **Total Requests**: Cumulative request count by status
4. **Active Requests**: Current number of in-flight requests
5. **Model Prediction Rate**: Predictions per second by model type
6. **Model Prediction Latency**: Time taken for model inference
7. **Data Drift Ratio**: Percentage of features out-of-distribution
8. **Data Drift Detections**: Count of drift detections by feature
9. **Model Health Status**: Binary health indicator (1=healthy, 0=unhealthy)
10. **API Endpoint Summary**: Table of request rates by endpoint

## 🚨 Alerting

### Pre-configured Alerts

1. **High Latency Alert**
   - Condition: P95 latency > 500ms for 5 minutes
   - Severity: Warning
   - Action: Log to Grafana alerts

2. **Data Drift Spike Alert**
   - Condition: Drift ratio > 30% for 2 minutes
   - Severity: Warning
   - Action: Log to Grafana alerts

### Adding Notification Channels

1. Go to Grafana → **Alerting** → **Notification channels**
2. Click **New channel**
3. Choose channel type (Slack, Email, PagerDuty, etc.)
4. Configure settings
5. Edit alert rules to use the channel

## 🔍 Troubleshooting

### Metrics Not Appearing

```bash
# Check API is running
docker-compose ps api

# Check API logs
docker-compose logs api

# Test metrics endpoint
curl http://localhost:8000/metrics
```

### Prometheus Not Scraping

```bash
# Check Prometheus targets
# Go to http://localhost:9090/targets
# Should show "crypto-prediction-api" as UP

# Check Prometheus logs
docker-compose logs prometheus

# Verify network connectivity
docker-compose exec prometheus ping api
```

### Grafana Dashboard Not Loading

```bash
# Check Grafana logs
docker-compose logs grafana

# Verify Prometheus datasource
# Go to Grafana → Configuration → Data sources
# Should see "Prometheus" datasource configured

# Check dashboard JSON syntax
cat monitoring/grafana/dashboards/crypto-prediction-monitoring.json | python -m json.tool
```

### Data Drift Not Working

```bash
# Check if training stats are loaded
docker-compose logs api | grep "Training stats"

# Initialize training stats (see Configuration section)
# Verify file exists
ls -la monitoring/training_stats.json
```

## 📚 Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Prometheus Client Python](https://github.com/prometheus/client_python)

## ✅ Verification Checklist

- [ ] All services running (`docker-compose ps`)
- [ ] Grafana accessible at http://localhost:3000
- [ ] Prometheus accessible at http://localhost:9090
- [ ] API metrics endpoint working (`curl http://localhost:8000/metrics`)
- [ ] Grafana dashboard loads and shows data
- [ ] Training stats initialized (for drift detection)
- [ ] Alerts configured (optional)
- [ ] Test prediction request generates metrics

## 🎉 Success!

If all checkboxes are checked, your monitoring system is fully operational!

You can now:
- Monitor API performance in real-time
- Track model prediction metrics
- Detect data drift automatically
- Receive alerts for anomalies

---

**Need Help?** Check the main README.md or open an issue on GitHub.

