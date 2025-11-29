# CoinGecko Cryptocurrency Prediction MLOps Pipeline

A comprehensive MLOps pipeline for cryptocurrency price prediction using Apache Airflow, DVC, MLflow, and Docker.

## 🏗️ Project Structure

```
├── README.md                           # Project documentation
├── phase_1.md                         # Phase 1 implementation details
├── .gitignore                         # Git ignore rules
├── requirements.txt                   # Python dependencies
├── config/                            # Configuration files
│   ├── airflow.cfg                   # Airflow configuration
│   ├── mlflow_config.yaml            # MLflow settings
│   └── pipeline_config.yaml          # Pipeline parameters
├── src/                               # Source code
│   ├── __init__.py                   
│   ├── data/                         # Data processing modules
│   │   ├── __init__.py
│   │   ├── extract.py                # CoinGecko API data extraction
│   │   ├── quality_checks.py         # Data quality validation
│   │   └── transform.py              # Feature engineering & transformation
│   └── utils/                        # Utility modules
│       ├── __init__.py
│       ├── profiling.py              # Data profiling & analysis
│       ├── storage.py                # Cloud storage utilities
│       └── dvc_version.py            # DVC data versioning
├── airflow/                          # Airflow orchestration
│   └── dags/
│       └── currency_prediction_dag.py # Main pipeline DAG
├── docker/                           # Docker configuration
│   ├── docker-compose.yml           # Multi-service setup
│   ├── Dockerfile.airflow           # Custom Airflow image
│   └── Dockerfile.mlflow            # Custom MLflow image
├── data/                             # Data directories (gitignored)
│   ├── raw/                         # Raw extracted data
│   ├── processed/                   # Transformed datasets
│   ├── cloud_storage/              # Local cloud storage
│   └── dvc_storage/                 # DVC remote storage
└── reports/                          # Generated reports (gitignored)
    └── profiles/                     # Data profiling reports
```

## 🚀 Features

### Phase I: Data Engineering Foundation (COMPLETE)

✅ **Data Extraction**
- CoinGecko API integration with retry logic
- Hourly cryptocurrency market data collection
- Automatic error handling and logging

✅ **Data Quality Gates**
- Mandatory quality checks (pipeline fails on violations)
- Schema validation and null value detection
- Data type and range validation

✅ **Feature Engineering**
- Temporal features (hour, day, month)
- Lag features (1h, 3h, 6h, 12h, 24h)
- Rolling statistics (mean, std, min, max)
- Price change and momentum indicators
- Volume analysis features

✅ **Data Profiling**
- Comprehensive pandas profiling reports
- MLflow integration for report tracking
- Statistical analysis and visualization

✅ **Cloud Storage Integration**
- MinIO, AWS S3, Azure Blob support
- Automatic timestamping and verification
- Multi-provider abstraction layer

✅ **Data Versioning**
- DVC (Data Version Control) integration
- Git integration for metadata tracking
- Reproducible data lineage

✅ **Orchestration**
- Apache Airflow DAG with 7 tasks
- Dependency management and scheduling
- Error handling and retry mechanisms

✅ **Containerization**
- Docker Compose multi-service setup
- Airflow, PostgreSQL, MinIO, MLflow services
- Development environment automation

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Orchestration** | Apache Airflow 2.5.1 | Workflow management and scheduling |
| **Data Processing** | Python, Pandas | ETL pipeline implementation |
| **Data API** | CoinGecko Public API | Cryptocurrency market data source |
| **Version Control** | DVC + Git | Dataset and code versioning |
| **Experiment Tracking** | MLflow + DagHub | Model and data tracking |
| **Storage** | MinIO, S3, Azure Blob | Object storage for datasets |
| **Containerization** | Docker, Docker Compose | Service orchestration |
| **Database** | PostgreSQL | Airflow metadata and workflow state |
| **Monitoring** | Prometheus, Grafana | Metrics collection, visualization, and alerting |
| **API Framework** | FastAPI | REST API for predictions |

## 📋 Prerequisites

- Docker and Docker Compose
- Python 3.9+
- Git with DVC
- 8GB+ RAM recommended

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/your-username/CoinGecko0-Cryptocurrency-Prediction-MLOps.git
cd CoinGecko0-Cryptocurrency-Prediction-MLOps
```

### 2. Start Services
```bash
# Navigate to docker directory
cd docker

# Start all services
docker-compose up -d

# Check service status
docker-compose ps
```

### 3. Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| **Airflow** | http://localhost:8080 | admin / admin |
| **MLflow** | http://localhost:5000 | - |
| **MinIO** | http://localhost:9001 | minioadmin / minioadmin |

### 4. Run Pipeline

1. Open Airflow at http://localhost:8080
2. Find `cryptocurrency_mlops_pipeline` DAG
3. Enable the DAG (toggle switch)
4. Trigger run manually or wait for schedule

## 📊 Pipeline Flow

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Extract   │───▶│ Quality Gate │───▶│ Transform   │
│  CoinGecko  │    │   (FAIL FAST)│    │  Features   │
└─────────────┘    └──────────────┘    └─────────────┘
                                              │
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Cleanup   │◀───│ DVC Version  │◀───│   Profile   │
│             │    │              │    │   & Log     │
└─────────────┘    └──────────────┘    └─────────────┘
                                              │
                         ┌──────────────┐    │
                         │ Upload to    │◀───┘
                         │ Cloud Storage│
                         └──────────────┘
```

## ⚙️ Configuration

### Pipeline Parameters
Edit `config/pipeline_config.yaml`:
```yaml
extraction:
  coin_id: "bitcoin"
  vs_currency: "usd"
  days: 1
  interval: "hourly"

quality:
  max_null_percentage: 1.0
  required_columns: ["price", "market_cap", "total_volume"]

storage:
  type: "minio"  # local, minio, s3, azure
  bucket: "mlops-data"
```

### Airflow Schedule
Update `SCHEDULE_INTERVAL` in the DAG file:
- `'0 */6 * * *'` - Every 6 hours (default)
- `'@hourly'` - Every hour
- `'@daily'` - Daily at midnight

## 📈 Monitoring & Observability

### Airflow Monitoring
- **DAG View**: Pipeline structure and dependencies
- **Task Instance**: Individual task logs and status
- **Gantt Chart**: Pipeline execution timeline
- **Graph View**: Real-time task progress

### MLflow Tracking
- **Experiments**: Data profiling experiments
- **Artifacts**: Profiling reports and metadata
- **Metrics**: Data quality scores over time

### Quality Gates
The pipeline implements **fail-fast** quality gates:
- **Null Values**: > 1% threshold fails the pipeline
- **Schema Changes**: Missing columns stop execution
- **Empty Datasets**: Zero rows trigger failure
- **Data Types**: Type mismatches cause failure

## 🔧 Development

### Adding New Features
1. Create new modules in `src/`
2. Update DAG to include new tasks
3. Add tests for new functionality
4. Update configuration as needed

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run individual modules
cd src
python -m data.extract
python -m data.quality_checks data/raw/sample.csv
python -m data.transform data/raw/sample.csv
```

### Testing
```bash
# Run data quality checks
python src/data/quality_checks.py data/test/sample.csv

# Test storage upload
python src/utils/storage.py data/test/sample.csv

# Test DVC versioning
python src/utils/dvc_version.py data/test/sample.csv
```

## 🔮 Future Phases

### Phase II: Model Experimentation
- Model training and evaluation framework
- Hyperparameter optimization
- Model comparison and selection
- Feature importance analysis

### Phase III: CI/CD Pipeline
- Automated testing and validation
- Model deployment automation
- Infrastructure as Code (IaC)
- Environment promotion workflow

### Phase IV: Monitoring and Observability (COMPLETE ✅)

✅ **Prometheus Metrics Collection**
- Embedded Prometheus client in FastAPI server
- Service metrics: API inference latency, total request count
- Model metrics: Prediction count, prediction latency
- Data drift metrics: Out-of-distribution feature ratio
- Real-time metrics exposure at `/metrics` endpoint

✅ **Grafana Dashboards**
- Live dashboard for service and model health metrics
- Real-time visualization of API performance
- Data drift monitoring and visualization
- Model health status tracking

✅ **Alerting System**
- Grafana alerts for high latency (>500ms threshold)
- Data drift spike detection alerts (>30% threshold)
- Model health status alerts
- Configurable alert notifications

## 📊 Phase IV: Monitoring Setup Guide

> **🚀 Quick Start:** See [QUICK_START.md](QUICK_START.md) for a 5-minute setup guide
> 
> **📖 Complete Guide:** See [COMPLETE_SETUP_GUIDE.md](COMPLETE_SETUP_GUIDE.md) for detailed step-by-step instructions

### Quick Setup (5 Minutes)

### Prerequisites
- Docker and Docker Compose installed
- All services from Phase I running

### Step 1: Start Monitoring Services

```bash
# Start all services including Prometheus and Grafana
docker-compose up -d

# Verify all services are running
docker-compose ps
```

### Step 2: Access Monitoring Dashboards

| Service | URL | Credentials |
|---------|-----|-------------|
| **Grafana** | http://localhost:3000 | admin / admin |
| **Prometheus** | http://localhost:9090 | - |
| **FastAPI** | http://localhost:8000 | - |
| **API Metrics** | http://localhost:8000/metrics | - |

### Step 3: Initialize Data Drift Detection

Before drift detection works, you need to initialize it with training data statistics:

```bash
# Initialize drift detector from training data
python src/utils/init_drift_detector.py data/processed/your_training_data.csv \
    --features models/your_model_features.joblib \
    --output monitoring/training_stats.json
```

Or programmatically in your code:

```python
from src.utils.metrics import drift_detector
import pandas as pd

# Load your training data
df = pd.read_csv('data/processed/training_data.csv')

# Calculate and set training statistics
from src.utils.init_drift_detector import calculate_training_stats
stats = calculate_training_stats(df, feature_columns)
drift_detector.set_training_stats(stats)
```

### Step 4: View Grafana Dashboard

1. Open http://localhost:3000
2. Login with admin/admin
3. Navigate to **Dashboards** → **Cryptocurrency Prediction API - Monitoring Dashboard**
4. The dashboard will automatically load with:
   - API Request Rate
   - API Request Latency (P50, P95, P99)
   - Total API Requests
   - Active API Requests
   - Model Prediction Rate
   - Model Prediction Latency
   - Data Drift Ratio
   - Data Drift Detections
   - Model Health Status
   - API Endpoint Summary

### Step 5: Configure Alerts

The dashboard includes pre-configured alerts:

1. **High Latency Alert**: Fires when P95 latency > 500ms for 5 minutes
2. **Data Drift Spike Alert**: Fires when drift ratio > 30% for 2 minutes

To configure alert notifications (Slack, email, etc.):

1. Go to **Alerting** → **Notification channels** in Grafana
2. Add your notification channel
3. Edit the alert rules in the dashboard to use your channel

### Available Metrics

#### Service Metrics
- `api_requests_total`: Total API requests by method, endpoint, status
- `api_request_latency_seconds`: Request latency histogram
- `api_active_requests`: Current number of active requests

#### Model Metrics
- `model_predictions_total`: Total predictions by model type and coin
- `model_prediction_latency_seconds`: Prediction latency histogram
- `model_health_status`: Model health (1=healthy, 0=unhealthy)

#### Data Drift Metrics
- `data_drift_ratio`: Ratio of out-of-distribution features (0.0-1.0)
- `data_drift_detections_total`: Total drift detections by model and feature

### Testing the Monitoring System

1. **Generate API Traffic**:
   ```bash
   # Make prediction requests
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"coin_id": "bitcoin", "hours_ahead": 1}'
   ```

2. **Check Metrics Endpoint**:
   ```bash
   curl http://localhost:8000/metrics
   ```

3. **View in Prometheus**:
   - Go to http://localhost:9090
   - Try queries like: `rate(api_requests_total[5m])`
   - Or: `data_drift_ratio`

4. **Monitor in Grafana**:
   - Watch real-time updates in the dashboard
   - Check alert status

### Troubleshooting

**Metrics not appearing?**
- Ensure the API service is running: `docker-compose ps api`
- Check API logs: `docker-compose logs api`
- Verify metrics endpoint: `curl http://localhost:8000/metrics`

**Prometheus not scraping?**
- Check Prometheus targets: http://localhost:9090/targets
- Verify API is accessible from Prometheus container
- Check Prometheus logs: `docker-compose logs prometheus`

**Grafana dashboard not loading?**
- Check Grafana logs: `docker-compose logs grafana`
- Verify Prometheus datasource is configured
- Check dashboard JSON syntax

**Data drift not working?**
- Ensure training stats are initialized (see Step 3)
- Check that feature names match between training and prediction
- Verify drift detector is initialized in API startup

## 📚 Documentation

- **Phase 1 Details**: See [phase_1.md](phase_1.md)
- **Phase 4 Details**: See monitoring setup guide above
- **API Documentation**: Auto-generated from docstrings
- **Configuration Guide**: See `config/` directory
- **Troubleshooting**: Check Airflow logs, MLflow UI, and monitoring dashboards

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **CoinGecko** for free cryptocurrency API
- **Apache Airflow** community
- **MLflow** and **DVC** development teams
- **DagHub** for free MLflow hosting

---

**Status**: Phase I & IV Complete ✅ | **Next**: Model Experimentation Phase II
