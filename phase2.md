# Phase II: Experimentation and Model Management

## Overview

This phase focuses on establishing a robust experiment tracking and model management system using MLflow and DagHub. The goal is to create a unified platform where all experiments, models, and data versions are tracked, compared, and managed in a collaborative environment.

## Architecture
```
Airflow DAG (Orchestrator)
    ↓
train.py (Training Script)
    ↓
MLflow Tracking (Experiment Logging)
    ↓
DagHub (Central Hub)
    ├── Git (Code Versioning)
    ├── DVC (Data Versioning)
    └── MLflow (Model & Experiment Tracking)
```

## Objectives

1. **Centralized Experiment Tracking**: Every model training run is logged with complete reproducibility
2. **Model Registry**: Maintain a versioned registry of all trained models
3. **Artifact Management**: Store and version all training artifacts (models, scalers, feature lists)
4. **Collaborative Workspace**: Enable team visibility into all experiments and results
5. **Integration with Airflow**: Seamless triggering of training from orchestration pipeline

---

## Step 4.1: DagHub Setup and Configuration

### What is DagHub?

DagHub is a unified platform that combines:
- **Git hosting** (like GitHub) for code versioning
- **DVC remote storage** for data versioning
- **MLflow tracking server** for experiment tracking
- **Collaborative UI** for team visibility

### Setup Process

#### 4.1.1 Create DagHub Account
- Sign up at https://dagshub.com
- Authenticate using GitHub/GitLab account (recommended)
- Verify email address

#### 4.1.2 Create New Repository
- Click "New Repository" 
- Repository name: `mlops-crypto-prediction`
- Visibility: Public (for learning) or Private (for production)
- Initialize with README: No (we already have local repo)
- Click "Create Repository"

#### 4.1.3 Get Authentication Credentials
- Navigate to User Settings → Tokens
- Click "Generate New Token"
- Token name: `mlops-project-token`
- Permissions needed:
  - `repo` (full repository access)
  - `workflow` (for GitHub Actions integration)
- Copy and save token securely (shown only once)

#### 4.1.4 Get MLflow Tracking URI
- Go to your repository page
- Click "Remote" tab
- Find MLflow Tracking URI section
- Copy the URI format: `https://dagshub.com/YOUR_USERNAME/mlops-crypto-prediction.mlflow`

---

## Step 4.2: Local Repository Configuration

### 4.2.1 Connect Git Remote to DagHub

**Purpose**: Link your local Git repository to DagHub for code versioning

**What to Configure**:
- Add DagHub as Git remote origin
- Push existing commits to DagHub
- Verify connection

**Key Configuration Details**:
- Remote URL format: `https://dagshub.com/YOUR_USERNAME/mlops-crypto-prediction.git`
- Authentication: Use DagHub token as password
- Branch structure: master (main), dev, test

### 4.2.2 Configure DVC Remote Storage

**Purpose**: Use DagHub's storage for versioning large data files

**What to Configure**:
- Set DagHub as default DVC remote
- Configure authentication (username + token)
- Set up local credentials (not committed to Git)
- Test connection with dummy file push

**Storage Details**:
- DVC remote URL format: `https://dagshub.com/YOUR_USERNAME/mlops-crypto-prediction.dvc`
- Storage type: DagHub DVC storage (S3-compatible)
- Authentication method: Basic auth with token

### 4.2.3 Configure MLflow Tracking

**Purpose**: Send all experiment data to DagHub's MLflow server

**What to Configure**:
- Set MLflow tracking URI environment variable
- Configure DagHub authentication for MLflow
- Test connection with dummy experiment
- Set default experiment name

**Tracking Details**:
- Tracking URI: From Step 4.1.4
- Experiment name: `crypto-price-prediction`
- Authentication: Via DAGSHUB_TOKEN environment variable

---

## Step 4.3: MLflow Tracking Implementation

### 4.3.1 Understanding MLflow Components

**MLflow Tracking**: Records experiments with parameters, metrics, and artifacts

**Key Concepts**:
- **Experiment**: Collection of related runs (e.g., "crypto-price-prediction")
- **Run**: Single execution of training script with specific parameters
- **Parameters**: Hyperparameters and configuration (logged once per run)
- **Metrics**: Performance measurements (can log multiple times per run)
- **Artifacts**: Files generated during run (models, plots, data files)
- **Tags**: Metadata for organizing runs (e.g., "model_type:random_forest")

### 4.3.2 What to Track in Training Script

#### Parameters to Log:
- **Model hyperparameters**: 
  - n_estimators, max_depth, learning_rate
  - alpha, C, kernel (depending on model type)
- **Feature engineering settings**:
  - prediction_horizon (e.g., 1 hour)
  - lag_windows (e.g., [1, 2, 3, 6, 12, 24])
  - rolling_windows (e.g., [6, 12, 24, 48])
- **Data configuration**:
  - train_test_split_ratio
  - cross_validation_folds
  - date_range (start_date, end_date)
- **Training settings**:
  - random_seed
  - scaling_method
  - target_column

#### Metrics to Log:
- **Primary metrics** (logged once after training):
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R² Score (Coefficient of Determination)
  - MAPE (Mean Absolute Percentage Error)
- **Cross-validation metrics**:
  - cv_rmse_mean
  - cv_rmse_std
  - cv_scores (all fold scores)
- **Time-based metrics**:
  - training_time_seconds
  - prediction_latency_ms
- **Data statistics**:
  - n_train_samples
  - n_test_samples
  - n_features

#### Artifacts to Log:
- **Model file**: Serialized trained model (`.pkl` or MLflow format)
- **Scaler file**: Feature scaler for preprocessing
- **Feature names**: List of all features used
- **Feature importance**: DataFrame or plot showing feature rankings
- **Prediction plots**: Actual vs predicted visualizations
- **Residual plots**: Error distribution analysis
- **Training metadata**: JSON with additional run information
- **Data profile report**: HTML report from ydata-profiling

### 4.3.3 MLflow Experiment Lifecycle

**Run Lifecycle**:
1. **Start Run**: Create new run in experiment
2. **Log Parameters**: Record all hyperparameters (immutable)
3. **Train Model**: Execute training logic
4. **Log Metrics**: Record performance metrics (can update)
5. **Log Artifacts**: Save model and related files
6. **End Run**: Finalize and close run

**Run Status**:
- RUNNING: Currently executing
- FINISHED: Completed successfully
- FAILED: Ended with error
- KILLED: Manually terminated

### 4.3.4 Model Registry Integration

**Purpose**: Centralized repository for model versions with lifecycle management

**Model Registry Stages**:
- **None**: Newly registered, not yet staged
- **Staging**: Being tested/validated
- **Production**: Currently deployed and serving
- **Archived**: Old version, kept for history

**Model Versioning**:
- Each registration creates new version (v1, v2, v3, ...)
- Version includes: model file, source run ID, registration timestamp
- Can transition versions between stages
- Can add descriptions and tags to versions

**What to Register**:
- Best performing model from each training run
- Include model metadata: algorithm type, training date
- Link to source run for full reproducibility

---

## Step 4.4: Training Script Structure

### 4.4.1 Script Organization

**Main Components**:
1. **Configuration Loading**: Read environment variables and settings
2. **Data Loading**: Load processed data from DVC-tracked storage
3. **Feature Preparation**: Use saved feature engineering pipeline
4. **Model Training Loop**: Train multiple model types
5. **Experiment Tracking**: Log everything to MLflow
6. **Model Registration**: Register best model to registry
7. **Artifact Saving**: Store all important files

### 4.4.2 Multi-Model Experimentation

**Models to Train and Compare**:
- **Random Forest Regressor**: Ensemble of decision trees
- **Gradient Boosting Regressor**: Sequential boosting algorithm
- **Ridge Regression**: Linear model with L2 regularization
- **XGBoost Regressor**: Optimized gradient boosting (optional)

**Comparison Strategy**:
- Train all models with same data split
- Use same cross-validation strategy
- Log metrics for fair comparison
- Select best model based on primary metric (RMSE)

### 4.4.3 Hyperparameter Configuration

**Parameter Management**:
- Default parameters defined in script
- Override via environment variables
- Support for hyperparameter tuning (future enhancement)
- Log all parameters regardless of source

**Best Practices**:
- Use sensible defaults for quick training
- Document parameter ranges and effects
- Track parameter changes across runs
- Consider parameter sweep for optimization

### 4.4.4 Cross-Validation Strategy

**Time Series Cross-Validation**:
- Use TimeSeriesSplit (not regular K-Fold)
- Respects temporal order of data
- Each fold uses only past data for training
- Tests on future data (realistic scenario)

**Configuration**:
- Number of splits: 5 (configurable)
- Validation strategy: Expanding window
- Metric aggregation: Mean and standard deviation

---

## Step 4.5: Integration with Airflow DAG

### 4.5.1 Airflow Task Configuration

**Task Definition**:
- Task ID: `train_model`
- Task type: PythonOperator
- Python callable: Function that executes train.py
- Dependencies: Runs after data transformation and versioning

**Environment Variables**:
- Pass all necessary config to training script
- Include DagHub credentials securely
- Set MLflow tracking URI
- Configure model parameters

### 4.5.2 Data Handoff

**Data Flow**:
1. Previous task (transform_data) produces processed CSV
2. CSV path passed via XCom (Airflow's inter-task communication)
3. Training task reads path from XCom
4. Loads data and begins training

**XCom Usage**:
- Push: `context['task_instance'].xcom_push(key='processed_data_path', value=path)`
- Pull: `context['task_instance'].xcom_pull(task_ids='transform_data', key='processed_data_path')`

### 4.5.3 Error Handling

**Failure Scenarios**:
- Data file not found: Check DVC pull status
- MLflow connection failed: Verify DagHub credentials
- Training timeout: Set reasonable task timeout
- Memory errors: Monitor resource usage

**Recovery Strategy**:
- Retry failed tasks (2 retries with 5-minute delay)
- Log detailed error messages
- Send alerts on repeated failures
- Maintain run history in MLflow even for failed runs

### 4.5.4 Success Criteria

**Task Completion**:
- Model trained successfully
- Metrics logged to MLflow
- Artifacts saved to DagHub
- Best model registered in Model Registry
- XCom push of model path/version for downstream tasks

---

## Step 4.6: DagHub UI Navigation

### 4.6.1 Repository Overview

**Main Dashboard**:
- Files tab: Browse code repository
- Commits tab: View Git history
- Branches tab: See branch structure
- Remote tab: Access MLflow and DVC links

### 4.6.2 MLflow Experiments View

**Experiments Page**:
- List of all experiments
- Click experiment to see all runs
- Table view: Compare runs side-by-side
- Chart view: Visualize metric trends

**Run Details Page**:
- Parameters: All hyperparameters used
- Metrics: Performance measurements with graphs
- Artifacts: Downloadable files (models, plots)
- Metadata: Run info (start time, duration, status)

**Comparison Features**:
- Select multiple runs to compare
- Parallel coordinates plot
- Scatter plots for metric correlations
- Export comparison as CSV

### 4.6.3 Model Registry View

**Registry Interface**:
- List of registered models
- Click model to see all versions
- Version details: metrics, stage, source run
- Stage transition buttons (Staging → Production)

**Version Management**:
- View version history
- Compare versions
- Download specific version
- Add version descriptions and tags

### 4.6.4 DVC Data View

**Data Files Tab**:
- List of DVC-tracked files
- File versions and history
- Download specific versions
- View file metadata (.dvc files)

---

## Step 4.7: Verification and Testing

### 4.7.1 Pre-Integration Testing

**Test MLflow Locally**:
1. Run training script outside Airflow
2. Verify experiments appear in DagHub
3. Check all parameters logged correctly
4. Confirm artifacts uploaded successfully
5. Test model registration

**Verification Checklist**:
- [ ] MLflow tracking URI connects to DagHub
- [ ] Experiments created with correct name
- [ ] Parameters logged for all runs
- [ ] Metrics logged with correct values
- [ ] Artifacts uploaded (model, scaler, etc.)
- [ ] Model registered in Model Registry
- [ ] Feature importance visible
- [ ] Plots and visualizations generated

### 4.7.2 Airflow Integration Testing

**Test DAG Execution**:
1. Trigger DAG manually from Airflow UI
2. Monitor training task logs
3. Verify task completes successfully
4. Check MLflow for new experiment run
5. Validate data versioning with DVC

**Integration Checklist**:
- [ ] Airflow task triggers training script
- [ ] Environment variables passed correctly
- [ ] Data loaded from correct DVC version
- [ ] Training completes without errors
- [ ] Metrics appear in DagHub immediately
- [ ] Model artifacts saved
- [ ] XCom data passed to next task

### 4.7.3 End-to-End Validation

**Complete Pipeline Test**:
1. Trigger full DAG (extract → transform → train)
2. Verify data extraction successful
3. Confirm data quality checks pass
4. Validate feature engineering
5. Check DVC versioning
6. Verify model training and logging
7. Confirm best model registered

**Success Criteria**:
- Pipeline completes without manual intervention
- All data versioned in DVC
- All experiments tracked in MLflow
- Best model available in Model Registry
- Reproducible: Can re-run and get similar results

---

## Step 4.8: Best Practices and Guidelines

### 4.8.1 Experiment Naming Conventions

**Run Names**:
- Format: `{model_type}_{date}_{run_number}`
- Example: `random_forest_2024-11-27_001`
- Include descriptive information
- Use consistent format across team

**Tag Usage**:
- `model_type`: Algorithm used (rf, gbm, ridge)
- `status`: Development, testing, production
- `dataset_version`: DVC commit hash
- `author`: Person who ran experiment
- `purpose`: Feature testing, hyperparameter tuning, production

### 4.8.2 Metrics Logging Guidelines

**Metric Names**:
- Use lowercase with underscores: `rmse`, `mae`, `r2_score`
- Prefix related metrics: `train_rmse`, `test_rmse`, `cv_rmse`
- Be consistent across all runs
- Document metric definitions

**Logging Frequency**:
- Log training metrics once at end
- Log validation metrics per epoch (if applicable)
- Log test metrics after final evaluation
- Don't over-log (avoid hundreds of metrics per run)

### 4.8.3 Artifact Management

**Artifact Organization**:
- Use subdirectories: `models/`, `plots/`, `data/`
- Include version info in filenames
- Compress large files before uploading
- Delete temporary artifacts after run

**File Naming**:
- Model: `{model_type}_v{version}.pkl`
- Plots: `{plot_type}_{timestamp}.png`
- Data: `{data_type}_{date}.csv`

### 4.8.4 Model Registry Management

**Registration Policy**:
- Only register models that meet quality threshold
- Add detailed description to each version
- Tag with relevant metadata
- Link to source experiment run

**Stage Transition Rules**:
- None → Staging: After successful training
- Staging → Production: After validation and testing
- Production → Archived: When replaced by better model
- Document reason for each transition

---

## Step 4.9: Troubleshooting Common Issues

### Issue 1: MLflow Connection Failed

**Symptoms**:
- `MlflowException: Could not connect to tracking server`
- Experiments not appearing in DagHub

**Solutions**:
- Verify MLFLOW_TRACKING_URI is correct
- Check DAGSHUB_TOKEN is set and valid
- Test connection with `mlflow.set_tracking_uri()` and `mlflow.list_experiments()`
- Ensure network connectivity to DagHub

### Issue 2: Artifacts Upload Timeout

**Symptoms**:
- Run completes but artifacts missing
- Timeout errors in logs

**Solutions**:
- Check artifact file sizes (DagHub has limits)
- Compress large artifacts before logging
- Increase timeout settings
- Use `mlflow.log_artifact()` instead of `mlflow.log_artifacts()` for large files

### Issue 3: Model Registry Error

**Symptoms**:
- `Model registration failed`
- Version not appearing in registry

**Solutions**:
- Verify model logged as artifact first
- Check model name follows naming conventions
- Ensure sufficient permissions on DagHub
- Use correct model URI format

### Issue 4: Parameter Not Logged

**Symptoms**:
- Parameters missing in MLflow UI
- Empty parameter section

**Solutions**:
- Verify `mlflow.log_param()` called within active run
- Check parameter data type (must be string, int, float)
- Ensure run started with `mlflow.start_run()`
- Look for exceptions in training logs

### Issue 5: DVC Push Failed

**Symptoms**:
- `ERROR: failed to push data to the cloud`
- Data not appearing in DagHub

**Solutions**:
- Verify DVC remote configured correctly
- Check authentication credentials
- Ensure file not too large for free tier
- Try `dvc push --verbose` for detailed errors

---

## Expected Outcomes

By the end of Phase II, you will have:

1. ✅ **Unified Tracking System**: All experiments visible in one place (DagHub)
2. ✅ **Reproducible Experiments**: Every run fully documented with parameters and data versions
3. ✅ **Model Versioning**: Complete history of all trained models
4. ✅ **Collaborative Workspace**: Team can view and compare experiments
5. ✅ **Integrated Pipeline**: Training seamlessly triggered by Airflow
6. ✅ **Artifact Storage**: All models and files centrally stored
7. ✅ **Performance Tracking**: Metrics automatically logged and visualized

---


## Phase Completion Checklist

- [ ] DagHub account created and configured
- [ ] Repository connected (Git + DVC + MLflow)
- [ ] MLflow tracking URI configured locally and in Airflow
- [ ] Training script logs parameters, metrics, and artifacts
- [ ] Models registered in Model Registry
- [ ] Airflow DAG successfully triggers training
- [ ] End-to-end pipeline tested (extract → transform → train)
- [ ] Team members can access and view experiments
- [ ] Documentation updated with DagHub credentials
- [ ] Troubleshooting guide created for common issues

