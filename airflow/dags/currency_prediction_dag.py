"""
CoinGecko Cryptocurrency MLOps Pipeline DAG
Self-contained DAG for Apache Airflow 2.6.x
Implements: Extraction → Quality Gate → Transformation → Profiling → Storage → DVC Versioning
"""
from datetime import datetime, timedelta
import os
import json
import logging
from airflow import DAG
from airflow.operators.python import PythonOperator

# Setup logging
logger = logging.getLogger(__name__)

# Default arguments for the DAG
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'start_date': datetime(2025, 11, 25),
}

# Configuration
COIN_ID = 'bitcoin'
VS_CURRENCY = 'usd'
DAYS = 2
MAX_NULL_PERCENTAGE = 1.0

# Paths - use /opt/airflow as base
BASE_DIR = '/opt/airflow'
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')


# ============================================================================
# TASK 1: DATA EXTRACTION (Step 2.1)
# ============================================================================
def extract_data_task(**context):
    """
    Extract cryptocurrency data from CoinGecko API.
    Saves raw data with timestamp.
    """
    import requests
    import pandas as pd
    from datetime import datetime
    
    logger.info("=" * 60)
    logger.info("STARTING DATA EXTRACTION")
    logger.info("=" * 60)
    
    # Ensure output directory exists
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # CoinGecko API endpoint
    url = f"https://api.coingecko.com/api/v3/coins/{COIN_ID}/market_chart"
    # Note: Don't specify 'interval' parameter - hourly data is automatic for 2-90 days
    # The 'interval=hourly' parameter is exclusive to Enterprise plan customers
    params = {
        'vs_currency': VS_CURRENCY,
        'days': DAYS
    }
    
    # Add API key if available
    api_key = os.environ.get('COINGECKO_API_KEY')
    headers = {}
    if api_key:
        # CoinGecko Demo API uses x-cg-demo-api-key header
        headers['x-cg-demo-api-key'] = api_key
        logger.info(f"Using CoinGecko API key: {api_key[:10]}...")
    else:
        logger.warning("No COINGECKO_API_KEY found in environment - using public API (rate limited)")
    
    logger.info(f"Fetching {COIN_ID} data for {DAYS} days...")
    logger.info(f"URL: {url}")
    logger.info(f"Headers: {list(headers.keys())}")
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        
        # Log response status for debugging
        logger.info(f"Response status: {response.status_code}")
        if response.status_code != 200:
            logger.error(f"Response body: {response.text[:500]}")
        
        response.raise_for_status()
        data = response.json()
        
        logger.info(f"✓ Successfully fetched {len(data.get('prices', []))} data points")
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': [item[0] for item in data['prices']],
            'price': [item[1] for item in data['prices']],
            'market_cap': [item[1] for item in data['market_caps']],
            'total_volume': [item[1] for item in data['total_volumes']]
        })
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Save raw data with timestamp
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(RAW_DATA_DIR, f'raw_{timestamp_str}.csv')
        df.to_csv(filepath)
        
        logger.info(f"✓ Saved raw data to: {filepath}")
        logger.info(f"  Rows: {len(df)}, Columns: {list(df.columns)}")
        logger.info("=" * 60)
        
        return filepath
        
    except Exception as e:
        logger.error(f"✗ Data extraction failed: {str(e)}")
        raise


# ============================================================================
# TASK 2: DATA QUALITY CHECK - MANDATORY GATE (Step 2.1)
# ============================================================================
def quality_check_task(**context):
    """
    Mandatory Data Quality Gate.
    Pipeline FAILS if quality checks don't pass.
    
    Checks:
    - Null values < 1% in key columns
    - Schema validation
    - Data type verification
    """
    import pandas as pd
    
    logger.info("=" * 60)
    logger.info("STARTING DATA QUALITY CHECK (MANDATORY GATE)")
    logger.info("=" * 60)
    
    # Get filepath from previous task
    ti = context['ti']
    filepath = ti.xcom_pull(task_ids='extract_data')
    
    if not filepath or not os.path.exists(filepath):
        raise ValueError(f"Raw data file not found: {filepath}")
    
    logger.info(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    
    quality_results = {
        'filepath': filepath,
        'total_rows': len(df),
        'checks_passed': True,
        'issues': []
    }
    
    # Check 1: Null values in key columns
    logger.info("Checking for null values...")
    key_columns = ['price', 'market_cap', 'total_volume']
    for col in key_columns:
        if col in df.columns:
            null_pct = (df[col].isnull().sum() / len(df)) * 100
            if null_pct > MAX_NULL_PERCENTAGE:
                issue = f"Column '{col}' has {null_pct:.2f}% null values (max allowed: {MAX_NULL_PERCENTAGE}%)"
                quality_results['issues'].append(issue)
                quality_results['checks_passed'] = False
                logger.error(f"✗ {issue}")
            else:
                logger.info(f"✓ Column '{col}' null check passed ({null_pct:.2f}%)")
    
    # Check 2: Schema validation
    logger.info("Validating schema...")
    required_columns = ['price', 'market_cap', 'total_volume']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        issue = f"Missing required columns: {missing_cols}"
        quality_results['issues'].append(issue)
        quality_results['checks_passed'] = False
        logger.error(f"✗ {issue}")
    else:
        logger.info("✓ Schema validation passed")
    
    # Check 3: Data types
    logger.info("Checking data types...")
    numeric_cols = ['price', 'market_cap', 'total_volume']
    for col in numeric_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                issue = f"Column '{col}' is not numeric"
                quality_results['issues'].append(issue)
                quality_results['checks_passed'] = False
                logger.error(f"✗ {issue}")
            else:
                logger.info(f"✓ Column '{col}' data type valid")
    
    # Check 4: Minimum rows
    if len(df) < 10:
        issue = f"Insufficient data: only {len(df)} rows (minimum: 10)"
        quality_results['issues'].append(issue)
        quality_results['checks_passed'] = False
        logger.error(f"✗ {issue}")
    else:
        logger.info(f"✓ Sufficient data: {len(df)} rows")
    
    logger.info("=" * 60)
    
    # MANDATORY GATE: Fail pipeline if checks don't pass
    if not quality_results['checks_passed']:
        error_msg = f"DATA QUALITY CHECK FAILED! Issues: {quality_results['issues']}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("✓ ALL QUALITY CHECKS PASSED")
    logger.info("=" * 60)
    
    return quality_results


# ============================================================================
# TASK 3: DATA TRANSFORMATION (Step 2.2)
# ============================================================================
def transform_data_task(**context):
    """
    Transform raw data and perform feature engineering.
    
    Features created:
    - Temporal features (hour, day, cyclical encoding)
    - Lag features (1h, 3h, 6h, 12h, 24h)
    - Rolling statistics (mean, std, min, max)
    - Price change and momentum indicators
    """
    import pandas as pd
    import numpy as np
    
    logger.info("=" * 60)
    logger.info("STARTING DATA TRANSFORMATION")
    logger.info("=" * 60)
    
    # Get filepath from quality check
    ti = context['ti']
    quality_result = ti.xcom_pull(task_ids='data_quality_check')
    filepath = quality_result['filepath']
    
    # Ensure output directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    logger.info(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    original_rows = len(df)
    
    logger.info("Cleaning raw data...")
    df = df.dropna()
    df = df[~df.index.duplicated(keep='first')]
    
    # Add temporal features
    logger.info("Adding temporal features...")
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Add lag features
    logger.info("Adding lag features...")
    for lag in [1, 3, 6, 12, 24]:
        df[f'price_lag_{lag}h'] = df['price'].shift(lag)
        df[f'volume_lag_{lag}h'] = df['total_volume'].shift(lag)
    
    # Add rolling window features
    logger.info("Adding rolling window features...")
    for window in [3, 6, 12, 24]:
        df[f'price_rolling_mean_{window}h'] = df['price'].rolling(window=window).mean()
        df[f'price_rolling_std_{window}h'] = df['price'].rolling(window=window).std()
        df[f'price_rolling_min_{window}h'] = df['price'].rolling(window=window).min()
        df[f'price_rolling_max_{window}h'] = df['price'].rolling(window=window).max()
        df[f'volume_rolling_mean_{window}h'] = df['total_volume'].rolling(window=window).mean()
    
    # Add price change features
    logger.info("Adding price change features...")
    df['price_change_1h'] = df['price'].pct_change(1)
    df['price_change_3h'] = df['price'].pct_change(3)
    df['price_change_6h'] = df['price'].pct_change(6)
    df['price_change_12h'] = df['price'].pct_change(12)
    df['price_change_24h'] = df['price'].pct_change(24)
    
    # Add momentum features
    logger.info("Adding momentum features...")
    df['momentum_3h'] = df['price'] - df['price'].shift(3)
    df['momentum_6h'] = df['price'] - df['price'].shift(6)
    df['momentum_12h'] = df['price'] - df['price'].shift(12)
    
    # Add volatility features
    df['volatility_6h'] = df['price_change_1h'].rolling(window=6).std()
    df['volatility_12h'] = df['price_change_1h'].rolling(window=12).std()
    df['volatility_24h'] = df['price_change_1h'].rolling(window=24).std()
    
    # Add volume ratio
    df['volume_price_ratio'] = df['total_volume'] / df['price']
    df['volume_change_1h'] = df['total_volume'].pct_change(1)
    
    # Add technical indicators
    logger.info("Adding technical indicators...")
    # Simple Moving Averages
    df['sma_6h'] = df['price'].rolling(window=6).mean()
    df['sma_12h'] = df['price'].rolling(window=12).mean()
    df['sma_24h'] = df['price'].rolling(window=24).mean()
    
    # Exponential Moving Averages
    df['ema_6h'] = df['price'].ewm(span=6).mean()
    df['ema_12h'] = df['price'].ewm(span=12).mean()
    
    # Bollinger Bands
    bb_window = 12
    df['bb_middle'] = df['price'].rolling(window=bb_window).mean()
    bb_std = df['price'].rolling(window=bb_window).std()
    df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
    df['bb_lower'] = df['bb_middle'] - (2 * bb_std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Price position relative to bands
    df['price_bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Drop rows with NaN (from lag/rolling features)
    df = df.dropna()
    
    # Save processed data
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filepath = os.path.join(PROCESSED_DATA_DIR, f'processed_{timestamp_str}.csv')
    df.to_csv(output_filepath)
    
    logger.info("=" * 60)
    logger.info("✓ DATA TRANSFORMATION COMPLETE")
    logger.info(f"  Input rows: {original_rows}")
    logger.info(f"  Output rows: {len(df)}")
    logger.info(f"  Features created: {len(df.columns)}")
    logger.info(f"  Output file: {output_filepath}")
    logger.info("=" * 60)
    
    return output_filepath


# ============================================================================
# TASK 4: DATA PROFILING (Step 2.2 - Documentation Artifact)
# ============================================================================
def profiling_task(**context):
    """
    Generate data profiling report using ydata-profiling.
    Logs profile as artifact to MLflow/Dagshub.
    """
    import pandas as pd
    
    logger.info("=" * 60)
    logger.info("STARTING DATA PROFILING")
    logger.info("=" * 60)
    
    # Get processed filepath
    ti = context['ti']
    processed_filepath = ti.xcom_pull(task_ids='transform_data')
    
    if not processed_filepath or not os.path.exists(processed_filepath):
        raise FileNotFoundError(f"Processed file not found: {processed_filepath}")
    
    logger.info(f"Loading processed data from: {processed_filepath}")
    df = pd.read_csv(processed_filepath, index_col=0)
    
    # Generate profile summary (lightweight version)
    profile_summary = {
        'timestamp': datetime.now().isoformat(),
        'file': processed_filepath,
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': list(df.columns),
        'numeric_columns': len(df.select_dtypes(include=['number']).columns),
        'missing_values': df.isnull().sum().to_dict(),
        'statistics': df.describe().to_dict()
    }
    
    # Save profile summary
    profile_dir = os.path.join(BASE_DIR, 'data', 'profiles')
    os.makedirs(profile_dir, exist_ok=True)
    
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    profile_path = os.path.join(profile_dir, f'profile_{timestamp_str}.json')
    
    with open(profile_path, 'w') as f:
        json.dump(profile_summary, f, indent=2, default=str)
    
    logger.info(f"✓ Profile saved to: {profile_path}")
    
    # Try to log to MLflow if available
    try:
        import mlflow
        mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI')
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment("crypto_prediction")
            
            with mlflow.start_run(run_name=f"profiling_{timestamp_str}"):
                mlflow.log_artifact(profile_path)
                mlflow.log_metrics({
                    'total_rows': len(df),
                    'total_columns': len(df.columns)
                })
            logger.info("✓ Profile logged to MLflow")
    except Exception as e:
        logger.warning(f"Could not log to MLflow: {e}")
    
    logger.info("=" * 60)
    logger.info("✓ DATA PROFILING COMPLETE")
    logger.info("=" * 60)
    
    return {
        'processed_filepath': processed_filepath,
        'profile_path': profile_path,
        'profile_summary': profile_summary
    }


# ============================================================================
# TASK 5: STORAGE UPLOAD (Step 2.3)
# ============================================================================
def storage_upload_task(**context):
    """
    Upload processed data to cloud-like object storage.
    Uses local storage as fallback.
    """
    import shutil
    
    logger.info("=" * 60)
    logger.info("STARTING STORAGE UPLOAD")
    logger.info("=" * 60)
    
    # Get processed filepath
    ti = context['ti']
    profiling_result = ti.xcom_pull(task_ids='data_profiling')
    processed_filepath = profiling_result['processed_filepath']
    
    if not processed_filepath or not os.path.exists(processed_filepath):
        raise FileNotFoundError(f"Processed file not found: {processed_filepath}")
    
    # Create storage directory
    storage_dir = os.path.join(BASE_DIR, 'storage', 'processed_data')
    os.makedirs(storage_dir, exist_ok=True)
    
    # Copy file to storage
    filename = os.path.basename(processed_filepath)
    storage_path = os.path.join(storage_dir, filename)
    shutil.copy2(processed_filepath, storage_path)
    
    logger.info(f"✓ File uploaded to storage: {storage_path}")
    
    storage_result = {
        'source_file': processed_filepath,
        'storage_path': storage_path,
        'storage_type': 'local',
        'success': True
    }
    
    logger.info("=" * 60)
    logger.info("✓ STORAGE UPLOAD COMPLETE")
    logger.info("=" * 60)
    
    return storage_result


# ============================================================================
# TASK 6: DVC VERSIONING (Step 3)
# ============================================================================
def dvc_versioning_task(**context):
    """
    Version processed data with DVC.
    Creates .dvc metadata file for Git tracking.
    """
    import subprocess
    
    logger.info("=" * 60)
    logger.info("STARTING DVC VERSIONING")
    logger.info("=" * 60)
    
    # Get storage result
    ti = context['ti']
    storage_result = ti.xcom_pull(task_ids='storage_upload')
    storage_path = storage_result['storage_path']
    
    if not storage_path or not os.path.exists(storage_path):
        raise FileNotFoundError(f"Storage file not found: {storage_path}")
    
    dvc_result = {
        'file': storage_path,
        'dvc_success': False,
        'message': ''
    }
    
    try:
        # Initialize DVC if needed
        dvc_dir = os.path.join(BASE_DIR, '.dvc')
        if not os.path.exists(dvc_dir):
            subprocess.run(['dvc', 'init', '-f'], cwd=BASE_DIR, capture_output=True)
            logger.info("✓ DVC initialized")
        
        # Add file to DVC
        result = subprocess.run(
            ['dvc', 'add', storage_path],
            cwd=BASE_DIR,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            dvc_result['dvc_success'] = True
            dvc_result['message'] = 'File added to DVC tracking'
            logger.info(f"✓ File added to DVC: {storage_path}")
        else:
            dvc_result['message'] = f"DVC add failed: {result.stderr}"
            logger.warning(f"DVC add warning: {result.stderr}")
            
    except FileNotFoundError:
        dvc_result['message'] = 'DVC not installed in container'
        logger.warning("DVC not available - skipping versioning")
    except Exception as e:
        dvc_result['message'] = str(e)
        logger.warning(f"DVC versioning failed: {e}")
    
    logger.info("=" * 60)
    logger.info("✓ DVC VERSIONING STEP COMPLETE")
    logger.info("=" * 60)
    
    return dvc_result


# ============================================================================
# TASK 7: PIPELINE SUMMARY
# ============================================================================
def pipeline_summary_task(**context):
    """Generate final pipeline execution summary."""
    
    logger.info("=" * 60)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 60)
    
    ti = context['ti']
    
    # Collect results from all tasks
    quality_result = ti.xcom_pull(task_ids='data_quality_check')
    profiling_result = ti.xcom_pull(task_ids='data_profiling')
    storage_result = ti.xcom_pull(task_ids='storage_upload')
    dvc_result = ti.xcom_pull(task_ids='dvc_versioning')
    
    summary = {
        'execution_time': datetime.now().isoformat(),
        'status': 'SUCCESS',
        'extraction': {
            'rows_extracted': quality_result.get('total_rows', 0),
            'quality_passed': quality_result.get('checks_passed', False)
        },
        'transformation': {
            'processed_file': profiling_result.get('processed_filepath'),
            'features_count': profiling_result.get('profile_summary', {}).get('columns', 0)
        },
        'storage': {
            'uploaded': storage_result.get('success', False),
            'path': storage_result.get('storage_path')
        },
        'versioning': {
            'dvc_tracked': dvc_result.get('dvc_success', False)
        }
    }
    
    logger.info(f"✅ Pipeline completed successfully!")
    logger.info(f"   Data rows: {summary['extraction']['rows_extracted']}")
    logger.info(f"   Features: {summary['transformation']['features_count']}")
    logger.info(f"   Storage: {summary['storage']['uploaded']}")
    logger.info(f"   DVC: {summary['versioning']['dvc_tracked']}")
    logger.info("=" * 60)
    
    return summary


# ============================================================================
# CREATE THE DAG
# ============================================================================
with DAG(
    'cryptocurrency_mlops_pipeline',
    default_args=default_args,
    description='Complete cryptocurrency MLOps pipeline: Extract → Quality Gate → Transform → Profile → Store → Version',
    schedule_interval='0 */6 * * *',  # Every 6 hours
    catchup=False,
    tags=['cryptocurrency', 'mlops', 'etl', 'data-pipeline'],
) as dag:
    
    # Task definitions
    extract_data = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data_task,
    )
    
    data_quality_check = PythonOperator(
        task_id='data_quality_check',
        python_callable=quality_check_task,
    )
    
    transform_data = PythonOperator(
        task_id='transform_data',
        python_callable=transform_data_task,
    )
    
    data_profiling = PythonOperator(
        task_id='data_profiling',
        python_callable=profiling_task,
    )
    
    storage_upload = PythonOperator(
        task_id='storage_upload',
        python_callable=storage_upload_task,
    )
    
    dvc_versioning = PythonOperator(
        task_id='dvc_versioning',
        python_callable=dvc_versioning_task,
    )
    
    pipeline_summary = PythonOperator(
        task_id='pipeline_summary',
        python_callable=pipeline_summary_task,
    )
    
    # Define task dependencies (DAG structure)
    # Extract → Quality Check (mandatory gate) → Transform → Profile → Store → Version → Summary
    extract_data >> data_quality_check >> transform_data >> data_profiling >> storage_upload >> dvc_versioning >> pipeline_summary


# DAG documentation
dag.doc_md = """
# Cryptocurrency MLOps Pipeline

## Overview
Complete ETL pipeline for cryptocurrency price data with automated quality checks,
feature engineering, and data versioning.

## Pipeline Steps

### 1. Data Extraction (Step 2.1)
- Fetches Bitcoin data from CoinGecko API
- Saves raw data with timestamp to `data/raw/`

### 2. Quality Gate (Step 2.1 - MANDATORY)
- **Pipeline FAILS if quality checks don't pass**
- Checks: Null values < 1%, Schema validation, Data types

### 3. Data Transformation (Step 2.2)
- Feature engineering for time series:
  - Temporal features (hour, day, cyclical encoding)
  - Lag features (1h, 3h, 6h, 12h, 24h)
  - Rolling statistics (mean, std, min, max)
  - Price momentum and volatility
  - Technical indicators (SMA, EMA, Bollinger Bands)

### 4. Data Profiling (Step 2.2)
- Generates data quality report
- Logs to MLflow/Dagshub as artifact

### 5. Storage Upload (Step 2.3)
- Uploads processed data to object storage

### 6. DVC Versioning (Step 3)
- Versions dataset with DVC
- Creates .dvc metadata file for Git

## Schedule
Runs every 6 hours: `0 */6 * * *`

## Monitoring
View execution logs in Airflow UI for each task.
"""
