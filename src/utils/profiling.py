"""
Pandas Profiling Module for Data Quality Reports
Generates comprehensive data quality and feature analysis reports
"""
import pandas as pd
import logging
from datetime import datetime
import os
from typing import Optional, Dict, Any
import json

# Try to import ydata_profiling (new name) or pandas_profiling (old name)
try:
    from ydata_profiling import ProfileReport
except ImportError:
    try:
        from pandas_profiling import ProfileReport
    except ImportError:
        ProfileReport = None
        logging.warning("Neither ydata-profiling nor pandas-profiling is installed. Install one with: pip install ydata-profiling")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProfiler:
    """
    Data profiling utility for generating comprehensive reports
    """
    
    def __init__(self):
        """Initialize the data profiler"""
        if ProfileReport is None:
            raise ImportError("Profiling library not available. Install with: pip install ydata-profiling")
    
    def generate_profile_report(self, 
                              df: pd.DataFrame, 
                              title: str = "Data Quality Report",
                              output_file: str = None,
                              minimal: bool = False) -> ProfileReport:
        """
        Generate a comprehensive data profile report
        
        Args:
            df: DataFrame to profile
            title: Report title
            output_file: Path to save HTML report (optional)
            minimal: Generate minimal report for faster processing
            
        Returns:
            ProfileReport object
        """
        logger.info(f"Generating profile report for dataset with {len(df)} rows and {len(df.columns)} columns")
        
        # Configure profiling settings
        config = {
            'title': title,
            'pool_size': 1,  # Reduce memory usage
            'progress_bar': True,
            'infer_dtypes': True,
            'correlations': {
                'auto': {'calculate': not minimal},
                'pearson': {'calculate': not minimal},
                'spearman': {'calculate': not minimal},
                'kendall': {'calculate': False},  # Skip for performance
                'phi_k': {'calculate': False},
                'cramers': {'calculate': False}
            },
            'interactions': {
                'continuous': not minimal,
                'targets': []
            },
            'missing_diagrams': {
                'bar': not minimal,
                'matrix': not minimal,
                'heatmap': not minimal,
                'dendrogram': False
            },
            'duplicates': {
                'head': 10 if not minimal else 3,
                'key': False  # Skip for performance
            },
            'samples': {
                'head': 10,
                'tail': 10,
                'random': 10 if not minimal else 0
            }
        }
        
        # Generate profile report
        profile = ProfileReport(df, **config)
        
        # Save to file if specified
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            profile.to_file(output_file)
            logger.info(f"✓ Profile report saved to: {output_file}")
        
        return profile
    
    def extract_summary_metrics(self, profile: ProfileReport) -> Dict[str, Any]:
        """
        Extract key metrics from profile report for MLflow logging
        
        Args:
            profile: ProfileReport object
            
        Returns:
            Dictionary with summary metrics
        """
        try:
            # Get the profile description
            description = profile.get_description()
            
            # Extract key metrics
            metrics = {
                'dataset_rows': int(description.get('table', {}).get('n', 0)),
                'dataset_columns': int(description.get('table', {}).get('n_var', 0)),
                'missing_cells': int(description.get('table', {}).get('n_cells_missing', 0)),
                'missing_percentage': float(description.get('table', {}).get('p_cells_missing', 0.0) * 100),
                'duplicate_rows': int(description.get('table', {}).get('n_duplicates', 0)),
                'memory_size_mb': float(description.get('table', {}).get('memory_size', 0)) / (1024 * 1024),
            }
            
            # Extract variable-specific metrics
            variables = description.get('variables', {})
            
            # Count variable types
            type_counts = {}
            missing_by_column = {}
            
            for var_name, var_info in variables.items():
                var_type = var_info.get('type', 'unknown')
                type_counts[var_type] = type_counts.get(var_type, 0) + 1
                
                # Missing values per column
                missing_count = var_info.get('n_missing', 0)
                missing_percentage = var_info.get('p_missing', 0.0) * 100
                
                if missing_count > 0:
                    missing_by_column[var_name] = {
                        'count': missing_count,
                        'percentage': missing_percentage
                    }
            
            metrics['variable_types'] = type_counts
            metrics['columns_with_missing'] = len(missing_by_column)
            metrics['missing_by_column'] = missing_by_column
            
            # Add data quality score (custom calculation)
            quality_score = 100.0
            quality_score -= min(metrics['missing_percentage'] * 2, 50)  # Penalize missing values
            quality_score -= min(metrics['duplicate_rows'] / max(metrics['dataset_rows'], 1) * 100 * 3, 30)  # Penalize duplicates
            metrics['data_quality_score'] = max(quality_score, 0.0)
            
            logger.info(f"✓ Extracted {len(metrics)} summary metrics from profile report")
            return metrics
            
        except Exception as e:
            logger.error(f"Error extracting summary metrics: {e}")
            # Return basic metrics
            return {
                'dataset_rows': len(profile.df) if hasattr(profile, 'df') else 0,
                'dataset_columns': len(profile.df.columns) if hasattr(profile, 'df') else 0,
                'data_quality_score': 0.0,
                'error': str(e)
            }


def generate_and_log_profile(input_filepath: str,
                           output_dir: str = "reports/profiles",
                           mlflow_tracking_uri: str = None,
                           experiment_name: str = "data_profiling") -> str:
    """
    Main profiling function for Airflow tasks
    
    Args:
        input_filepath: Path to processed CSV file
        output_dir: Output directory for reports
        mlflow_tracking_uri: MLflow tracking server URI
        experiment_name: MLflow experiment name
        
    Returns:
        Path to generated HTML report
    """
    logger.info("=" * 60)
    logger.info("STARTING DATA PROFILING")
    logger.info("=" * 60)
    
    try:
        # Load processed data
        df = pd.read_csv(input_filepath, index_col=0, parse_dates=True)
        logger.info(f"Loaded data: {df.shape}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create profiler
        profiler = DataProfiler()
        
        # Generate profile report
        report_filename = f"data_profile_{timestamp}.html"
        report_path = os.path.join(output_dir, report_filename)
        
        # Use minimal profiling for large datasets
        minimal_mode = len(df) > 10000 or len(df.columns) > 50
        
        profile = profiler.generate_profile_report(
            df=df,
            title=f"Cryptocurrency Data Quality Report - {timestamp}",
            output_file=report_path,
            minimal=minimal_mode
        )
        
        # Extract summary metrics
        summary_metrics = profiler.extract_summary_metrics(profile)
        
        # Log to MLflow if tracking URI provided
        if mlflow_tracking_uri:
            try:
                import mlflow
                import mlflow.artifacts
                
                # Set tracking URI
                mlflow.set_tracking_uri(mlflow_tracking_uri)
                
                # Set or create experiment
                try:
                    experiment = mlflow.get_experiment_by_name(experiment_name)
                    if experiment is None:
                        experiment_id = mlflow.create_experiment(experiment_name)
                    else:
                        experiment_id = experiment.experiment_id
                    mlflow.set_experiment(experiment_id)
                except Exception as e:
                    logger.warning(f"Could not set MLflow experiment: {e}")
                    mlflow.set_experiment(experiment_name)
                
                # Start MLflow run
                with mlflow.start_run(run_name=f"data_profile_{timestamp}"):
                    # Log metrics
                    for metric_name, metric_value in summary_metrics.items():
                        if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool):
                            mlflow.log_metric(metric_name, metric_value)
                    
                    # Log parameters
                    mlflow.log_param("input_file", os.path.basename(input_filepath))
                    mlflow.log_param("profiling_timestamp", timestamp)
                    mlflow.log_param("minimal_mode", minimal_mode)
                    
                    # Log the HTML report as artifact
                    mlflow.log_artifact(report_path, "reports")
                    
                    # Log summary metrics as JSON artifact
                    metrics_path = os.path.join(output_dir, f"summary_metrics_{timestamp}.json")
                    with open(metrics_path, 'w') as f:
                        json.dump(summary_metrics, f, indent=2, default=str)
                    mlflow.log_artifact(metrics_path, "metrics")
                
                logger.info("✓ Results logged to MLflow")
                
            except ImportError:
                logger.warning("MLflow not available. Skipping MLflow logging.")
            except Exception as e:
                logger.error(f"Error logging to MLflow: {e}")
        
        logger.info("=" * 60)
        logger.info("✓ DATA PROFILING COMPLETE")
        logger.info(f"  Report: {report_path}")
        logger.info(f"  Data Quality Score: {summary_metrics.get('data_quality_score', 'N/A'):.1f}%")
        logger.info(f"  Rows: {summary_metrics.get('dataset_rows', 'N/A'):,}")
        logger.info(f"  Columns: {summary_metrics.get('dataset_columns', 'N/A')}")
        logger.info(f"  Missing %: {summary_metrics.get('missing_percentage', 'N/A'):.2f}%")
        logger.info("=" * 60)
        
        return report_path
        
    except Exception as e:
        logger.error(f"✗ DATA PROFILING FAILED: {e}")
        raise


if __name__ == "__main__":
    # Test the profiler
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        report_path = generate_and_log_profile(
            input_filepath=input_file,
            output_dir="reports/profiles"
        )
        print(f"Profile report generated: {report_path}")
    else:
        print("Usage: python profiling.py <input_filepath>")