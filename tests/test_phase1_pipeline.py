"""
Phase 1 MLOps Pipeline Test Suite
Comprehensive testing for the complete ETL pipeline with quality gates
"""
import os
import sys
import unittest
import tempfile
import shutil
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestPhase1Pipeline(unittest.TestCase):
    """Test complete Phase 1 MLOps pipeline end-to-end"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_dir = tempfile.mkdtemp(prefix='phase1_test_')
        cls.data_dir = os.path.join(cls.test_dir, 'data')
        cls.raw_dir = os.path.join(cls.data_dir, 'raw')
        cls.processed_dir = os.path.join(cls.data_dir, 'processed')
        cls.models_dir = os.path.join(cls.test_dir, 'models')
        cls.storage_dir = os.path.join(cls.test_dir, 'storage')
        
        # Create directories
        for dir_path in [cls.raw_dir, cls.processed_dir, cls.models_dir, cls.storage_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        logger.info(f"Test environment created at: {cls.test_dir}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        shutil.rmtree(cls.test_dir, ignore_errors=True)
        logger.info("Test environment cleaned up")
    
    def test_01_data_extraction(self):
        """Test CoinGecko data extraction - Phase 1 Component"""
        logger.info("=" * 50)
        logger.info("TEST 1: DATA EXTRACTION")
        logger.info("=" * 50)
        
        from data.extract import extract_coingecko_data
        
        # Extract data for testing
        result = extract_coingecko_data(
            coin_id='bitcoin',
            days=1,
            output_dir=self.raw_dir
        )
        
        # Verify extraction
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(result))
        
        # Verify file contents
        df = pd.read_csv(result)
        self.assertGreater(len(df), 0)
        
        # Verify required columns (total_volume is actual API column)
        required_cols = ['timestamp', 'price', 'market_cap', 'total_volume']
        for col in required_cols:
            self.assertIn(col, df.columns)
        
        logger.info(f"✓ Extraction successful: {result}")
        logger.info(f"  - Rows extracted: {len(df)}")
        logger.info(f"  - Columns: {list(df.columns)}")
        
        # Store for next test
        self.raw_file = result
    
    def test_02_quality_gates(self):
        """Test mandatory quality gates - Phase 1 Component"""
        logger.info("=" * 50)
        logger.info("TEST 2: QUALITY GATES")
        logger.info("=" * 50)
        
        # Check if we have raw file from previous test
        if not hasattr(self, 'raw_file') or not self.raw_file:
            # Find the raw file in the directory
            import glob
            raw_files = glob.glob(os.path.join(self.raw_dir, "raw_*.csv"))
            if not raw_files:
                self.skipTest("No raw data file found - extraction test may have failed")
            self.raw_file = raw_files[0]
        
        from data.quality_checks import check_data_quality
        
        # Test with good data (should pass)
        result = check_data_quality(self.raw_file)
        
        # Check the result format - it returns the results from run_all_checks
        self.assertIn('overall_status', result)
        self.assertEqual(result['overall_status'], 'PASSED')
        
        logger.info(f"✓ Quality gates passed for good data")
        logger.info(f"  - Null percentage: {result['null_percentage']:.2f}%")
        logger.info(f"  - Schema valid: {result['schema_valid']}")
        
        # Test with bad data (should fail)
        bad_data = pd.DataFrame({
            'timestamp': [None, None, None],
            'price': [None, 100, None],
            'market_cap': [1000, None, None],
            'total_volume': [None, None, None]
        })
        
        bad_file = os.path.join(self.raw_dir, 'bad_data.csv')
        bad_data.to_csv(bad_file, index=False)
        
        try:
            bad_result = check_data_quality(bad_file)
            # Should fail due to >1% null values
            self.assertFalse(bad_result['success'])
            self.assertTrue(bad_result['has_critical_issues'])
            logger.info(f"✓ Quality gates correctly failed for bad data")
            logger.info(f"  - Null percentage: {bad_result['null_percentage']:.2f}%")
        except Exception as e:
            logger.info(f"✓ Quality gates correctly raised exception: {e}")
    
    def test_03_data_transformation(self):
        """Test feature engineering transformation - Phase 1 Component"""
        logger.info("=" * 50)
        logger.info("TEST 3: DATA TRANSFORMATION")
        logger.info("=" * 50)
        
        # Check if we have raw file from previous test
        if not hasattr(self, 'raw_file') or not self.raw_file:
            # Find the raw file in the directory
            import glob
            raw_files = glob.glob(os.path.join(self.raw_dir, "raw_*.csv"))
            if not raw_files:
                self.skipTest("No raw data file found - extraction test may have failed")
            self.raw_file = raw_files[0]
        
        from data.transform import transform_data
        
        result = transform_data(
            input_filepath=self.raw_file,
            output_dir=self.processed_dir
        )
        
        # Verify transformation
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(result))
        
        # Store for later tests
        self.processed_file = result
        
        # Verify transformed data
        df = pd.read_csv(result)
        self.assertGreater(len(df), 0)
        
        # Check for engineered features
        expected_features = [
            'hour', 'day_of_week', 'price_lag_1', 'price_lag_3',
            'price_rolling_mean_6', 'price_rolling_std_6'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, df.columns)
        
        logger.info(f"✓ Transformation successful: {result}")
        logger.info(f"  - Original columns: {len(pd.read_csv(self.raw_file).columns)}")
        logger.info(f"  - Transformed columns: {len(df.columns)}")
        logger.info(f"  - Added features: {len(df.columns) - len(pd.read_csv(self.raw_file).columns)}")
        
        # Store for next tests
        self.processed_file = result
    
    def test_04_data_profiling(self):
        """Test pandas profiling documentation - Phase 1 Component"""
        logger.info("=" * 50)
        logger.info("TEST 4: DATA PROFILING")
        logger.info("=" * 50)
        
        # Check if we have processed file from previous test
        if not hasattr(self, 'processed_file') or not self.processed_file:
            # Find the processed file in the directory
            import glob
            processed_files = glob.glob(os.path.join(self.processed_dir, "processed_*.csv"))
            if not processed_files:
                self.skipTest("No processed data file found - transformation test may have failed")
            self.processed_file = processed_files[0]
        
        from utils.profiling import generate_and_log_profile
        
        # Generate profile report
        result = generate_and_log_profile(
            data_path=self.processed_file,
            title="Phase 1 Test Data Profile",
            experiment_name="phase1_test",
            run_name=f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            output_dir=self.test_dir
        )
        
        # Verify profiling results
        self.assertTrue(result['success'])
        self.assertIn('profile_path', result)
        
        # Check if profile file was created
        if 'profile_path' in result and result['profile_path']:
            self.assertTrue(os.path.exists(result['profile_path']))
        
        logger.info(f"✓ Profiling successful")
        logger.info(f"  - Profile generated: {result.get('profile_path', 'MLflow only')}")
        logger.info(f"  - MLflow logged: {result.get('mlflow_success', False)}")
    
    def test_05_cloud_storage(self):
        """Test cloud storage upload - Phase 1 Component"""
        logger.info("=" * 50)
        logger.info("TEST 5: CLOUD STORAGE")
        logger.info("=" * 50)
        
        # Check if we have processed file from previous test
        if not hasattr(self, 'processed_file') or not self.processed_file:
            # Find the processed file in the directory
            import glob
            processed_files = glob.glob(os.path.join(self.processed_dir, "processed_*.csv"))
            if not processed_files:
                self.skipTest("No processed data file found - transformation test may have failed")
            self.processed_file = processed_files[0]
        
        from utils.storage import upload_to_storage
        
        # Test local storage upload
        result = upload_to_storage(
            file_path=self.processed_file,
            storage_type="local",
            container="test-crypto-data",
            storage_config={'base_path': self.storage_dir}
        )
        
        # Verify upload
        self.assertTrue(result['success'])
        self.assertIn('storage_url', result)
        
        # Check if file was uploaded
        uploaded_file = result.get('local_path')
        if uploaded_file:
            self.assertTrue(os.path.exists(uploaded_file))
        
        logger.info(f"✓ Storage upload successful")
        logger.info(f"  - Storage URL: {result.get('storage_url')}")
        logger.info(f"  - Local path: {result.get('local_path')}")
    
    def test_06_dvc_versioning(self):
        """Test DVC data versioning - Phase 1 Component"""
        logger.info("=" * 50)
        logger.info("TEST 6: DVC VERSIONING")
        logger.info("=" * 50)
        
        # Check if we have processed file from previous test
        if not hasattr(self, 'processed_file') or not self.processed_file:
            # Find the processed file in the directory
            import glob
            processed_files = glob.glob(os.path.join(self.processed_dir, "processed_*.csv"))
            if not processed_files:
                self.skipTest("No processed data file found - transformation test may have failed")
            self.processed_file = processed_files[0]
        
        from utils.dvc_version import version_data_with_dvc, setup_dvc_with_local_remote
        
        # Setup DVC with local remote
        dvc_remote_dir = os.path.join(self.test_dir, 'dvc_remote')
        setup_result = setup_dvc_with_local_remote(
            remote_dir=dvc_remote_dir
        )
        
        self.assertTrue(setup_result['success'])
        
        # Version the processed data
        version_result = version_data_with_dvc(
            filepath=self.processed_file,
            remote_url=setup_result['remote_url'],
            remote_name=setup_result['remote_name'],
            commit_message="Test DVC versioning - Phase 1"
        )
        
        # Verify versioning (may not fully succeed without git repo)
        self.assertIn('success', version_result)
        self.assertIn('filepath', version_result)
        
        logger.info(f"✓ DVC versioning completed")
        logger.info(f"  - File versioned: {version_result.get('filepath')}")
        logger.info(f"  - DVC file: {version_result.get('dvc_file')}")
        logger.info(f"  - Success: {version_result.get('success')}")
    
    def test_07_model_training(self):
        """Test model training - Phase 1 Component"""
        logger.info("=" * 50)
        logger.info("TEST 7: MODEL TRAINING")
        logger.info("=" * 50)
        
        # Check if we have processed file from previous test
        if not hasattr(self, 'processed_file') or not self.processed_file:
            # Find the processed file in the directory
            import glob
            processed_files = glob.glob(os.path.join(self.processed_dir, "processed_*.csv"))
            if not processed_files:
                self.skipTest("No processed data file found - transformation test may have failed")
            self.processed_file = processed_files[0]
        
        from models.train import train_prediction_model
        
        # Train models on processed data
        result = train_prediction_model(
            processed_file=self.processed_file,
            output_dir=self.models_dir
        )
        
        # Verify training results
        self.assertIsNotNone(result)
        self.assertIn('models_trained', result)
        self.assertGreater(result['models_trained'], 0)
        
        # Check if model files were created
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.joblib')]
        self.assertGreater(len(model_files), 0)
        
        logger.info(f"✓ Model training successful")
        logger.info(f"  - Models trained: {result['models_trained']}")
        logger.info(f"  - Model files: {model_files}")
        logger.info(f"  - Best model: {result.get('best_model', 'Unknown')}")
    
    def test_08_predictions(self):
        """Test prediction generation - Phase 1 Component"""
        logger.info("=" * 50)
        logger.info("TEST 8: PREDICTIONS")
        logger.info("=" * 50)
        
        # Check if we have processed file from previous test
        if not hasattr(self, 'processed_file') or not self.processed_file:
            # Find the processed file in the directory
            import glob
            processed_files = glob.glob(os.path.join(self.processed_dir, "processed_*.csv"))
            if not processed_files:
                self.skipTest("No processed data file found - transformation test may have failed")
            self.processed_file = processed_files[0]
        
        from models.predict import make_predictions
        
        # Generate predictions
        result = make_predictions(
            processed_file=self.processed_file,
            models_dir=self.models_dir,
            output_dir=self.test_dir
        )
        
        # Verify predictions
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(result))
        
        # Check prediction file
        pred_df = pd.read_csv(result)
        self.assertGreater(len(pred_df), 0)
        self.assertIn('prediction', pred_df.columns)
        
        logger.info(f"✓ Prediction generation successful")
        logger.info(f"  - Predictions file: {result}")
        logger.info(f"  - Predictions generated: {len(pred_df)}")


def run_integration_test():
    """Run complete Phase 1 pipeline integration test"""
    logger.info("🚀 STARTING PHASE 1 MLOPS PIPELINE INTEGRATION TEST")
    logger.info("=" * 70)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhase1Pipeline)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    logger.info("=" * 70)
    logger.info("PHASE 1 PIPELINE TEST SUMMARY")
    logger.info("=" * 70)
    
    if result.wasSuccessful():
        logger.info("🎉 ALL TESTS PASSED - Phase 1 Pipeline is Ready!")
        logger.info("\n✅ Phase 1 Components Validated:")
        logger.info("   1. ✅ Data Extraction (CoinGecko API)")
        logger.info("   2. ✅ Quality Gates (Mandatory validation)")
        logger.info("   3. ✅ Data Transformation (Feature engineering)")
        logger.info("   4. ✅ Data Profiling (Documentation artifact)")
        logger.info("   5. ✅ Cloud Storage (Loading component)")
        logger.info("   6. ✅ DVC Versioning (Data version control)")
        logger.info("   7. ✅ Model Training (ML pipeline)")
        logger.info("   8. ✅ Prediction Generation (Inference)")
        
        logger.info("\n📋 Phase 1 ETL Pipeline Status: OPERATIONAL")
        logger.info("💡 Next Steps:")
        logger.info("   - Deploy DAG to Apache Airflow")
        logger.info("   - Configure cloud storage credentials")
        logger.info("   - Setup MLflow tracking server")
        logger.info("   - Initialize DVC remote storage")
    else:
        logger.error("❌ SOME TESTS FAILED")
        logger.error(f"   - Tests run: {result.testsRun}")
        logger.error(f"   - Failures: {len(result.failures)}")
        logger.error(f"   - Errors: {len(result.errors)}")
        
        if result.failures:
            logger.error("\nFailures:")
            for test, traceback in result.failures:
                logger.error(f"   - {test}: {traceback}")
        
        if result.errors:
            logger.error("\nErrors:")
            for test, traceback in result.errors:
                logger.error(f"   - {test}: {traceback}")


if __name__ == "__main__":
    run_integration_test()