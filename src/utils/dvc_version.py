"""
DVC (Data Version Control) Module - Phase 1 Versioning Component
Handles data versioning with DVC for processed datasets
"""
import os
import subprocess
import logging
from typing import Optional, Dict, List
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DVCManager:
    """Manage DVC operations for data versioning"""
    
    def __init__(self, repo_path: str = "."):
        """
        Initialize DVC manager
        
        Args:
            repo_path: Path to git repository root
        """
        self.repo_path = repo_path
        
    def _run_command(self, command: List[str], cwd: str = None) -> tuple:
        """
        Run a shell command and return output
        
        Args:
            command: Command as list of strings
            cwd: Working directory (default: repo_path)
            
        Returns:
            Tuple of (success: bool, output: str, error: str)
        """
        if cwd is None:
            cwd = self.repo_path
        
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            success = result.returncode == 0
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            logger.error(f"Command timeout: {' '.join(command)}")
            return False, "", "Command timeout"
        except Exception as e:
            logger.error(f"Error running command: {e}")
            return False, "", str(e)
    
    def init_dvc(self) -> bool:
        """
        Initialize DVC in the repository
        
        Returns:
            True if successful
        """
        logger.info("Initializing DVC...")
        
        success, stdout, stderr = self._run_command(['dvc', 'init'])
        
        if success or "already initialized" in stderr.lower():
            logger.info("✓ DVC initialized")
            return True
        else:
            logger.error(f"Failed to initialize DVC: {stderr}")
            return False
    
    def add_remote(self, 
                   remote_name: str,
                   remote_url: str,
                   default: bool = True) -> bool:
        """
        Add DVC remote storage
        
        Args:
            remote_name: Name of the remote
            remote_url: URL of remote storage (can be local path for testing)
            default: Set as default remote
            
        Returns:
            True if successful
        """
        logger.info(f"Adding DVC remote: {remote_name} -> {remote_url}")
        
        # Add remote
        success, stdout, stderr = self._run_command([
            'dvc', 'remote', 'add', '-f', remote_name, remote_url
        ])
        
        if not success:
            logger.error(f"Failed to add remote: {stderr}")
            return False
        
        # Set as default if requested
        if default:
            success, stdout, stderr = self._run_command([
                'dvc', 'remote', 'default', remote_name
            ])
            
            if not success:
                logger.warning(f"Failed to set default remote: {stderr}")
        
        logger.info(f"✓ DVC remote added: {remote_name}")
        return True
    
    def add_file(self, filepath: str) -> bool:
        """
        Add file to DVC tracking
        
        Args:
            filepath: Path to file to track
            
        Returns:
            True if successful
        """
        logger.info(f"Adding file to DVC tracking: {filepath}")
        
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return False
        
        success, stdout, stderr = self._run_command(['dvc', 'add', filepath])
        
        if success:
            logger.info(f"✓ File added to DVC: {filepath}")
            logger.info(f"  DVC metadata file created: {filepath}.dvc")
            return True
        else:
            logger.error(f"Failed to add file to DVC: {stderr}")
            return False
    
    def push(self, remote_name: Optional[str] = None) -> bool:
        """
        Push data to DVC remote storage
        
        Args:
            remote_name: Name of remote (uses default if None)
            
        Returns:
            True if successful
        """
        logger.info("Pushing data to DVC remote storage...")
        
        command = ['dvc', 'push']
        if remote_name:
            command.extend(['-r', remote_name])
        
        success, stdout, stderr = self._run_command(command)
        
        if success:
            logger.info("✓ Data pushed to DVC remote storage")
            return True
        else:
            logger.error(f"Failed to push to DVC remote: {stderr}")
            return False
    
    def git_add_dvc_files(self, filepath: str) -> bool:
        """
        Add DVC metadata files to git staging
        
        Args:
            filepath: Original data file path
            
        Returns:
            True if successful
        """
        logger.info("Adding DVC metadata files to git...")
        
        files_to_add = [
            f"{filepath}.dvc",
            ".gitignore"  # DVC updates .gitignore to exclude the data file
        ]
        
        success_count = 0
        for file in files_to_add:
            if os.path.exists(file):
                success, stdout, stderr = self._run_command(['git', 'add', file])
                if success:
                    logger.info(f"✓ Added to git staging: {file}")
                    success_count += 1
                else:
                    logger.warning(f"Failed to add to git: {file} - {stderr}")
        
        return success_count > 0
    
    def git_commit(self, message: str) -> bool:
        """
        Commit DVC changes to git
        
        Args:
            message: Commit message
            
        Returns:
            True if successful
        """
        logger.info(f"Committing DVC metadata to git: {message}")
        
        success, stdout, stderr = self._run_command(['git', 'commit', '-m', message])
        
        if success or "nothing to commit" in stdout.lower() or "nothing to commit" in stderr.lower():
            logger.info("✓ Git commit successful")
            return True
        else:
            logger.error(f"Failed to commit: {stderr}")
            return False


def version_data_with_dvc(filepath: str,
                         remote_url: Optional[str] = None,
                         remote_name: str = "storage",
                         commit_message: Optional[str] = None) -> Dict:
    """
    Main DVC versioning function for Airflow task - Phase 1 Versioning component
    
    Args:
        filepath: Path to processed data file to version
        remote_url: DVC remote storage URL (can be local path for testing)
        remote_name: Name for the remote
        commit_message: Git commit message
        
    Returns:
        Dictionary with versioning status and metadata
    """
    logger.info("=" * 60)
    logger.info("STARTING DATA VERSIONING WITH DVC")
    logger.info("=" * 60)
    
    # Get repository root (assuming we're in the repo)
    repo_path = os.getcwd()
    dvc = DVCManager(repo_path=repo_path)
    
    try:
        # Initialize DVC if needed
        dvc.init_dvc()
        
        # Add remote storage if provided
        if remote_url:
            # For local testing, ensure remote directory exists
            if not remote_url.startswith(('s3://', 'gs://', 'azure://', 'http')):
                os.makedirs(remote_url, exist_ok=True)
            
            dvc.add_remote(remote_name, remote_url, default=True)
        
        # Add file to DVC tracking
        success = dvc.add_file(filepath)
        if not success:
            raise RuntimeError("Failed to add file to DVC tracking")
        
        # Push data to remote storage
        push_success = dvc.push(remote_name if remote_url else None)
        
        # Add DVC metadata files to git
        git_add_success = dvc.git_add_dvc_files(filepath)
        
        # Commit DVC metadata to git
        if commit_message is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            commit_message = f"Version processed dataset: {os.path.basename(filepath)} - {timestamp}"
        
        commit_success = dvc.git_commit(commit_message)
        
        # Create result summary
        result = {
            "success": success and push_success,
            "filepath": filepath,
            "dvc_file": f"{filepath}.dvc",
            "remote_name": remote_name,
            "remote_url": remote_url,
            "push_success": push_success,
            "git_add_success": git_add_success,
            "commit_success": commit_success,
            "commit_message": commit_message,
            "timestamp": datetime.now().isoformat(),
            "file_size_bytes": os.path.getsize(filepath) if os.path.exists(filepath) else 0
        }
        
        logger.info("=" * 60)
        if result["success"]:
            logger.info("✓ DATA VERSIONING COMPLETE")
            logger.info(f"  Data file: {filepath}")
            logger.info(f"  DVC metadata: {filepath}.dvc")
            logger.info(f"  Remote storage: {remote_url or 'default'}")
            logger.info(f"  Pushed to remote: {push_success}")
            logger.info(f"  Committed to git: {commit_success}")
        else:
            logger.error("✗ DATA VERSIONING FAILED")
            if not push_success:
                logger.error("  - Failed to push data to remote storage")
        logger.info("=" * 60)
        
        return result
        
    except Exception as e:
        logger.error(f"✗ DATA VERSIONING FAILED: {e}")
        result = {
            "success": False,
            "error": str(e),
            "filepath": filepath,
            "timestamp": datetime.now().isoformat()
        }
        return result


def setup_dvc_with_local_remote(data_dir: str = "data/processed",
                               remote_dir: str = "data/dvc_storage") -> Dict:
    """
    Setup DVC with local remote storage for testing/development
    
    Args:
        data_dir: Directory containing data files
        remote_dir: Directory to use as DVC remote storage
        
    Returns:
        Dictionary with setup status
    """
    logger.info("Setting up DVC with local remote storage...")
    
    # Create remote directory
    os.makedirs(remote_dir, exist_ok=True)
    
    # Initialize DVC
    dvc = DVCManager()
    dvc.init_dvc()
    
    # Add local remote
    remote_url = os.path.abspath(remote_dir)
    dvc.add_remote("local", remote_url, default=True)
    
    logger.info(f"✓ DVC configured with local remote: {remote_url}")
    
    return {
        "success": True,
        "remote_url": remote_url,
        "remote_name": "local",
        "remote_type": "local"
    }


if __name__ == "__main__":
    # Test DVC operations
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        
        # Setup local remote for testing
        setup_info = setup_dvc_with_local_remote()
        
        # Version the file
        result = version_data_with_dvc(
            filepath=filepath,
            remote_url=setup_info["remote_url"],
            remote_name=setup_info["remote_name"]
        )
        
        print(f"\n✓ DVC versioning result:")
        print(json.dumps(result, indent=2, default=str))
    else:
        print("Usage: python dvc_version.py <filepath>")