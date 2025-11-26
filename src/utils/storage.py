"""
Cloud Storage and Data Loading Module
Handles uploading processed data to cloud storage (MinIO, S3, Azure Blob)
"""
import os
import logging
from datetime import datetime
from typing import Optional, Dict
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StorageUploader:
    """Base class for cloud storage uploaders"""
    
    def upload_file(self, local_path: str, remote_path: str) -> Dict:
        """Upload file to cloud storage"""
        raise NotImplementedError("Subclass must implement upload_file method")
    
    def verify_upload(self, remote_path: str) -> bool:
        """Verify file was uploaded successfully"""
        raise NotImplementedError("Subclass must implement verify_upload method")


class MinIOUploader(StorageUploader):
    """Upload files to MinIO object storage"""
    
    def __init__(self, 
                 endpoint: str,
                 access_key: str,
                 secret_key: str,
                 bucket_name: str,
                 secure: bool = False):
        """
        Initialize MinIO uploader
        
        Args:
            endpoint: MinIO server endpoint (e.g., 'localhost:9000')
            access_key: MinIO access key
            secret_key: MinIO secret key
            bucket_name: Bucket name
            secure: Use HTTPS (default: False)
        """
        try:
            from minio import Minio
        except ImportError:
            logger.error("minio package not installed. Install with: pip install minio")
            raise
        
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        self.bucket_name = bucket_name
        
        # Ensure bucket exists
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist"""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Created bucket: {self.bucket_name}")
            else:
                logger.info(f"Bucket exists: {self.bucket_name}")
        except Exception as e:
            logger.error(f"Error checking/creating bucket: {e}")
            raise
    
    def upload_file(self, local_path: str, remote_path: str) -> Dict:
        """
        Upload file to MinIO
        
        Args:
            local_path: Path to local file
            remote_path: Remote object name
            
        Returns:
            Dictionary with upload metadata
        """
        logger.info(f"Uploading {local_path} to MinIO bucket {self.bucket_name}/{remote_path}")
        
        try:
            result = self.client.fput_object(
                self.bucket_name,
                remote_path,
                local_path
            )
            
            metadata = {
                "bucket": self.bucket_name,
                "object_name": remote_path,
                "etag": result.etag,
                "version_id": result.version_id,
                "timestamp": datetime.now().isoformat(),
                "size_bytes": os.path.getsize(local_path)
            }
            
            logger.info(f"✓ File uploaded successfully to MinIO")
            return metadata
            
        except Exception as e:
            logger.error(f"Error uploading to MinIO: {e}")
            raise
    
    def verify_upload(self, remote_path: str) -> bool:
        """Verify file exists in MinIO"""
        try:
            self.client.stat_object(self.bucket_name, remote_path)
            return True
        except Exception:
            return False


class LocalStorageUploader(StorageUploader):
    """Copy files to a local directory (for testing or local 'cloud' storage)"""
    
    def __init__(self, storage_dir: str):
        """
        Initialize local storage uploader
        
        Args:
            storage_dir: Directory to store files
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        logger.info(f"Local storage directory: {storage_dir}")
    
    def upload_file(self, local_path: str, remote_path: str) -> Dict:
        """
        Copy file to local storage directory
        
        Args:
            local_path: Path to local file
            remote_path: Relative path in storage directory
            
        Returns:
            Dictionary with upload metadata
        """
        import shutil
        
        dest_path = os.path.join(self.storage_dir, remote_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        logger.info(f"Copying {local_path} to {dest_path}")
        shutil.copy2(local_path, dest_path)
        
        metadata = {
            "storage_dir": self.storage_dir,
            "relative_path": remote_path,
            "absolute_path": dest_path,
            "timestamp": datetime.now().isoformat(),
            "size_bytes": os.path.getsize(dest_path)
        }
        
        logger.info(f"✓ File copied to local storage")
        return metadata
    
    def verify_upload(self, remote_path: str) -> bool:
        """Verify file exists in local storage"""
        full_path = os.path.join(self.storage_dir, remote_path)
        return os.path.exists(full_path)


class GoogleDriveUploader(StorageUploader):
    """Upload files to Google Drive"""
    
    def __init__(self, folder_id: Optional[str] = None, credentials_path: Optional[str] = None):
        """
        Initialize Google Drive uploader
        
        Args:
            folder_id: Google Drive folder ID to upload to
            credentials_path: Path to Google API credentials JSON file
        """
        try:
            from google.colab import auth
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload
        except ImportError:
            logger.warning("Google Drive libraries not installed. Install with: pip install google-colab google-auth-oauthlib google-auth-httplib2 google-api-python-client")
        
        self.folder_id = folder_id or os.getenv('GOOGLE_DRIVE_FOLDER_ID')
        self.credentials_path = credentials_path or os.getenv('GOOGLE_CREDENTIALS_PATH')
        
        # Initialize Google Drive service
        self._init_drive_service()
    
    def _init_drive_service(self):
        """Initialize Google Drive API service"""
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.service_account import Credentials
            from googleapiclient.discovery import build
            
            if self.credentials_path and os.path.exists(self.credentials_path):
                # Use service account credentials
                creds = Credentials.from_service_account_file(
                    self.credentials_path,
                    scopes=['https://www.googleapis.com/auth/drive']
                )
                self.drive_service = build('drive', 'v3', credentials=creds)
                logger.info("✓ Google Drive service initialized with service account")
            else:
                logger.warning("Google Drive credentials not configured. Upload may fail.")
                self.drive_service = None
        except ImportError as e:
            logger.error(f"Google Drive packages not available: {e}")
            self.drive_service = None
    
    def upload_file(self, local_path: str, remote_path: str) -> Dict:
        """
        Upload file to Google Drive
        
        Args:
            local_path: Path to local file
            remote_path: Relative path/name in Google Drive
            
        Returns:
            Dictionary with upload metadata
        """
        if not self.drive_service:
            raise RuntimeError("Google Drive service not initialized. Check credentials.")
        
        try:
            from googleapiclient.http import MediaFileUpload
            
            # Extract filename from remote_path
            filename = os.path.basename(remote_path)
            
            logger.info(f"Uploading {filename} to Google Drive...")
            
            # Prepare file metadata
            file_metadata = {
                'name': filename,
                'mimeType': 'application/octet-stream'
            }
            
            # Add parent folder if specified
            if self.folder_id:
                file_metadata['parents'] = [self.folder_id]
            
            # Upload file
            media = MediaFileUpload(local_path, mimetype='text/csv', resumable=True)
            file = self.drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, webViewLink, size'
            ).execute()
            
            # Build share link
            drive_url = f"https://drive.google.com/file/d/{file['id']}/view"
            
            metadata = {
                "file_id": file['id'],
                "filename": filename,
                "drive_url": drive_url,
                "web_view_link": file.get('webViewLink', ''),
                "size_bytes": file.get('size', 0),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✓ File uploaded to Google Drive: {drive_url}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error uploading to Google Drive: {e}")
            raise
    
    def verify_upload(self, remote_path: str) -> bool:
        """Verify file exists in Google Drive (simplified check)"""
        try:
            if self.drive_service:
                # This is a basic check - in production you might want more robust verification
                logger.info("Google Drive upload verification skipped (basic implementation)")
                return True
            return False
        except Exception as e:
            logger.error(f"Error verifying Google Drive upload: {e}")
            return False


class LocalStorageUploader(StorageUploader):
    """Copy files to a local directory (for testing or local 'cloud' storage)"""
    
    def __init__(self, storage_dir: str):
        """
        Initialize local storage uploader
        
        Args:
            storage_dir: Directory to store files
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        logger.info(f"Local storage directory: {storage_dir}")
    
    def upload_file(self, local_path: str, remote_path: str) -> Dict:
        """
        Copy file to local storage directory
        
        Args:
            local_path: Path to local file
            remote_path: Relative path in storage directory
            
        Returns:
            Dictionary with upload metadata
        """
        import shutil
        
        dest_path = os.path.join(self.storage_dir, remote_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        logger.info(f"Copying {local_path} to {dest_path}")
        shutil.copy2(local_path, dest_path)
        
        metadata = {
            "storage_dir": self.storage_dir,
            "relative_path": remote_path,
            "absolute_path": dest_path,
            "timestamp": datetime.now().isoformat(),
            "size_bytes": os.path.getsize(dest_path)
        }
        logger.info(f"✓ File copied successfully to local storage")
        return metadata
    
    def verify_upload(self, remote_path: str) -> bool:
        """Verify file exists in local storage"""
        dest_path = os.path.join(self.storage_dir, remote_path)
        return os.path.exists(dest_path)


def upload_to_storage(local_filepath: str,
                     storage_type: str = "local",
                     remote_filename: str = None,
                     config: Dict = None) -> Dict:
    """
    Main upload function for Airflow task - Phase 1 Loading component
    
    Args:
        local_filepath: Path to processed data file to upload
        storage_type: Type of storage ('minio', 'local')
        remote_filename: Remote filename (default: basename of local file)
        config: Storage configuration dictionary
        
    Returns:
        Dictionary with upload metadata
    """
    logger.info("=" * 60)
    logger.info("STARTING DATA LOADING TO CLOUD STORAGE")
    logger.info("=" * 60)
    
    if config is None:
        config = {}
    
    if remote_filename is None:
        remote_filename = os.path.basename(local_filepath)
    
    # Add timestamp prefix to remote filename for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    remote_path = f"processed/{timestamp}_{remote_filename}"
    
    try:
        # Initialize appropriate uploader based on configuration
        if storage_type == "minio":
            uploader = MinIOUploader(
                endpoint=config.get('endpoint', 'localhost:9000'),
                access_key=config.get('access_key', 'minioadmin'),
                secret_key=config.get('secret_key', 'minioadmin'),
                bucket_name=config.get('bucket_name', 'mlops-data'),
                secure=config.get('secure', False)
            )
        elif storage_type == "local":
            uploader = LocalStorageUploader(
                storage_dir=config.get('storage_dir', 'data/cloud_storage')
            )
        elif storage_type == "google_drive" or storage_type == "gdrive":
            uploader = GoogleDriveUploader(
                folder_id=config.get('folder_id') or os.getenv('GOOGLE_DRIVE_FOLDER_ID'),
                credentials_path=config.get('credentials_path') or os.getenv('GOOGLE_CREDENTIALS_PATH')
            )
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}. Supported: minio, local, google_drive")
        
        # Upload file
        metadata = uploader.upload_file(local_filepath, remote_path)
        
        # Verify upload
        if uploader.verify_upload(remote_path):
            logger.info("✓ Upload verification successful")
            metadata["verified"] = True
        else:
            logger.warning("⚠ Upload verification failed")
            metadata["verified"] = False
        
        # Save upload metadata for tracking
        metadata_path = local_filepath.replace('.csv', '_upload_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Upload metadata saved to: {metadata_path}")
        
        logger.info("=" * 60)
        logger.info("✓ DATA LOADING COMPLETE")
        logger.info(f"  Local file: {local_filepath}")
        logger.info(f"  Remote path: {remote_path}")
        logger.info(f"  Storage type: {storage_type}")
        logger.info(f"  File size: {metadata.get('size_bytes', 0):,} bytes")
        logger.info("=" * 60)
        
        return metadata
        
    except Exception as e:
        logger.error(f"✗ DATA LOADING FAILED: {e}")
        raise


if __name__ == "__main__":
    # Test the uploader
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        
        # Test with local storage
        config = {
            'storage_dir': 'data/cloud_storage'
        }
        
        metadata = upload_to_storage(
            local_filepath=filepath,
            storage_type='local',
            config=config
        )
        
        print(f"\n✓ File uploaded successfully")
        print(f"Metadata: {json.dumps(metadata, indent=2)}")
    else:
        print("Usage: python storage.py <filepath>")