import boto3
import joblib
import pickle
import tempfile
import os
from typing import Any, Optional, Dict
from pathlib import Path
import logging
from datetime import datetime

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Model registry for storing and loading ML models from S3/MinIO."""
    
    def __init__(self):
        self.bucket_name = settings.s3_bucket
        self.s3_client = self._create_s3_client()
    
    def _create_s3_client(self):
        """Create S3 client based on configuration."""
        client_kwargs = {
            'region_name': settings.s3_region,
        }
        
        # Use custom endpoint for MinIO
        if settings.s3_endpoint_url:
            client_kwargs['endpoint_url'] = settings.s3_endpoint_url
        
        # Use explicit credentials if provided
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            client_kwargs['aws_access_key_id'] = settings.aws_access_key_id
            client_kwargs['aws_secret_access_key'] = settings.aws_secret_access_key
        
        return boto3.client('s3', **client_kwargs)
    
    def save_model(
        self, 
        model: Any, 
        model_name: str, 
        version: Optional[str] = None,
        metadata: Optional[Dict] = None,
        serializer: str = 'joblib'
    ) -> str:
        """
        Save a model to the registry.
        
        Args:
            model: Model object to save
            model_name: Name of the model
            version: Model version (defaults to timestamp)
            metadata: Additional metadata to store
            serializer: Serialization method ('joblib' or 'pickle')
            
        Returns:
            S3 key of the saved model
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_key = f"models/{model_name}/{version}/model"
        metadata_key = f"models/{model_name}/{version}/metadata.json"
        
        try:
            # Serialize model to temporary file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                if serializer == 'joblib':
                    joblib.dump(model, tmp_file.name)
                    model_key += '.joblib'
                elif serializer == 'pickle':
                    with open(tmp_file.name, 'wb') as f:
                        pickle.dump(model, f)
                    model_key += '.pkl'
                else:
                    raise ValueError(f"Unsupported serializer: {serializer}")
                
                # Upload model file
                self.s3_client.upload_file(tmp_file.name, self.bucket_name, model_key)
                
                # Clean up temporary file
                os.unlink(tmp_file.name)
            
            # Save metadata if provided
            if metadata:
                metadata['saved_at'] = datetime.now().isoformat()
                metadata['serializer'] = serializer
                
                import json
                metadata_str = json.dumps(metadata, indent=2)
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=metadata_key,
                    Body=metadata_str.encode('utf-8'),
                    ContentType='application/json'
                )
            
            # Update latest pointer
            latest_key = f"models/{model_name}/latest"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=latest_key,
                Body=version.encode('utf-8'),
                ContentType='text/plain'
            )
            
            logger.info(f"Model saved successfully: {model_key}")
            return model_key
            
        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {e}")
            raise
    
    def load_model(self, model_name: str, version: str = 'latest') -> Any:
        """
        Load a model from the registry.
        
        Args:
            model_name: Name of the model
            version: Model version ('latest' or specific version)
            
        Returns:
            Loaded model object
        """
        try:
            # Resolve version if 'latest'
            if version == 'latest':
                latest_key = f"models/{model_name}/latest"
                response = self.s3_client.get_object(
                    Bucket=self.bucket_name, 
                    Key=latest_key
                )
                version = response['Body'].read().decode('utf-8').strip()
            
            # Try different file extensions
            for ext, loader in [('.joblib', joblib.load), ('.pkl', pickle.load)]:
                model_key = f"models/{model_name}/{version}/model{ext}"
                
                try:
                    # Download model to temporary file
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        self.s3_client.download_file(
                            self.bucket_name, 
                            model_key, 
                            tmp_file.name
                        )
                        
                        # Load model
                        if ext == '.joblib':
                            model = loader(tmp_file.name)
                        else:  # pickle
                            with open(tmp_file.name, 'rb') as f:
                                model = loader(f)
                        
                        # Clean up
                        os.unlink(tmp_file.name)
                        
                        logger.info(f"Model loaded successfully: {model_key}")
                        return model
                        
                except self.s3_client.exceptions.NoSuchKey:
                    continue
            
            raise FileNotFoundError(f"Model not found: {model_name} v{version}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def load_metadata(self, model_name: str, version: str = 'latest') -> Optional[Dict]:
        """
        Load model metadata.
        
        Args:
            model_name: Name of the model
            version: Model version ('latest' or specific version)
            
        Returns:
            Metadata dictionary or None if not found
        """
        try:
            # Resolve version if 'latest'
            if version == 'latest':
                latest_key = f"models/{model_name}/latest"
                response = self.s3_client.get_object(
                    Bucket=self.bucket_name, 
                    Key=latest_key
                )
                version = response['Body'].read().decode('utf-8').strip()
            
            metadata_key = f"models/{model_name}/{version}/metadata.json"
            response = self.s3_client.get_object(
                Bucket=self.bucket_name, 
                Key=metadata_key
            )
            
            import json
            metadata = json.loads(response['Body'].read().decode('utf-8'))
            return metadata
            
        except self.s3_client.exceptions.NoSuchKey:
            logger.warning(f"Metadata not found for {model_name} v{version}")
            return None
        except Exception as e:
            logger.error(f"Failed to load metadata for {model_name}: {e}")
            raise
    
    def list_models(self) -> Dict[str, list]:
        """
        List all available models and their versions.
        
        Returns:
            Dictionary mapping model names to lists of versions
        """
        try:
            models = {}
            
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix='models/',
                Delimiter='/'
            )
            
            for page in page_iterator:
                if 'CommonPrefixes' in page:
                    for prefix in page['CommonPrefixes']:
                        model_name = prefix['Prefix'].split('/')[1]
                        
                        # List versions for this model
                        version_paginator = self.s3_client.get_paginator('list_objects_v2')
                        version_iterator = version_paginator.paginate(
                            Bucket=self.bucket_name,
                            Prefix=f'models/{model_name}/',
                            Delimiter='/'
                        )
                        
                        versions = []
                        for version_page in version_iterator:
                            if 'CommonPrefixes' in version_page:
                                for version_prefix in version_page['CommonPrefixes']:
                                    version = version_prefix['Prefix'].split('/')[2]
                                    if version != 'latest':  # Skip latest pointer
                                        versions.append(version)
                        
                        models[model_name] = sorted(versions, reverse=True)
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise


# Global model registry instance
model_registry = ModelRegistry()