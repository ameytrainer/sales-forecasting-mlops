"""
Model Service - Business Logic for Model Operations

Handles model loading, caching, and registry operations.

Author: Amey Talkatkar
"""

import logging
from typing import Optional, Dict, Any
import mlflow
from mlflow.tracking import MlflowClient
import sys

sys.path.insert(0, '/home/ubuntu/sales-forecasting-mlops')

from src.config import get_settings
from src.models import ModelRegistry

logger = logging.getLogger(__name__)


class ModelService:
    """
    Service for managing ML models.
    
    Features:
    - Model loading from MLflow
    - Model caching for performance
    - Registry operations
    - Version management
    """
    
    def __init__(self):
        """Initialize model service."""
        self.settings = get_settings()
        self.registry = ModelRegistry()
        self.model_cache: Dict[str, Any] = {}
        self.client = MlflowClient()
        
        logger.info("ModelService initialized")
    
    def load_model_from_registry(
        self, 
        model_name: str, 
        stage: str = "Production",
        version: Optional[int] = None
    ) -> Any:
        """
        Load model from MLflow registry.
        
        Args:
            model_name: Name of registered model
            stage: Model stage (Production, Staging, Archived)
            version: Specific version number (optional)
            
        Returns:
            Loaded model object
            
        Example:
            >>> service = ModelService()
            >>> model = service.load_model_from_registry("sales_forecasting", "Production")
        """
        cache_key = f"{model_name}_{stage}_{version}"
        
        # Check cache first
        if cache_key in self.model_cache:
            logger.info(f"Loading model from cache: {cache_key}")
            return self.model_cache[cache_key]
        
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            else:
                model_uri = f"models:/{model_name}/{stage}"
            
            logger.info(f"Loading model from MLflow: {model_uri}")
            model = mlflow.pyfunc.load_model(model_uri)
            
            # Cache the model
            self.model_cache[cache_key] = model
            logger.info(f"Model loaded and cached: {cache_key}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def get_production_model(self, model_name: str) -> Any:
        """
        Get current production model.
        
        Args:
            model_name: Name of registered model
            
        Returns:
            Production model object
        """
        return self.load_model_from_registry(model_name, stage="Production")
    
    def get_model_info(
        self, 
        model_name: str, 
        stage: Optional[str] = None,
        version: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get model metadata from registry.
        
        Args:
            model_name: Name of registered model
            stage: Filter by stage (optional)
            version: Specific version (optional)
            
        Returns:
            Model metadata dictionary
            
        Example:
            >>> info = service.get_model_info("sales_forecasting", stage="Production")
            >>> print(info['version'], info['metrics'])
        """
        try:
            if version:
                model_version = self.client.get_model_version(model_name, str(version))
                versions = [model_version]
            elif stage:
                versions = self.client.get_latest_versions(model_name, stages=[stage])
            else:
                # Get all versions
                versions = self.client.search_model_versions(f"name='{model_name}'")
            
            if not versions:
                raise ValueError(f"No model found for {model_name}")
            
            # Return info for latest/specified version
            mv = versions[0]
            
            # Get run details for metrics
            run = self.client.get_run(mv.run_id)
            
            return {
                'name': model_name,
                'version': mv.version,
                'stage': mv.current_stage,
                'run_id': mv.run_id,
                'creation_timestamp': mv.creation_timestamp,
                'last_updated_timestamp': mv.last_updated_timestamp,
                'description': mv.description,
                'tags': mv.tags,
                'metrics': run.data.metrics,
                'params': run.data.params,
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            raise
    
    def list_all_models(self) -> list:
        """
        List all registered models.
        
        Returns:
            List of model information dictionaries
        """
        try:
            registered_models = self.client.search_registered_models()
            
            models_info = []
            for rm in registered_models:
                for version in rm.latest_versions:
                    try:
                        info = self.get_model_info(rm.name, version=int(version.version))
                        models_info.append(info)
                    except Exception as e:
                        logger.warning(f"Failed to get info for {rm.name} v{version.version}: {e}")
                        continue
            
            return models_info
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise
    
    def promote_model(
        self, 
        model_name: str, 
        version: int, 
        stage: str
    ) -> Dict[str, Any]:
        """
        Promote model to specified stage.
        
        Args:
            model_name: Name of model
            version: Version number
            stage: Target stage (Staging, Production, Archived)
            
        Returns:
            Promotion result dictionary
        """
        try:
            logger.info(f"Promoting {model_name} v{version} to {stage}")
            
            # Validate stage
            valid_stages = ['Staging', 'Production', 'Archived']
            if stage not in valid_stages:
                raise ValueError(f"Stage must be one of {valid_stages}")
            
            # Use registry to transition
            self.registry.transition_model(model_name, version, stage)
            
            # Clear cache since model changed
            self.clear_cache()
            
            logger.info(f"✅ Model promoted successfully")
            
            return {
                'status': 'success',
                'model_name': model_name,
                'version': version,
                'stage': stage,
                'message': f'Model promoted to {stage}'
            }
            
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            raise
    
    def archive_old_versions(
        self, 
        model_name: str, 
        keep_latest_n: int = 3
    ) -> Dict[str, Any]:
        """
        Archive old model versions.
        
        Args:
            model_name: Name of model
            keep_latest_n: Number of latest versions to keep
            
        Returns:
            Archive result dictionary
        """
        try:
            logger.info(f"Archiving old versions of {model_name}, keeping {keep_latest_n}")
            
            archived_count = self.registry.archive_old_versions(model_name, keep_latest_n)
            
            return {
                'status': 'success',
                'model_name': model_name,
                'archived_count': archived_count,
                'message': f'Archived {archived_count} old versions'
            }
            
        except Exception as e:
            logger.error(f"Failed to archive versions: {e}")
            raise
    
    def clear_cache(self):
        """Clear model cache."""
        logger.info("Clearing model cache")
        self.model_cache.clear()
    
    def get_model_metrics(
        self, 
        model_name: str, 
        stage: str = "Production"
    ) -> Dict[str, float]:
        """
        Get metrics for specified model.
        
        Args:
            model_name: Name of model
            stage: Model stage
            
        Returns:
            Dictionary of metrics
        """
        try:
            info = self.get_model_info(model_name, stage=stage)
            return info.get('metrics', {})
            
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {}
    
    def compare_models(
        self, 
        model_name: str, 
        stages: list = ['Staging', 'Production']
    ) -> Dict[str, Any]:
        """
        Compare models across stages.
        
        Args:
            model_name: Name of model
            stages: List of stages to compare
            
        Returns:
            Comparison dictionary
        """
        try:
            comparison = {}
            
            for stage in stages:
                try:
                    info = self.get_model_info(model_name, stage=stage)
                    comparison[stage] = {
                        'version': info['version'],
                        'metrics': info['metrics'],
                        'updated': info['last_updated_timestamp']
                    }
                except Exception as e:
                    logger.warning(f"No model in {stage}: {e}")
                    comparison[stage] = None
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            raise
