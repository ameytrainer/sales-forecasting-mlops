"""
Model Registry Module

Manage MLflow model registry operations.

Author: Amey Talkatkar
"""

from typing import Optional, List, Dict
import mlflow
from mlflow.tracking import MlflowClient

from src.config import get_settings
from src.utils import setup_logging


logger = setup_logging(__name__)


class ModelRegistry:
    """
    Manage MLflow model registry.
    
    Examples:
        >>> registry = ModelRegistry()
        >>> registry.register_model(run_id, "sales_model")
        >>> registry.transition_model("sales_model", "1", "Production")
    """
    
    def __init__(self):
        """Initialize registry manager."""
        self.settings = get_settings()
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        self.settings.setup_mlflow_auth()
        self.client = MlflowClient()
        logger.info("ModelRegistry initialized")
    
    def register_model(
        self,
        run_id: str,
        model_name: str,
        model_path: str = "model"
    ) -> str:
        """
        Register model from run.
        
        Args:
            run_id: MLflow run ID
            model_name: Name for registered model
            model_path: Path within run
            
        Returns:
            Model version
        """
        logger.info(f"Registering model: {model_name} from run {run_id}")
        
        model_uri = f"runs:/{run_id}/{model_path}"
        
        result = mlflow.register_model(model_uri, model_name)
        version = result.version
        
        logger.info(f"Registered model version: {version}")
        return version
    
    def transition_model(
        self,
        model_name: str,
        version: str,
        stage: str
    ) -> None:
        """
        Transition model to stage.
        
        Args:
            model_name: Model name
            version: Model version
            stage: Target stage (Staging, Production, Archived)
        """
        logger.info(f"Transitioning {model_name} v{version} to {stage}")
        
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        
        logger.info(f"✅ Transition complete")
    
    def get_production_model(self, model_name: str) -> Optional[str]:
        """Get production model version."""
        versions = self.client.get_latest_versions(model_name, stages=["Production"])
        return versions[0].version if versions else None
    
    def archive_old_versions(
        self,
        model_name: str,
        keep_latest_n: int = 3
    ) -> None:
        """Archive old model versions."""
        logger.info(f"Archiving old versions of {model_name}")
        
        versions = self.client.search_model_versions(f"name='{model_name}'")
        versions = sorted(versions, key=lambda x: int(x.version), reverse=True)
        
        for version in versions[keep_latest_n:]:
            if version.current_stage not in ["Production", "Staging"]:
                self.transition_model(model_name, version.version, "Archived")
        
        logger.info(f"Archived {max(0, len(versions) - keep_latest_n)} versions")


if __name__ == "__main__":
    registry = ModelRegistry()
    logger.info("Model registry tests completed")
