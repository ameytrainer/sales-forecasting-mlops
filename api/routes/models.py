"""
Model Management API Routes

Endpoints for MLflow model registry operations.

Author: Amey Talkatkar
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime
import logging
import sys

sys.path.insert(0, '/home/ubuntu/sales-forecasting-mlops')

from api.models import ModelInfo, ModelList, ModelTransition
from src.models import ModelRegistry
from src.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_model=ModelList)
async def list_models():
    """List all registered models."""
    try:
        registry = ModelRegistry()
        
        # Get all registered models (using MLflow client)
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        registered_models = client.search_registered_models()
        
        models = []
        for rm in registered_models:
            for version in rm.latest_versions:
                models.append(ModelInfo(
                    name=rm.name,
                    version=version.version,
                    stage=version.current_stage,
                    created_at=datetime.fromtimestamp(version.creation_timestamp / 1000),
                    tags=version.tags
                ))
        
        return ModelList(
            models=models,
            count=len(models),
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{name}/latest", response_model=ModelInfo)
async def get_latest_model(name: str, stage: str = "Production"):
    """Get latest model version for specified stage."""
    try:
        registry = ModelRegistry()
        
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        versions = client.get_latest_versions(name, stages=[stage])
        
        if not versions:
            raise HTTPException(status_code=404, detail=f"No {stage} model found")
        
        version = versions[0]
        
        return ModelInfo(
            name=name,
            version=version.version,
            stage=version.current_stage,
            created_at=datetime.fromtimestamp(version.creation_timestamp / 1000),
            tags=version.tags
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{name}/promote")
async def promote_model(name: str, transition: ModelTransition):
    """Promote model to specified stage."""
    try:
        registry = ModelRegistry()
        
        registry.transition_model(name, int(transition.version), transition.stage)
        
        return {
            "status": "success",
            "message": f"Model {name} v{transition.version} promoted to {transition.stage}",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Failed to promote model: {e}")
        raise HTTPException(status_code=500, detail=str(e))
