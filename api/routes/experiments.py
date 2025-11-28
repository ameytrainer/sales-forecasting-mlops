"""Experiment tracking routes"""
from fastapi import APIRouter
import mlflow
router = APIRouter()

@router.get("/")
async def list_experiments():
    """List all experiments."""
    experiments = mlflow.search_experiments()
    return {"experiments": [e.to_dictionary() for e in experiments]}
