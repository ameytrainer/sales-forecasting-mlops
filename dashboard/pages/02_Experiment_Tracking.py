"""Experiment Tracking Page"""
import streamlit as st
import mlflow
import pandas as pd
import sys
sys.path.insert(0, '/home/ubuntu/sales-forecasting-mlops')

from src.config import get_settings

st.title("🔬 Experiment Tracking")

settings = get_settings()
mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

try:
    experiment = mlflow.get_experiment_by_name(settings.mlflow_experiment_name)
    
    if experiment:
        st.success(f"Experiment: {experiment.name}")
        st.write(f"Experiment ID: {experiment.experiment_id}")
        
        # Get recent runs
        runs = mlflow.search_runs(experiment.experiment_id, max_results=10)
        
        if not runs.empty:
            st.dataframe(runs[['start_time', 'tags.model_type', 'metrics.test_rmse', 
                              'metrics.test_mae']], use_container_width=True)
        else:
            st.info("No runs found")
    else:
        st.warning("Experiment not found")
except Exception as e:
    st.error(f"Error: {e}")
