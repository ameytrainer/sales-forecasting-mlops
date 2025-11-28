"""
Streamlit Dashboard - Main App

Author: Amey Talkatkar
"""
import streamlit as st
import sys
sys.path.insert(0, '/home/ubuntu/sales-forecasting-mlops')

from src.config import get_settings

st.set_page_config(
    page_title="MLOps Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("🚀 Sales Forecasting MLOps Dashboard")
st.markdown("Real-time monitoring and visualization for ML pipelines")

# Sidebar
st.sidebar.title("Navigation")
st.sidebar.info("""
**Available Pages:**
- Model Comparison
- Experiment Tracking  
- Predictions
- Data Drift
- System Health
""")

# Main page content
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Active Models", "3", "+1")

with col2:
    st.metric("Predictions Today", "1,234", "+15%")

with col3:
    st.metric("Model Accuracy", "87.5%", "+2.1%")

st.info("👈 Select a page from the sidebar to get started!")
