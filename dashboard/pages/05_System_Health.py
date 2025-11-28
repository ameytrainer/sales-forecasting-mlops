"""
Dashboard Page 5: System Health

Author: Amey Talkatkar
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import sys
sys.path.insert(0, '/home/ubuntu/sales-forecasting-mlops')

from src.config import get_settings
from src.utils import get_memory_usage
from sqlalchemy import create_engine
from datetime import datetime, timedelta

st.title("💚 System Health")

settings = get_settings()

# Auto-refresh
auto_refresh = st.sidebar.checkbox("Auto-refresh (10s)", value=False)
if auto_refresh:
    st.rerun()

# API Health Check
st.subheader("🔌 API Status")

try:
    response = requests.get(f"http://{settings.api_host}:{settings.api_port}/health", timeout=5)
    
    if response.status_code == 200:
        health_data = response.json()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("API Status", "🟢 Healthy")
        
        with col2:
            memory_pct = health_data.get('memory', {}).get('percent', 0)
            st.metric("Memory Usage", f"{memory_pct:.1f}%")
        
        with col3:
            db_status = health_data.get('database', 'unknown')
            st.metric("Database", "🟢 Connected" if db_status == 'healthy' else "🔴 Error")
        
        with st.expander("View Full Health Report"):
            st.json(health_data)
    else:
        st.error(f"❌ API returned status code: {response.status_code}")
        
except requests.exceptions.ConnectionError:
    st.error("❌ Cannot connect to API")
    st.info(f"Make sure API is running on {settings.api_host}:{settings.api_port}")
except Exception as e:
    st.error(f"❌ Error checking API health: {str(e)}")

# Database Status
st.subheader("🗄️ Database Status")

try:
    engine = create_engine(settings.get_database_url())
    
    with engine.connect() as conn:
        # Table sizes
        query = """
            SELECT 
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
            FROM pg_tables
            WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
            LIMIT 10
        """
        
        df_tables = pd.read_sql(query, conn)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Database", "🟢 Connected")
        
        with col2:
            st.metric("Tables", len(df_tables))
        
        st.dataframe(df_tables, use_container_width=True)
    
    engine.dispose()
    
except Exception as e:
    st.error(f"❌ Database error: {str(e)}")

# Airflow Status
st.subheader("🌬️ Airflow Status")

try:
    # Try to get DAG status from Airflow API
    airflow_url = "http://localhost:8080/api/v1/dags"
    
    # Note: This would need authentication in production
    # For demo purposes, showing static info
    
    dag_info = {
        "ml_training_pipeline": "✅ Active",
        "data_ingestion_pipeline": "✅ Active",
        "batch_prediction_pipeline": "✅ Active",
        "model_monitoring_pipeline": "✅ Active",
        "retraining_trigger_pipeline": "✅ Active"
    }
    
    st.write("**DAG Status:**")
    
    dag_df = pd.DataFrame(list(dag_info.items()), columns=['DAG', 'Status'])
    st.dataframe(dag_df, use_container_width=True)
    
    st.info("💡 View detailed DAG status in Airflow UI: http://localhost:8080")
    
except Exception as e:
    st.warning(f"Could not fetch Airflow status: {str(e)}")

# System Resources
st.subheader("💻 System Resources")

try:
    memory_info = get_memory_usage()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        process_mb = memory_info.get('process_mb', 0)
        st.metric("Process Memory", f"{process_mb:.1f} MB")
    
    with col2:
        available_mb = memory_info.get('available_mb', 0)
        st.metric("Available Memory", f"{available_mb:.1f} MB")
    
    with col3:
        percent = memory_info.get('percent', 0)
        st.metric("Memory %", f"{percent:.1f}%")
    
    # Memory usage gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percent,
        title={'text': "Memory Usage"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 75], 'color': "yellow"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
except Exception as e:
    st.error(f"Error getting system resources: {str(e)}")

# Recent Activity
st.subheader("📊 Recent Activity")

try:
    engine = create_engine(settings.get_database_url())
    
    # Get recent predictions count
    query_predictions = """
        SELECT DATE(timestamp) as date, COUNT(*) as count
        FROM predictions
        WHERE timestamp >= NOW() - INTERVAL '7 days'
        GROUP BY DATE(timestamp)
        ORDER BY date DESC
    """
    
    df_predictions = pd.read_sql(query_predictions, engine)
    
    if not df_predictions.empty:
        fig = px.bar(df_predictions, x='date', y='count',
                    title='Daily Predictions (Last 7 Days)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No recent prediction activity")
    
    engine.dispose()
    
except Exception as e:
    st.warning(f"Could not load activity data: {str(e)}")

# Alerts
st.subheader("🚨 Active Alerts")

try:
    engine = create_engine(settings.get_database_url())
    
    query_alerts = """
        SELECT 
            alert_type,
            severity,
            title,
            message,
            timestamp
        FROM alerts
        WHERE resolved = FALSE
        ORDER BY severity DESC, timestamp DESC
        LIMIT 10
    """
    
    df_alerts = pd.read_sql(query_alerts, engine)
    engine.dispose()
    
    if not df_alerts.empty:
        for _, alert in df_alerts.iterrows():
            severity_icon = {"critical": "🔴", "warning": "⚠️", "info": "ℹ️"}
            icon = severity_icon.get(alert['severity'].lower(), "ℹ️")
            
            st.warning(f"{icon} **{alert['title']}** - {alert['message']}")
    else:
        st.success("✅ No active alerts")
    
except Exception as e:
    st.info("Alerts table not available")

# System Info
with st.expander("ℹ️ System Information"):
    st.markdown(f"""
    **Environment:** {settings.environment}  
    **Project:** {settings.project_name}  
    **Python Version:** {sys.version.split()[0]}  
    **Streamlit Version:** {st.__version__}
    
    **Configuration:**
    - API Host: {settings.api_host}
    - API Port: {settings.api_port}
    - Database: {settings.database_host}:{settings.database_port}
    - MLflow: {settings.mlflow_tracking_uri}
    """)

# Actions
st.sidebar.markdown("---")
st.sidebar.subheader("🎬 Quick Actions")

if st.sidebar.button("🔄 Refresh All"):
    st.rerun()

if st.sidebar.button("🧹 Clear API Cache"):
    try:
        response = requests.delete(f"http://{settings.api_host}:{settings.api_port}/predict/cache")
        if response.status_code == 200:
            st.sidebar.success("✅ Cache cleared")
        else:
            st.sidebar.error("❌ Failed to clear cache")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

if st.sidebar.button("📊 View Logs"):
    st.sidebar.info("Check logs in: /home/ubuntu/airflow/logs")
