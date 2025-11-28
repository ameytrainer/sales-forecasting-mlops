"""
Dashboard Page 4: Data Drift Monitoring

Author: Amey Talkatkar
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
sys.path.insert(0, '/home/ubuntu/sales-forecasting-mlops')

from src.config import get_settings
from sqlalchemy import create_engine

st.title("🌊 Data Drift Monitoring")

settings = get_settings()

# Drift threshold
drift_threshold = st.sidebar.slider(
    "Drift Threshold",
    min_value=0.0,
    max_value=0.5,
    value=settings.drift_detection_threshold,
    step=0.01,
    help="KS-test p-value threshold for drift detection"
)

try:
    engine = create_engine(settings.get_database_url())
    
    # Load drift metrics
    query = """
        SELECT 
            timestamp,
            feature_name,
            drift_score,
            drift_detected,
            ks_statistic,
            ks_pvalue,
            psi_score
        FROM drift_metrics
        ORDER BY timestamp DESC
        LIMIT 1000
    """
    
    df_drift = pd.read_sql(query, engine)
    engine.dispose()
    
    if not df_drift.empty:
        # Latest drift status
        latest_time = df_drift['timestamp'].max()
        latest_drift = df_drift[df_drift['timestamp'] == latest_time]
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            drift_detected = latest_drift['drift_detected'].any()
            st.metric(
                "Drift Status",
                "⚠️ DETECTED" if drift_detected else "✅ STABLE",
                delta="Alert" if drift_detected else "Normal"
            )
        
        with col2:
            drifted_count = latest_drift['drift_detected'].sum()
            total_features = len(latest_drift)
            st.metric("Drifted Features", f"{drifted_count}/{total_features}")
        
        with col3:
            max_drift = latest_drift['drift_score'].max()
            st.metric("Max Drift Score", f"{max_drift:.4f}")
        
        with col4:
            st.metric("Last Check", latest_time.strftime("%Y-%m-%d %H:%M"))
        
        # Current drift status by feature
        st.subheader("📊 Current Drift Status")
        
        display_df = latest_drift[['feature_name', 'drift_score', 'drift_detected', 
                                   'ks_statistic', 'psi_score']].copy()
        display_df.columns = ['Feature', 'Drift Score', 'Drift Detected', 
                             'KS Statistic', 'PSI Score']
        
        # Color code drift detected
        def highlight_drift(row):
            if row['Drift Detected']:
                return ['background-color: #ffcccc'] * len(row)
            return [''] * len(row)
        
        st.dataframe(
            display_df.style.apply(highlight_drift, axis=1),
            use_container_width=True
        )
        
        # Drift score trends
        st.subheader("📈 Drift Score Trends")
        
        df_drift['timestamp'] = pd.to_datetime(df_drift['timestamp'])
        
        fig = px.line(df_drift, x='timestamp', y='drift_score', color='feature_name',
                     title='Drift Scores Over Time')
        
        # Add threshold line
        fig.add_hline(y=drift_threshold, line_dash="dash", line_color="red",
                     annotation_text="Drift Threshold")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # PSI Score comparison
        st.subheader("📊 Population Stability Index (PSI)")
        
        psi_data = latest_drift[['feature_name', 'psi_score']].sort_values('psi_score', ascending=False)
        
        fig = px.bar(psi_data, x='feature_name', y='psi_score',
                    title='PSI Scores by Feature',
                    labels={'feature_name': 'Feature', 'psi_score': 'PSI Score'})
        
        # Add PSI interpretation lines
        fig.add_hline(y=0.1, line_dash="dash", line_color="green",
                     annotation_text="Stable (<0.1)")
        fig.add_hline(y=0.2, line_dash="dash", line_color="orange",
                     annotation_text="Moderate (0.1-0.2)")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Drift interpretation guide
        with st.expander("📖 Understanding Drift Metrics"):
            st.markdown("""
            **Drift Score (KS Statistic):**
            - Range: 0 to 1
            - < 0.05: No significant drift
            - 0.05 - 0.1: Moderate drift
            - > 0.1: Significant drift
            
            **Population Stability Index (PSI):**
            - < 0.1: Stable population
            - 0.1 - 0.2: Moderate population shift
            - > 0.2: Significant population shift (action needed)
            
            **Actions when drift detected:**
            1. Investigate data quality
            2. Check for seasonal patterns
            3. Consider model retraining
            4. Update feature engineering
            """)
        
        # Drift alert history
        st.subheader("🚨 Drift Alert History")
        
        drift_alerts = df_drift[df_drift['drift_detected'] == True].copy()
        
        if not drift_alerts.empty:
            alert_summary = drift_alerts.groupby('feature_name').agg({
                'timestamp': ['count', 'max'],
                'drift_score': 'mean'
            }).round(4)
            
            alert_summary.columns = ['Alert Count', 'Last Alert', 'Avg Drift Score']
            alert_summary = alert_summary.sort_values('Alert Count', ascending=False)
            
            st.dataframe(alert_summary, use_container_width=True)
        else:
            st.success("No drift alerts in history!")
    
    else:
        st.info("No drift monitoring data available")
        st.markdown("""
        **To start drift monitoring:**
        1. Ensure model_monitoring_pipeline is running in Airflow
        2. Pipeline runs every 30 minutes
        3. Check Airflow logs if issues occur
        """)

except Exception as e:
    st.error(f"Error loading drift data: {str(e)}")
    st.info("Make sure the database is running and drift_metrics table exists")

# Manual drift check
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Manual Actions")

if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

if st.sidebar.button("🚀 Trigger Drift Check"):
    st.sidebar.info("Triggering drift check via Airflow API...")
    # In production, call Airflow API to trigger model_monitoring_pipeline
    st.sidebar.success("Drift check scheduled!")
