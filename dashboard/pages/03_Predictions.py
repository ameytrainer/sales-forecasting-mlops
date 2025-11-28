"""
Dashboard Page 3: Predictions Results

Author: Amey Talkatkar
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import sys
sys.path.insert(0, '/home/ubuntu/sales-forecasting-mlops')

from src.config import get_settings
from sqlalchemy import create_engine

st.title("🔮 Prediction Results")

settings = get_settings()

try:
    # Load predictions from database
    engine = create_engine(settings.get_database_url())
    
    query = """
        SELECT 
            timestamp,
            date,
            region,
            product,
            price,
            quantity,
            predicted_sales,
            model_version,
            actual_sales,
            absolute_error
        FROM predictions
        ORDER BY timestamp DESC
        LIMIT 1000
    """
    
    df = pd.read_sql(query, engine)
    engine.dispose()
    
    if not df.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", len(df))
        
        with col2:
            avg_pred = df['predicted_sales'].mean()
            st.metric("Avg Prediction", f"${avg_pred:,.2f}")
        
        with col3:
            if 'absolute_error' in df.columns and df['absolute_error'].notna().any():
                avg_error = df['absolute_error'].mean()
                st.metric("Avg Error", f"${avg_error:,.2f}")
            else:
                st.metric("Avg Error", "N/A")
        
        with col4:
            unique_regions = df['region'].nunique()
            st.metric("Regions", unique_regions)
        
        # Recent predictions table
        st.subheader("📋 Recent Predictions")
        
        display_cols = ['timestamp', 'date', 'region', 'product', 
                       'predicted_sales', 'model_version']
        st.dataframe(df[display_cols].head(50), use_container_width=True)
        
        # Predictions over time
        st.subheader("📈 Predictions Over Time")
        
        df_time = df.copy()
        df_time['timestamp'] = pd.to_datetime(df_time['timestamp'])
        df_time = df_time.sort_values('timestamp')
        
        fig = px.line(df_time, x='timestamp', y='predicted_sales',
                     title='Predicted Sales Over Time')
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution by region
        st.subheader("🗺️ Predictions by Region")
        
        region_stats = df.groupby('region')['predicted_sales'].agg(['count', 'mean', 'sum'])
        region_stats.columns = ['Count', 'Average', 'Total']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(region_stats, use_container_width=True)
        
        with col2:
            fig = px.pie(df, names='region', values='predicted_sales',
                        title='Sales Distribution by Region')
            st.plotly_chart(fig, use_container_width=True)
        
        # Error analysis (if actual values available)
        if 'actual_sales' in df.columns and df['actual_sales'].notna().any():
            st.subheader("📊 Error Analysis")
            
            df_with_actuals = df[df['actual_sales'].notna()].copy()
            
            if not df_with_actuals.empty:
                fig = px.scatter(df_with_actuals, 
                               x='actual_sales', 
                               y='predicted_sales',
                               title='Predicted vs Actual Sales',
                               labels={'actual_sales': 'Actual', 'predicted_sales': 'Predicted'})
                
                # Add perfect prediction line
                max_val = max(df_with_actuals['actual_sales'].max(), 
                            df_with_actuals['predicted_sales'].max())
                fig.add_shape(type='line',
                            x0=0, y0=0, x1=max_val, y1=max_val,
                            line=dict(color='red', dash='dash'))
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Download data
        st.subheader("💾 Export Data")
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name=f"predictions_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
    else:
        st.info("No predictions found in database")
        st.markdown("""
        **To generate predictions:**
        1. Trigger the batch_prediction_pipeline in Airflow
        2. Or use the API endpoint: `POST /predict/batch`
        """)

except Exception as e:
    st.error(f"Error loading predictions: {str(e)}")
    st.info("Make sure the database is running and predictions table exists")
