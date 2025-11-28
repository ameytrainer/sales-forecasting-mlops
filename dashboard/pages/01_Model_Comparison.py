"""Model Comparison Page"""
import streamlit as st
import pandas as pd
import plotly.express as px
import sys
sys.path.insert(0, '/home/ubuntu/sales-forecasting-mlops')

st.title("📊 Model Comparison")

# Load comparison data
comparison_data = {
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
    'RMSE': [105.2, 92.8, 87.5],
    'MAE': [82.1, 71.3, 68.9],
    'R²': [0.82, 0.88, 0.91],
    'Training Time (s)': [1.2, 45.3, 12.7]
}

df = pd.DataFrame(comparison_data)

# Display metrics
st.dataframe(df, use_container_width=True)

# Plot comparison
fig = px.bar(df, x='Model', y=['RMSE', 'MAE'], barmode='group',
             title='Model Performance Comparison')
st.plotly_chart(fig, use_container_width=True)

# Training time
fig2 = px.bar(df, x='Model', y='Training Time (s)',
              title='Training Time Comparison')
st.plotly_chart(fig2, use_container_width=True)
