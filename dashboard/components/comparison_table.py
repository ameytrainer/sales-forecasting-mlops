"""
Comparison Table Component

Reusable tables for model comparison.

Author: Amey Talkatkar
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any


def display_model_comparison_table(
    models_data: List[Dict[str, Any]],
    highlight_best: bool = True
):
    """
    Display model comparison table with highlighting.
    
    Args:
        models_data: List of model dictionaries with metrics
        highlight_best: Whether to highlight best values
        
    Example:
        >>> models = [
        ...     {'Model': 'LR', 'RMSE': 105, 'MAE': 82, 'R²': 0.82},
        ...     {'Model': 'RF', 'RMSE': 92, 'MAE': 71, 'R²': 0.88}
        ... ]
        >>> display_model_comparison_table(models)
    """
    df = pd.DataFrame(models_data)
    
    if highlight_best:
        # Style function to highlight best values
        def highlight_best_values(s):
            if s.name in ['RMSE', 'MAE']:  # Lower is better
                is_min = s == s.min()
                return ['background-color: lightgreen' if v else '' for v in is_min]
            elif s.name in ['R²', 'Accuracy']:  # Higher is better
                is_max = s == s.max()
                return ['background-color: lightgreen' if v else '' for v in is_max]
            return ['' for _ in s]
        
        styled_df = df.style.apply(highlight_best_values, subset=df.select_dtypes(include='number').columns)
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)


def display_metrics_summary(metrics: Dict[str, float]):
    """
    Display metrics in a nice summary format.
    
    Args:
        metrics: Dictionary of metric names and values
    """
    cols = st.columns(len(metrics))
    
    for i, (metric_name, metric_value) in enumerate(metrics.items()):
        with cols[i]:
            st.metric(
                label=metric_name,
                value=f"{metric_value:.4f}" if isinstance(metric_value, float) else metric_value
            )


def display_version_comparison_table(versions_data: List[Dict[str, Any]]):
    """
    Display model version comparison table.
    
    Args:
        versions_data: List of version dictionaries
    """
    df = pd.DataFrame(versions_data)
    
    # Format timestamps if present
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
    
    st.dataframe(df, use_container_width=True)
