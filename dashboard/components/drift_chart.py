"""
Drift Chart Component

Visualizations for data drift monitoring.

Author: Amey Talkatkar
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List


def plot_drift_scores(
    feature_names: List[str],
    drift_scores: List[float],
    threshold: float = 0.05
) -> go.Figure:
    """
    Create bar chart of drift scores with threshold line.
    
    Args:
        feature_names: List of feature names
        drift_scores: List of drift scores
        threshold: Drift detection threshold
        
    Returns:
        Plotly figure
    """
    # Determine colors based on threshold
    colors = ['red' if score > threshold else 'green' for score in drift_scores]
    
    fig = go.Figure(go.Bar(
        x=feature_names,
        y=drift_scores,
        marker_color=colors,
        text=[f"{score:.4f}" for score in drift_scores],
        textposition='auto',
    ))
    
    # Add threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Threshold: {threshold}",
        annotation_position="right"
    )
    
    fig.update_layout(
        title="Feature Drift Scores",
        xaxis_title="Feature",
        yaxis_title="Drift Score (KS Statistic)",
        height=400,
        template='plotly_white'
    )
    
    return fig


def plot_feature_distribution_comparison(
    reference_data: pd.Series,
    current_data: pd.Series,
    feature_name: str
) -> go.Figure:
    """
    Compare distributions of reference vs current data.
    
    Args:
        reference_data: Reference period data
        current_data: Current period data
        feature_name: Name of feature
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=reference_data,
        name='Reference',
        opacity=0.7,
        nbinsx=30
    ))
    
    fig.add_trace(go.Histogram(
        x=current_data,
        name='Current',
        opacity=0.7,
        nbinsx=30
    ))
    
    fig.update_layout(
        title=f"Distribution Comparison: {feature_name}",
        xaxis_title=feature_name,
        yaxis_title="Frequency",
        barmode='overlay',
        height=400,
        template='plotly_white'
    )
    
    return fig


def plot_drift_over_time(
    time_data: pd.DataFrame,
    time_col: str,
    drift_col: str,
    threshold: float = 0.05
) -> go.Figure:
    """
    Plot drift score evolution over time.
    
    Args:
        time_data: DataFrame with time series drift data
        time_col: Column name for timestamps
        drift_col: Column name for drift scores
        threshold: Drift threshold
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Add drift score line
    fig.add_trace(go.Scatter(
        x=time_data[time_col],
        y=time_data[drift_col],
        mode='lines+markers',
        name='Drift Score',
        line=dict(width=2)
    ))
    
    # Add threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text="Threshold"
    )
    
    # Add shaded region for drift
    fig.add_hrect(
        y0=threshold,
        y1=time_data[drift_col].max() * 1.1,
        fillcolor="red",
        opacity=0.1,
        line_width=0,
    )
    
    fig.update_layout(
        title="Drift Score Over Time",
        xaxis_title="Time",
        yaxis_title="Drift Score",
        height=400,
        template='plotly_white'
    )
    
    return fig
