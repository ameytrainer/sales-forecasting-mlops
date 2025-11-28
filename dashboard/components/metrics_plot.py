"""
Metrics Plotting Component

Reusable Plotly charts for metrics visualization.

Author: Amey Talkatkar
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List, Optional


def plot_metrics_comparison(
    metrics_df: pd.DataFrame,
    metric_names: List[str],
    title: str = "Metrics Comparison"
) -> go.Figure:
    """
    Create bar chart comparing metrics across models.
    
    Args:
        metrics_df: DataFrame with models and metrics
        metric_names: List of metric column names to plot
        title: Chart title
        
    Returns:
        Plotly figure
        
    Example:
        >>> df = pd.DataFrame({
        ...     'Model': ['LR', 'RF', 'XGB'],
        ...     'RMSE': [105, 92, 87],
        ...     'MAE': [82, 71, 69]
        ... })
        >>> fig = plot_metrics_comparison(df, ['RMSE', 'MAE'])
    """
    fig = go.Figure()
    
    for metric in metric_names:
        fig.add_trace(go.Bar(
            name=metric,
            x=metrics_df['Model'],
            y=metrics_df[metric],
            text=metrics_df[metric].round(2),
            textposition='auto',
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Model",
        yaxis_title="Metric Value",
        barmode='group',
        height=400,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def plot_metric_over_time(
    data: pd.DataFrame,
    time_col: str,
    metric_col: str,
    title: str = "Metric Over Time"
) -> go.Figure:
    """
    Create line chart showing metric evolution over time.
    
    Args:
        data: DataFrame with time series data
        time_col: Column name for timestamp
        metric_col: Column name for metric values
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data[time_col],
        y=data[metric_col],
        mode='lines+markers',
        name=metric_col,
        line=dict(width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=metric_col,
        height=400,
        template='plotly_white',
        hovermode='x'
    )
    
    return fig


def plot_prediction_distribution(
    predictions: List[float],
    actual: Optional[List[float]] = None,
    bins: int = 30
) -> go.Figure:
    """
    Create histogram of prediction distribution.
    
    Args:
        predictions: List of predicted values
        actual: List of actual values (optional)
        bins: Number of histogram bins
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=predictions,
        name='Predictions',
        nbinsx=bins,
        opacity=0.7
    ))
    
    if actual is not None:
        fig.add_trace(go.Histogram(
            x=actual,
            name='Actual',
            nbinsx=bins,
            opacity=0.7
        ))
    
    fig.update_layout(
        title="Prediction Distribution",
        xaxis_title="Value",
        yaxis_title="Frequency",
        barmode='overlay',
        height=400,
        template='plotly_white'
    )
    
    return fig


def plot_actual_vs_predicted(
    actual: List[float],
    predicted: List[float],
    title: str = "Actual vs Predicted"
) -> go.Figure:
    """
    Create scatter plot of actual vs predicted values.
    
    Args:
        actual: List of actual values
        predicted: List of predicted values
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=actual,
        y=predicted,
        mode='markers',
        name='Predictions',
        marker=dict(
            size=8,
            opacity=0.6,
            color='blue'
        )
    ))
    
    # Add perfect prediction line
    min_val = min(min(actual), min(predicted))
    max_val = max(max(actual), max(predicted))
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Actual",
        yaxis_title="Predicted",
        height=500,
        template='plotly_white',
        showlegend=True
    )
    
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importance_values: List[float],
    top_n: int = 10
) -> go.Figure:
    """
    Create horizontal bar chart of feature importance.
    
    Args:
        feature_names: List of feature names
        importance_values: List of importance values
        top_n: Number of top features to show
        
    Returns:
        Plotly figure
    """
    # Create DataFrame and sort
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    })
    df = df.nlargest(top_n, 'Importance')
    
    fig = go.Figure(go.Bar(
        x=df['Importance'],
        y=df['Feature'],
        orientation='h',
        text=df['Importance'].round(3),
        textposition='auto',
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Feature Importance",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=400,
        template='plotly_white'
    )
    
    return fig


def plot_residuals(
    actual: List[float],
    predicted: List[float]
) -> go.Figure:
    """
    Create residual plot.
    
    Args:
        actual: List of actual values
        predicted: List of predicted values
        
    Returns:
        Plotly figure
    """
    residuals = [a - p for a, p in zip(actual, predicted)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=predicted,
        y=residuals,
        mode='markers',
        marker=dict(
            size=8,
            opacity=0.6,
            color='blue'
        )
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title="Residual Plot",
        xaxis_title="Predicted Value",
        yaxis_title="Residual (Actual - Predicted)",
        height=400,
        template='plotly_white'
    )
    
    return fig
