"""
Dashboard Components Package

Reusable UI components for Streamlit dashboard.
"""

from .metrics_plot import (
    plot_metrics_comparison,
    plot_metric_over_time,
    plot_prediction_distribution,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_residuals
)

from .comparison_table import (
    display_model_comparison_table,
    display_metrics_summary,
    display_version_comparison_table
)

from .drift_chart import (
    plot_drift_scores,
    plot_feature_distribution_comparison,
    plot_drift_over_time
)

__all__ = [
    'plot_metrics_comparison',
    'plot_metric_over_time',
    'plot_prediction_distribution',
    'plot_actual_vs_predicted',
    'plot_feature_importance',
    'plot_residuals',
    'display_model_comparison_table',
    'display_metrics_summary',
    'display_version_comparison_table',
    'plot_drift_scores',
    'plot_feature_distribution_comparison',
    'plot_drift_over_time'
]
