"""
Model Monitoring Pipeline DAG

Monitor model performance and data drift:
- Check data drift
- Monitor prediction performance
- Compare with baseline
- Generate alerts
- Store monitoring metrics

Author: Amey Talkatkar
"""

from datetime import datetime, timedelta
import pendulum

from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import PythonOperator
from airflow.models import Variable

import pandas as pd
import sys
sys.path.insert(0, '/home/ubuntu/sales-forecasting-mlops')

from src.config import get_settings
from src.data import DataLoader
from src.monitoring import DriftDetector, PerformanceMonitor

import logging
logger = logging.getLogger(__name__)

default_args = {
    'owner': 'amey',
    'depends_on_past': False,
    'email': ['ameytalkatkar169@gmail.com'],
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    dag_id='model_monitoring_pipeline',
    default_args=default_args,
    description='Monitor model performance and data drift',
    schedule='*/30 * * * *',  # Every 30 minutes
    start_date=pendulum.datetime(2024, 1, 1, tz='UTC'),
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'monitoring', 'drift'],
    doc_md=__doc__,
)


# Task 1: Load reference and current data
@task(task_id='load_data', dag=dag)
def load_data() -> dict:
    """Load reference and current data for comparison."""
    logger.info("Loading data for monitoring...")
    
    settings = get_settings()
    loader = DataLoader()
    
    # Load full dataset
    df = loader.load_csv(settings.raw_data_dir / "sales_data.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Split into reference (training period) and current (recent)
    split_date = df['date'].max() - timedelta(days=30)
    
    reference_data = df[df['date'] < split_date]
    current_data = df[df['date'] >= split_date]
    
    # Save for downstream tasks
    reference_data.to_csv(settings.processed_data_dir / "reference_data.csv", index=False)
    current_data.to_csv(settings.processed_data_dir / "current_data.csv", index=False)
    
    logger.info(f"Reference data: {len(reference_data)} rows")
    logger.info(f"Current data: {len(current_data)} rows")
    
    return {
        'reference_count': len(reference_data),
        'current_count': len(current_data),
        'split_date': str(split_date),
    }


# Task 2: Detect data drift
@task(task_id='detect_data_drift', dag=dag)
def detect_data_drift(data_info: dict) -> dict:
    """Detect drift in input features."""
    logger.info("Detecting data drift...")
    
    settings = get_settings()
    detector = DriftDetector()
    
    # Load data
    reference = pd.read_csv(settings.processed_data_dir / "reference_data.csv")
    current = pd.read_csv(settings.processed_data_dir / "current_data.csv")
    
    # Check drift on key features
    features = ['price', 'quantity', 'region', 'product']
    numeric_features = ['price', 'quantity']
    
    # Detect drift
    drift_report = detector.detect_drift(
        reference[numeric_features],
        current[numeric_features],
        features=numeric_features
    )
    
    logger.info(f"Drift detection complete:")
    logger.info(f"  - Has drift: {drift_report['has_drift']}")
    logger.info(f"  - Drifted features: {drift_report['drifted_features']}")
    logger.info(f"  - Drift percentage: {drift_report['drift_percentage']:.1f}%")
    
    # Calculate PSI for numeric features
    psi_scores = {}
    for feature in numeric_features:
        psi = detector.calculate_psi(
            reference[feature].values,
            current[feature].values
        )
        psi_scores[feature] = float(psi)
        logger.info(f"  - PSI ({feature}): {psi:.4f}")
    
    drift_report['psi_scores'] = psi_scores
    
    return drift_report


# Task 3: Monitor prediction performance
@task(task_id='monitor_performance', dag=dag)
def monitor_performance(data_info: dict) -> dict:
    """Monitor model prediction performance."""
    logger.info("Monitoring model performance...")
    
    settings = get_settings()
    monitor = PerformanceMonitor()
    
    # Load latest predictions
    try:
        predictions = pd.read_csv(settings.predictions_dir / "latest.csv")
    except FileNotFoundError:
        logger.warning("No predictions found, skipping performance monitoring")
        return {'monitoring_skipped': True}
    
    # Get baseline metrics from Variable (set during training)
    try:
        baseline_rmse = float(Variable.get('baseline_rmse', default_var=100.0))
        baseline_mae = float(Variable.get('baseline_mae', default_var=80.0))
        baseline_r2 = float(Variable.get('baseline_r2', default_var=0.85))
    except:
        baseline_rmse = 100.0
        baseline_mae = 80.0
        baseline_r2 = 0.85
    
    baseline_metrics = {
        'rmse': baseline_rmse,
        'mae': baseline_mae,
        'r2': baseline_r2,
    }
    
    # Check performance
    performance_report = monitor.check_performance(
        predictions,
        baseline_metrics,
        y_true_col='sales',
        y_pred_col='predicted_sales'
    )
    
    logger.info(f"Performance monitoring complete:")
    logger.info(f"  - Degradation detected: {performance_report['degradation_detected']}")
    
    for metric, comparison in performance_report['metric_comparisons'].items():
        logger.info(f"  - {metric.upper()}:")
        logger.info(f"      Baseline: {comparison['baseline']:.4f}")
        logger.info(f"      Current: {comparison['current']:.4f}")
        logger.info(f"      Change: {comparison['degradation_pct']:.2f}%")
    
    return performance_report


# Task 4: Generate alerts
@task(task_id='generate_alerts', dag=dag)
def generate_alerts(drift_report: dict, performance_report: dict) -> dict:
    """Generate alerts based on monitoring results."""
    logger.info("Generating alerts...")
    
    alerts = []
    
    # Check for data drift
    if drift_report.get('has_drift', False):
        severity = 'HIGH' if drift_report['drift_percentage'] > 30 else 'MEDIUM'
        alerts.append({
            'type': 'DATA_DRIFT',
            'severity': severity,
            'message': f"Data drift detected in {len(drift_report['drifted_features'])} features",
            'details': drift_report['drifted_features'],
        })
    
    # Check PSI scores
    for feature, psi in drift_report.get('psi_scores', {}).items():
        if psi > 0.2:  # Significant drift
            alerts.append({
                'type': 'HIGH_PSI',
                'severity': 'HIGH' if psi > 0.3 else 'MEDIUM',
                'message': f"High PSI score for {feature}: {psi:.4f}",
                'details': {'feature': feature, 'psi': psi},
            })
    
    # Check for performance degradation
    if performance_report.get('degradation_detected', False):
        alerts.append({
            'type': 'PERFORMANCE_DEGRADATION',
            'severity': 'HIGH',
            'message': "Model performance has degraded",
            'details': performance_report['metric_comparisons'],
        })
    
    logger.info(f"Generated {len(alerts)} alerts")
    for alert in alerts:
        logger.warning(f"⚠️ ALERT [{alert['severity']}] {alert['type']}: {alert['message']}")
    
    return {
        'alert_count': len(alerts),
        'alerts': alerts,
    }


# Task 5: Store monitoring metrics
@task(task_id='store_metrics', dag=dag)
def store_metrics(drift_report: dict, performance_report: dict, alerts: dict) -> dict:
    """Store monitoring metrics in database."""
    logger.info("Storing monitoring metrics...")
    
    settings = get_settings()
    
    # Create monitoring summary
    monitoring_data = {
        'timestamp': datetime.now(),
        'has_drift': drift_report.get('has_drift', False),
        'drift_percentage': drift_report.get('drift_percentage', 0.0),
        'drifted_features': ','.join(drift_report.get('drifted_features', [])),
        'degradation_detected': performance_report.get('degradation_detected', False),
        'alert_count': alerts['alert_count'],
    }
    
    # Add current metrics
    if 'current_metrics' in performance_report:
        for metric, value in performance_report['current_metrics'].items():
            monitoring_data[f'current_{metric}'] = value
    
    # Store in database
    from sqlalchemy import create_engine
    engine = create_engine(settings.get_database_url())
    
    try:
        df = pd.DataFrame([monitoring_data])
        df.to_sql('monitoring_metrics', engine, if_exists='append', index=False)
        logger.info("✅ Monitoring metrics stored")
        success = True
    except Exception as e:
        logger.error(f"Failed to store metrics: {e}")
        success = False
    
    engine.dispose()
    
    return {'stored': success}


# Task 6: Check if retraining needed
@task(task_id='check_retraining_needed', dag=dag)
def check_retraining_needed(drift_report: dict, performance_report: dict) -> dict:
    """Determine if model retraining is needed."""
    logger.info("Checking if retraining needed...")
    
    from src.monitoring import PerformanceMonitor
    monitor = PerformanceMonitor()
    
    # Use monitoring results to decide
    decision = monitor.should_retrain(performance_report, drift_report)
    
    logger.info(f"Retraining decision: {decision['should_retrain']}")
    if decision['should_retrain']:
        logger.info(f"Reasons: {', '.join(decision['reasons'])}")
        logger.info(f"Priority: {decision['priority']}")
    
    # Store decision in Airflow Variable for retraining pipeline
    if decision['should_retrain']:
        Variable.set('retraining_needed', 'true')
        Variable.set('retraining_priority', decision['priority'])
        Variable.set('retraining_reasons', ','.join(decision['reasons']))
    
    return decision


# Task 7: Send monitoring report
@task(task_id='send_monitoring_report', dag=dag)
def send_monitoring_report(alerts: dict, retraining: dict) -> None:
    """Send monitoring report notification."""
    logger.info("📧 Sending monitoring report...")
    
    message = f"""
    📊 Model Monitoring Report
    
    - Alerts generated: {alerts['alert_count']}
    - Retraining needed: {retraining['should_retrain']}
    """
    
    if alerts['alert_count'] > 0:
        message += "\n⚠️ Alerts:\n"
        for alert in alerts['alerts']:
            message += f"  - [{alert['severity']}] {alert['message']}\n"
    
    if retraining['should_retrain']:
        message += f"\n🔄 Retraining recommended (Priority: {retraining['priority']})\n"
        message += f"Reasons: {', '.join(retraining['reasons'])}\n"
    
    logger.info(message)


# Define task dependencies
data = load_data()
drift = detect_data_drift(data)
perf = monitor_performance(data)
alerts = generate_alerts(drift, perf)
store = store_metrics(drift, perf, alerts)
retrain = check_retraining_needed(drift, perf)
report = send_monitoring_report(alerts, retrain)

data >> [drift, perf]
[drift, perf] >> alerts >> store
[drift, perf] >> retrain >> report


if __name__ == "__main__":
    dag.test()
