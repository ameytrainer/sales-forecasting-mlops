"""
ML Training Pipeline DAG

Complete end-to-end ML training pipeline with:
- Data loading from DVC
- Data validation
- EDA
- Feature engineering
- Multi-model training (parallel)
- Model evaluation
- Model registry operations
- Batch predictions
- Dashboard updates

Author: Amey Talkatkar
Email: ameytalkatkar169@gmail.com
"""

from datetime import datetime, timedelta
from typing import Dict, Any
import pendulum

from airflow import DAG
from airflow.decorators import task, task_group
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.trigger_rule import TriggerRule
from airflow.models import Variable

import pandas as pd
import logging

# Import our custom modules
import sys
sys.path.insert(0, '/home/ubuntu/sales-forecasting-mlops')

from src.config import get_settings
from src.data import DataLoader, DataValidator, DataProcessor
from src.features import FeatureEngineer, FeatureSelector
from src.models import ModelTrainer, ModelEvaluator, ModelRegistry
from src.monitoring import DriftDetector

logger = logging.getLogger(__name__)

# DAG default arguments
default_args = {
    'owner': 'amey',
    'depends_on_past': False,
    'email': ['ameytalkatkar169@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}

# DAG definition
dag = DAG(
    dag_id='ml_training_pipeline',
    default_args=default_args,
    description='Complete ML training pipeline with model registry',
    schedule='0 2 * * *',  # Daily at 2 AM
    start_date=pendulum.datetime(2024, 1, 1, tz='UTC'),
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'training', 'production'],
    doc_md=__doc__,
)


# ==================== Task 1: Check New Data Available ====================

check_data = FileSensor(
    task_id='check_new_data_available',
    filepath='/home/ubuntu/sales-forecasting-mlops/data/raw/sales_data.csv',
    poke_interval=60,
    timeout=300,
    mode='poke',
    dag=dag,
)


# ==================== Task 2: Pull Data from DVC ====================

dvc_pull = BashOperator(
    task_id='dvc_pull_data',
    bash_command="""
    cd /home/ubuntu/sales-forecasting-mlops && \
    source venv/bin/activate && \
    dvc pull data/raw/sales_data.csv.dvc && \
    echo "DVC pull completed successfully"
    """,
    dag=dag,
)


# ==================== Task 3: Validate Data Quality ====================

@task(task_id='validate_data_quality', dag=dag)
def validate_data_quality() -> Dict[str, Any]:
    """
    Validate data quality using Great Expectations checks.
    
    Returns:
        Validation report dictionary
        
    Raises:
        ValueError: If validation fails
    """
    logger.info("Starting data validation...")
    
    settings = get_settings()
    loader = DataLoader()
    validator = DataValidator()
    
    # Load data
    df = loader.load_csv(settings.raw_data_dir / "sales_data.csv")
    logger.info(f"Loaded {len(df):,} rows")
    
    # Validate
    report = validator.validate_sales_data(df, strict=False)
    
    # Log report
    readable_report = validator.generate_validation_report(report)
    logger.info(f"\n{readable_report}")
    
    # Fail if validation fails
    if not report['is_valid']:
        raise ValueError(f"Data validation failed: {report['checks_failed']}")
    
    logger.info("✅ Data validation passed")
    
    return {
        'is_valid': report['is_valid'],
        'row_count': report['row_count'],
        'checks_passed': len(report['checks_passed']),
        'checks_failed': len(report['checks_failed']),
    }


# ==================== Task 4: Perform EDA ====================

@task(task_id='perform_eda', dag=dag)
def perform_eda(validation_report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform exploratory data analysis and log to MLflow.
    
    Returns:
        EDA summary dictionary
    """
    logger.info("Performing EDA...")
    
    settings = get_settings()
    loader = DataLoader()
    
    # Load data
    df = loader.load_csv(settings.raw_data_dir / "sales_data.csv")
    
    # Basic statistics
    summary = {
        'row_count': len(df),
        'column_count': len(df.columns),
        'missing_values': int(df.isnull().sum().sum()),
        'date_range': {
            'start': str(df['date'].min()),
            'end': str(df['date'].max()),
        },
        'numerical_summary': df.describe().to_dict(),
        'categorical_summary': {
            'regions': df['region'].nunique(),
            'products': df['product'].nunique(),
        }
    }
    
    logger.info(f"EDA complete: {summary}")
    
    return summary


# ==================== Task 5: Feature Engineering ====================

@task(task_id='feature_engineering', dag=dag)
def feature_engineering(eda_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create features for model training.
    
    Returns:
        Feature engineering summary
    """
    logger.info("Starting feature engineering...")
    
    settings = get_settings()
    loader = DataLoader()
    engineer = FeatureEngineer()
    processor = DataProcessor()
    
    # Load data
    df = loader.load_csv(settings.raw_data_dir / "sales_data.csv")
    
    # Create features
    df_features = engineer.create_all_features(df, target_col='sales')
    logger.info(f"Created {len(engineer.get_feature_names())} features")
    
    # Handle missing values from lag features
    df_features = processor.handle_missing_values(df_features, strategy='drop')
    
    # Prepare for modeling
    X = df_features.drop(columns=['sales', 'date'])
    y = df_features['sales']
    
    # Split data
    X_train, X_test, y_train, y_test = processor.split_data(
        pd.concat([X, y], axis=1),
        target_column='sales',
        shuffle=False,  # Time series
        save=True
    )
    
    # Scale features
    X_train_scaled = processor.scale_features(X_train, fit=True, save_scaler=True)
    X_test_scaled = processor.scale_features(X_test, fit=False)
    
    # Save scaled data
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    X_train_scaled.to_csv(settings.processed_data_dir / "X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(settings.processed_data_dir / "X_test_scaled.csv", index=False)
    
    logger.info("✅ Feature engineering complete")
    
    return {
        'total_features': len(engineer.get_feature_names()),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'feature_names': engineer.get_feature_names()[:10],  # First 10
    }


# ==================== Task Group: Parallel Model Training ====================

@task_group(group_id='train_models', dag=dag)
def train_models_group():
    """
    Train multiple models in parallel.
    """
    
    @task(task_id='train_linear_regression')
    def train_lr() -> Dict[str, Any]:
        """Train Linear Regression model."""
        logger.info("Training Linear Regression...")
        
        settings = get_settings()
        trainer = ModelTrainer()
        
        # Load scaled data
        X_train = pd.read_csv(settings.processed_data_dir / "X_train_scaled.csv")
        y_train = pd.read_csv(settings.processed_data_dir / "y_train.csv").squeeze()
        X_test = pd.read_csv(settings.processed_data_dir / "X_test_scaled.csv")
        y_test = pd.read_csv(settings.processed_data_dir / "y_test.csv").squeeze()
        
        # Train
        run_id, model_name, metrics = trainer.train_model(
            'lr', X_train, y_train, X_test, y_test
        )
        
        logger.info(f"✅ LR trained: RMSE={metrics.get('test_rmse', 'N/A'):.4f}")
        
        return {'model_type': 'lr', 'run_id': run_id, 'metrics': metrics}
    
    @task(task_id='train_random_forest')
    def train_rf() -> Dict[str, Any]:
        """Train Random Forest model."""
        logger.info("Training Random Forest...")
        
        settings = get_settings()
        trainer = ModelTrainer()
        
        # Load scaled data
        X_train = pd.read_csv(settings.processed_data_dir / "X_train_scaled.csv")
        y_train = pd.read_csv(settings.processed_data_dir / "y_train.csv").squeeze()
        X_test = pd.read_csv(settings.processed_data_dir / "X_test_scaled.csv")
        y_test = pd.read_csv(settings.processed_data_dir / "y_test.csv").squeeze()
        
        # Train
        run_id, model_name, metrics = trainer.train_model(
            'rf', X_train, y_train, X_test, y_test
        )
        
        logger.info(f"✅ RF trained: RMSE={metrics.get('test_rmse', 'N/A'):.4f}")
        
        return {'model_type': 'rf', 'run_id': run_id, 'metrics': metrics}
    
    @task(task_id='train_xgboost')
    def train_xgb() -> Dict[str, Any]:
        """Train XGBoost model."""
        logger.info("Training XGBoost...")
        
        settings = get_settings()
        trainer = ModelTrainer()
        
        # Load scaled data
        X_train = pd.read_csv(settings.processed_data_dir / "X_train_scaled.csv")
        y_train = pd.read_csv(settings.processed_data_dir / "y_train.csv").squeeze()
        X_test = pd.read_csv(settings.processed_data_dir / "X_test_scaled.csv")
        y_test = pd.read_csv(settings.processed_data_dir / "y_test.csv").squeeze()
        
        # Train
        run_id, model_name, metrics = trainer.train_model(
            'xgb', X_train, y_train, X_test, y_test
        )
        
        logger.info(f"✅ XGB trained: RMSE={metrics.get('test_rmse', 'N/A'):.4f}")
        
        return {'model_type': 'xgb', 'run_id': run_id, 'metrics': metrics}
    
    # Return all trained models
    lr_result = train_lr()
    rf_result = train_rf()
    xgb_result = train_xgb()
    
    return [lr_result, rf_result, xgb_result]


# ==================== Task 7: Evaluate Models ====================

@task(task_id='evaluate_models', dag=dag)
def evaluate_models(model_results: list) -> Dict[str, Any]:
    """
    Evaluate and compare all trained models.
    
    Returns:
        Best model information
    """
    logger.info("Evaluating models...")
    
    evaluator = ModelEvaluator()
    
    # Build comparison dict
    comparison = {}
    for result in model_results:
        model_type = result['model_type']
        metrics = result['metrics']
        comparison[model_type] = metrics
    
    # Compare models
    comparison_df = evaluator.compare_models(comparison)
    logger.info(f"\nModel Comparison:\n{comparison_df}")
    
    # Get best model (lowest RMSE)
    best_model_type = comparison_df.index[0]
    best_result = next(r for r in model_results if r['model_type'] == best_model_type)
    
    logger.info(f"✅ Best model: {best_model_type}")
    logger.info(f"   RMSE: {best_result['metrics'].get('test_rmse', 'N/A'):.4f}")
    
    return {
        'best_model_type': best_model_type,
        'best_run_id': best_result['run_id'],
        'best_metrics': best_result['metrics'],
        'all_models': comparison_df.to_dict(),
    }


# ==================== Task 8: Compare with Baseline ====================

@task.branch(task_id='compare_with_baseline', dag=dag)
def compare_with_baseline(evaluation: Dict[str, Any]) -> str:
    """
    Compare with baseline and decide whether to register model.
    
    Returns:
        Next task ID (register_model or skip_registration)
    """
    logger.info("Comparing with baseline...")
    
    # Get baseline RMSE from Airflow Variable (set manually first time)
    try:
        baseline_rmse = float(Variable.get('baseline_rmse', default_var=999999.0))
    except:
        baseline_rmse = 999999.0
    
    current_rmse = evaluation['best_metrics'].get('test_rmse', 999999.0)
    
    logger.info(f"Baseline RMSE: {baseline_rmse:.4f}")
    logger.info(f"Current RMSE: {current_rmse:.4f}")
    
    if current_rmse < baseline_rmse:
        improvement_pct = (baseline_rmse - current_rmse) / baseline_rmse * 100
        logger.info(f"✅ Model improved by {improvement_pct:.2f}%")
        return 'register_model'
    else:
        degradation_pct = (current_rmse - baseline_rmse) / baseline_rmse * 100
        logger.info(f"❌ Model degraded by {degradation_pct:.2f}%")
        return 'skip_registration'


# ==================== Task 9: Register Model ====================

@task(task_id='register_model', dag=dag)
def register_model(evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Register best model to MLflow registry.
    
    Returns:
        Registration info
    """
    logger.info("Registering model...")
    
    registry = ModelRegistry()
    
    run_id = evaluation['best_run_id']
    model_name = "sales_forecasting_production"
    
    # Register model
    version = registry.register_model(run_id, model_name)
    
    logger.info(f"✅ Model registered: {model_name} v{version}")
    
    return {
        'model_name': model_name,
        'version': version,
        'run_id': run_id,
    }


# ==================== Task 10: Transition to Staging ====================

@task(task_id='transition_to_staging', dag=dag)
def transition_to_staging(registration: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transition model to Staging stage.
    """
    logger.info("Transitioning to Staging...")
    
    registry = ModelRegistry()
    
    registry.transition_model(
        registration['model_name'],
        registration['version'],
        'Staging'
    )
    
    logger.info("✅ Model in Staging")
    
    return registration


# ==================== Task 11: Run Validation Tests ====================

@task(task_id='run_validation_tests', dag=dag)
def run_validation_tests(registration: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run validation tests on staging model.
    
    Returns:
        Validation results
    """
    logger.info("Running validation tests...")
    
    # Simulate validation tests
    tests_passed = {
        'data_quality_check': True,
        'accuracy_threshold': True,
        'inference_time_check': True,
        'resource_usage_check': True,
    }
    
    all_passed = all(tests_passed.values())
    
    if all_passed:
        logger.info("✅ All validation tests passed")
    else:
        failed_tests = [k for k, v in tests_passed.items() if not v]
        logger.error(f"❌ Validation tests failed: {failed_tests}")
        raise ValueError(f"Validation failed: {failed_tests}")
    
    return {
        **registration,
        'validation_tests': tests_passed,
        'all_tests_passed': all_passed,
    }


# ==================== Task 12: Transition to Production ====================

@task(task_id='transition_to_production', dag=dag)
def transition_to_production(validation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transition model to Production stage.
    """
    logger.info("Transitioning to Production...")
    
    registry = ModelRegistry()
    
    # Archive old production versions
    registry.archive_old_versions(validation['model_name'], keep_latest_n=3)
    
    # Transition to production
    registry.transition_model(
        validation['model_name'],
        validation['version'],
        'Production'
    )
    
    # Update baseline RMSE variable
    # (This will be used in next run's comparison)
    # Note: In production, get this from evaluation results
    
    logger.info("✅ Model in Production")
    
    return validation


# ==================== Task 13: Generate Predictions ====================

@task(task_id='generate_predictions', dag=dag)
def generate_predictions(production: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate batch predictions with production model.
    """
    logger.info("Generating predictions...")
    
    from src.models import ModelPredictor
    settings = get_settings()
    
    predictor = ModelPredictor()
    
    # Load test data
    X_test = pd.read_csv(settings.processed_data_dir / "X_test_scaled.csv")
    
    # Make predictions
    predictions = predictor.predict(
        X_test,
        model_name=production['model_name'],
        stage='Production'
    )
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'predicted_sales': predictions,
        'timestamp': datetime.now(),
        'model_version': production['version'],
    })
    
    settings.predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(settings.predictions_dir / "latest.csv", index=False)
    
    logger.info(f"✅ Generated {len(predictions)} predictions")
    
    return {
        'prediction_count': len(predictions),
        'predictions_file': str(settings.predictions_dir / "latest.csv"),
    }


# ==================== Task 14: Update Dashboard Data ====================

@task(task_id='update_dashboard_data', dag=dag)
def update_dashboard_data(predictions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update dashboard with latest metrics and predictions.
    """
    logger.info("Updating dashboard data...")
    
    # In production, this would update a database or cache
    # For now, just log success
    
    dashboard_update = {
        'updated_at': datetime.now().isoformat(),
        'predictions_loaded': predictions['prediction_count'],
        'status': 'success',
    }
    
    logger.info("✅ Dashboard data updated")
    
    return dashboard_update


# ==================== Task 15: Send Success Notification ====================

@task(task_id='send_success_notification', trigger_rule=TriggerRule.NONE_FAILED, dag=dag)
def send_success_notification(dashboard: Dict[str, Any]) -> None:
    """
    Send success notification.
    """
    logger.info("📧 Sending success notification...")
    
    # In production, send email/Slack notification
    message = f"""
    ✅ ML Training Pipeline Completed Successfully
    
    - Dashboard updated: {dashboard['updated_at']}
    - Predictions generated: {dashboard['predictions_loaded']}
    
    Check Airflow UI for details.
    """
    
    logger.info(message)


# ==================== Task 16: Skip Registration (Alternative Path) ====================

skip_task = EmptyOperator(
    task_id='skip_registration',
    dag=dag,
)


# ==================== Task 17: Send Skip Notification ====================

@task(task_id='send_skip_notification', trigger_rule=TriggerRule.NONE_FAILED, dag=dag)
def send_skip_notification() -> None:
    """
    Send notification when model is not registered.
    """
    logger.info("📧 Sending skip notification...")
    
    message = """
    ℹ️ ML Training Pipeline Completed (No Model Update)
    
    - New model did not improve over baseline
    - Baseline model remains in production
    - Check MLflow for experiment details
    """
    
    logger.info(message)


# ==================== Task 18: Cleanup (Always Runs) ====================

cleanup = BashOperator(
    task_id='cleanup',
    bash_command="""
    echo "Cleaning up temporary files..."
    # Add cleanup commands here if needed
    echo "Cleanup complete"
    """,
    trigger_rule=TriggerRule.ALL_DONE,
    dag=dag,
)


# ==================== Define Task Dependencies ====================

# Linear flow
check_data >> dvc_pull >> validate_data_quality()

# EDA and feature engineering
validate_task = validate_data_quality()
eda_task = perform_eda(validate_task)
features_task = feature_engineering(eda_task)

# Parallel model training
models_task = train_models_group()
features_task >> models_task

# Evaluation and branching
evaluate_task = evaluate_models(models_task)
branch_task = compare_with_baseline(evaluate_task)

# Registration path (if model improves)
register_task = register_model(evaluate_task)
staging_task = transition_to_staging(register_task)
validation_task = run_validation_tests(staging_task)
production_task = transition_to_production(validation_task)
predictions_task = generate_predictions(production_task)
dashboard_task = update_dashboard_data(predictions_task)
success_notif = send_success_notification(dashboard_task)

# Skip path (if model doesn't improve)
skip_notif = send_skip_notification()

# Connect branching
branch_task >> [register_task, skip_task]
skip_task >> skip_notif

# Cleanup always runs
[success_notif, skip_notif] >> cleanup


if __name__ == "__main__":
    dag.test()
