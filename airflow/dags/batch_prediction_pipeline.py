"""
Batch Prediction Pipeline DAG

Generate predictions on new data using production model:
- Load new data
- Preprocess features
- Load production model
- Generate predictions
- Store results in database
- Update monitoring metrics

Author: Amey Talkatkar
"""

from datetime import datetime, timedelta
import pendulum

from airflow import DAG
from airflow.decorators import task
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor

import pandas as pd
import sys
sys.path.insert(0, '/home/ubuntu/sales-forecasting-mlops')

from src.config import get_settings
from src.data import DataLoader, DataProcessor
from src.features import FeatureEngineer
from src.models import ModelPredictor

import logging
logger = logging.getLogger(__name__)

default_args = {
    'owner': 'amey',
    'depends_on_past': False,
    'email': ['ameytalkatkar169@gmail.com'],
    'email_on_failure': True,
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
}

dag = DAG(
    dag_id='batch_prediction_pipeline',
    default_args=default_args,
    description='Generate batch predictions with production model',
    schedule='0 */6 * * *',  # Every 6 hours
    start_date=pendulum.datetime(2024, 1, 1, tz='UTC'),
    catchup=False,
    tags=['ml', 'prediction', 'batch'],
    doc_md=__doc__,
)


# Task 1: Check for new data
check_data = FileSensor(
    task_id='check_new_data',
    filepath='/home/ubuntu/sales-forecasting-mlops/data/raw/sales_data.csv',
    poke_interval=60,
    timeout=300,
    dag=dag,
)


# Task 2: Load and prepare data
@task(task_id='load_and_prepare_data', dag=dag)
def load_and_prepare_data() -> dict:
    """Load data and prepare features for prediction."""
    logger.info("Loading and preparing data...")
    
    settings = get_settings()
    loader = DataLoader()
    engineer = FeatureEngineer()
    processor = DataProcessor()
    
    # Load latest data
    df = loader.load_csv(settings.raw_data_dir / "sales_data.csv")
    logger.info(f"Loaded {len(df):,} rows")
    
    # Take only recent data (last 30 days for prediction)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').tail(1000)  # Last 1000 records
    
    # Create features
    df_features = engineer.create_all_features(df, target_col='sales')
    
    # Handle missing values
    df_features = processor.handle_missing_values(df_features, strategy='drop')
    
    # Prepare X (drop target and date)
    X = df_features.drop(columns=['sales', 'date'], errors='ignore')
    
    # Load scaler
    scaler = processor.load_scaler()
    if scaler is not None:
        X_scaled = processor.scale_features(X, fit=False)
    else:
        logger.warning("No scaler found, using unscaled features")
        X_scaled = X
    
    # Save prepared data
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    X_scaled.to_csv(settings.processed_data_dir / "batch_input.csv", index=False)
    
    # Save dates and actuals for later reference
    df_features[['date', 'sales']].to_csv(
        settings.processed_data_dir / "batch_metadata.csv",
        index=False
    )
    
    logger.info(f"✅ Prepared {len(X_scaled)} samples for prediction")
    
    return {
        'sample_count': len(X_scaled),
        'feature_count': len(X_scaled.columns),
    }


# Task 3: Load production model and predict
@task(task_id='generate_predictions', dag=dag)
def generate_predictions(prep_info: dict) -> dict:
    """Generate predictions using production model."""
    logger.info("Generating predictions...")
    
    settings = get_settings()
    predictor = ModelPredictor()
    
    # Load prepared data
    X = pd.read_csv(settings.processed_data_dir / "batch_input.csv")
    metadata = pd.read_csv(settings.processed_data_dir / "batch_metadata.csv")
    
    # Make predictions
    predictions = predictor.predict(
        X,
        model_name="sales_forecasting_production",
        stage="Production"
    )
    
    # Combine with metadata
    results = metadata.copy()
    results['predicted_sales'] = predictions
    results['prediction_timestamp'] = datetime.now()
    results['absolute_error'] = abs(results['sales'] - results['predicted_sales'])
    results['percentage_error'] = (results['absolute_error'] / results['sales']) * 100
    
    # Save results
    settings.predictions_dir.mkdir(parents=True, exist_ok=True)
    output_file = settings.predictions_dir / f"batch_{datetime.now():%Y%m%d_%H%M%S}.csv"
    results.to_csv(output_file, index=False)
    
    # Also save as latest
    results.to_csv(settings.predictions_dir / "latest.csv", index=False)
    
    logger.info(f"✅ Generated {len(predictions)} predictions")
    
    return {
        'prediction_count': len(predictions),
        'output_file': str(output_file),
        'mean_prediction': float(predictions.mean()),
        'std_prediction': float(predictions.std()),
    }


# Task 4: Store in database
@task(task_id='store_in_database', dag=dag)
def store_in_database(pred_info: dict) -> dict:
    """Store predictions in PostgreSQL database."""
    logger.info("Storing predictions in database...")
    
    settings = get_settings()
    
    # Read predictions
    predictions = pd.read_csv(settings.predictions_dir / "latest.csv")
    
    # Connect to database
    from sqlalchemy import create_engine
    engine = create_engine(settings.get_database_url())
    
    # Store in predictions table
    try:
        predictions.to_sql(
            'predictions',
            engine,
            if_exists='append',
            index=False,
            method='multi',
            chunksize=1000
        )
        logger.info(f"✅ Stored {len(predictions)} predictions in database")
        success = True
    except Exception as e:
        logger.error(f"Failed to store in database: {e}")
        success = False
    
    engine.dispose()
    
    return {
        'stored_count': len(predictions) if success else 0,
        'success': success,
    }


# Task 5: Calculate performance metrics
@task(task_id='calculate_metrics', dag=dag)
def calculate_metrics(db_info: dict) -> dict:
    """Calculate prediction performance metrics."""
    logger.info("Calculating performance metrics...")
    
    settings = get_settings()
    predictions = pd.read_csv(settings.predictions_dir / "latest.csv")
    
    from src.models import ModelEvaluator
    evaluator = ModelEvaluator()
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(
        predictions['sales'],
        predictions['predicted_sales']
    )
    
    logger.info(f"Performance metrics: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}")
    
    return metrics


# Task 6: Send notification
@task(task_id='send_notification', dag=dag)
def send_notification(metrics: dict, pred_info: dict) -> None:
    """Send completion notification."""
    logger.info("📧 Sending notification...")
    
    message = f"""
    ✅ Batch Prediction Pipeline Complete
    
    - Predictions generated: {pred_info['prediction_count']}
    - RMSE: {metrics['rmse']:.2f}
    - MAE: {metrics['mae']:.2f}
    - R²: {metrics['r2']:.4f}
    
    Output: {pred_info['output_file']}
    """
    
    logger.info(message)


# Define task dependencies
prep = load_and_prepare_data()
preds = generate_predictions(prep)
db = store_in_database(preds)
metrics = calculate_metrics(db)
notif = send_notification(metrics, preds)

check_data >> prep >> preds >> db >> metrics >> notif


if __name__ == "__main__":
    dag.test()
