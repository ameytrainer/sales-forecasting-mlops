"""
Data Ingestion Pipeline DAG

Handles periodic data collection and DVC versioning:
- Generate new data (simulated)
- Validate data
- Version with DVC
- Push to remote

Author: Amey Talkatkar
"""

from datetime import datetime, timedelta
import pendulum

from airflow import DAG
from airflow.decorators import task
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule

import sys
sys.path.insert(0, '/home/ubuntu/sales-forecasting-mlops')

from src.config import get_settings
from src.data import DataValidator

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
    dag_id='data_ingestion_pipeline',
    default_args=default_args,
    description='Ingest and version new data with DVC',
    schedule='0 */6 * * *',  # Every 6 hours
    start_date=pendulum.datetime(2024, 1, 1, tz='UTC'),
    catchup=False,
    tags=['data', 'ingestion', 'dvc'],
    doc_md=__doc__,
)


# Task 1: Generate new data (simulated)
generate_data = BashOperator(
    task_id='generate_new_data',
    bash_command="""
    cd /home/ubuntu/sales-forecasting-mlops && \
    source venv/bin/activate && \
    python data/generate_data.py --rows 1000 --append && \
    echo "Data generation complete"
    """,
    dag=dag,
)


# Task 2: Validate new data
@task(task_id='validate_new_data', dag=dag)
def validate_new_data() -> dict:
    """Validate newly generated data."""
    logger.info("Validating new data...")
    
    settings = get_settings()
    validator = DataValidator()
    
    from src.data import DataLoader
    loader = DataLoader()
    df = loader.load_csv(settings.raw_data_dir / "sales_data.csv")
    
    report = validator.validate_sales_data(df)
    
    if not report['is_valid']:
        raise ValueError("Data validation failed")
    
    logger.info("✅ Data validation passed")
    return {'row_count': report['row_count'], 'is_valid': True}


# Task 3: Add to DVC
dvc_add = BashOperator(
    task_id='dvc_add',
    bash_command="""
    cd /home/ubuntu/sales-forecasting-mlops && \
    source venv/bin/activate && \
    dvc add data/raw/sales_data.csv && \
    echo "DVC add complete"
    """,
    dag=dag,
)


# Task 4: Git commit DVC file
git_commit = BashOperator(
    task_id='git_commit_dvc_file',
    bash_command="""
    cd /home/ubuntu/sales-forecasting-mlops && \
    git add data/raw/sales_data.csv.dvc data/raw/.gitignore && \
    git commit -m "Update data: $(date '+%Y-%m-%d %H:%M:%S')" || echo "No changes to commit" && \
    echo "Git commit complete"
    """,
    dag=dag,
)


# Task 5: Push to DVC remote
dvc_push = BashOperator(
    task_id='dvc_push',
    bash_command="""
    cd /home/ubuntu/sales-forecasting-mlops && \
    source venv/bin/activate && \
    dvc push && \
    echo "DVC push complete"
    """,
    dag=dag,
)


# Task 6: Send notification
@task(task_id='send_notification', trigger_rule=TriggerRule.ALL_SUCCESS, dag=dag)
def send_notification(validation: dict) -> None:
    """Send success notification."""
    logger.info(f"✅ Data ingestion complete: {validation['row_count']} rows")


# Define dependencies
generate_data >> validate_new_data() >> dvc_add >> git_commit >> dvc_push >> send_notification(validate_new_data())


if __name__ == "__main__":
    dag.test()
