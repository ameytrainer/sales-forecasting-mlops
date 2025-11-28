"""
Retraining Trigger Pipeline DAG

Automatically trigger retraining when conditions are met:
- Check monitoring flags
- Validate conditions
- Trigger training pipeline
- Reset flags

Author: Amey Talkatkar
"""

from datetime import datetime, timedelta
import pendulum

from airflow import DAG
from airflow.decorators import task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.models import Variable
from airflow.sensors.external_task import ExternalTaskSensor

import sys
sys.path.insert(0, '/home/ubuntu/sales-forecasting-mlops')

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
    dag_id='retraining_trigger_pipeline',
    default_args=default_args,
    description='Trigger model retraining when needed',
    schedule='0 */1 * * *',  # Every hour
    start_date=pendulum.datetime(2024, 1, 1, tz='UTC'),
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'retraining', 'automation'],
    doc_md=__doc__,
)


# Task 1: Check if retraining is needed
@task(task_id='check_retraining_flag', dag=dag)
def check_retraining_flag() -> dict:
    """Check if retraining flag is set by monitoring pipeline."""
    logger.info("Checking retraining flag...")
    
    # Get flag from Airflow Variable
    try:
        retraining_needed = Variable.get('retraining_needed', default_var='false')
        priority = Variable.get('retraining_priority', default_var='LOW')
        reasons = Variable.get('retraining_reasons', default_var='')
    except:
        retraining_needed = 'false'
        priority = 'LOW'
        reasons = ''
    
    is_needed = retraining_needed.lower() == 'true'
    
    logger.info(f"Retraining needed: {is_needed}")
    if is_needed:
        logger.info(f"Priority: {priority}")
        logger.info(f"Reasons: {reasons}")
    
    return {
        'retraining_needed': is_needed,
        'priority': priority,
        'reasons': reasons,
    }


# Task 2: Validate conditions
@task.branch(task_id='validate_conditions', dag=dag)
def validate_conditions(flag_info: dict) -> str:
    """Validate that conditions are still met for retraining."""
    logger.info("Validating retraining conditions...")
    
    if not flag_info['retraining_needed']:
        logger.info("❌ Retraining not needed")
        return 'skip_retraining'
    
    # Check if enough time has passed since last training
    try:
        last_training = Variable.get('last_training_timestamp', default_var='')
        if last_training:
            last_training_dt = datetime.fromisoformat(last_training)
            hours_since = (datetime.now() - last_training_dt).total_seconds() / 3600
            
            # Don't retrain if less than 24 hours ago (unless HIGH priority)
            if hours_since < 24 and flag_info['priority'] != 'HIGH':
                logger.info(f"⏰ Only {hours_since:.1f} hours since last training, skipping")
                return 'skip_retraining'
    except:
        pass
    
    logger.info("✅ Conditions validated, proceeding with retraining")
    return 'send_pre_training_notification'


# Task 3: Send pre-training notification
@task(task_id='send_pre_training_notification', dag=dag)
def send_pre_training_notification(flag_info: dict) -> None:
    """Send notification before triggering retraining."""
    logger.info("📧 Sending pre-training notification...")
    
    message = f"""
    🔄 Automated Retraining Triggered
    
    Priority: {flag_info['priority']}
    Reasons: {flag_info['reasons']}
    
    Training pipeline will start shortly...
    """
    
    logger.info(message)


# Task 4: Trigger training pipeline
trigger_training = TriggerDagRunOperator(
    task_id='trigger_training_pipeline',
    trigger_dag_id='ml_training_pipeline',
    wait_for_completion=False,  # Don't wait, let it run independently
    conf={
        'triggered_by': 'retraining_trigger',
        'automated': True,
    },
    dag=dag,
)


# Task 5: Reset retraining flag
@task(task_id='reset_retraining_flag', dag=dag)
def reset_retraining_flag() -> None:
    """Reset retraining flag after triggering."""
    logger.info("Resetting retraining flag...")
    
    Variable.set('retraining_needed', 'false')
    Variable.set('last_training_timestamp', datetime.now().isoformat())
    
    logger.info("✅ Flag reset, training timestamp updated")


# Task 6: Skip retraining (empty task for branch)
@task(task_id='skip_retraining', dag=dag)
def skip_retraining() -> None:
    """Skip retraining when conditions not met."""
    logger.info("⏭️ Skipping retraining")


# Task 7: Send completion notification
@task(task_id='send_completion_notification', dag=dag)
def send_completion_notification(flag_info: dict) -> None:
    """Send completion notification."""
    logger.info("📧 Sending completion notification...")
    
    if flag_info['retraining_needed']:
        message = "✅ Retraining pipeline triggered successfully"
    else:
        message = "ℹ️ No retraining needed at this time"
    
    logger.info(message)


# Define task dependencies
flag = check_retraining_flag()
branch = validate_conditions(flag)
pre_notif = send_pre_training_notification(flag)
reset = reset_retraining_flag()
skip = skip_retraining()
final_notif = send_completion_notification(flag)

flag >> branch
branch >> pre_notif >> trigger_training >> reset >> final_notif
branch >> skip >> final_notif


if __name__ == "__main__":
    dag.test()
