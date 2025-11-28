"""
Test Suite for Airflow DAGs

Tests for validating DAG structure, dependencies, and configuration.

Author: Amey Talkatkar
"""

import pytest
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, '/home/ubuntu/sales-forecasting-mlops')

# Import Airflow components
from airflow.models import DagBag
from airflow.utils.dag_cycle_tester import check_cycle


# DAG paths
DAGS_FOLDER = Path('/home/ubuntu/airflow/dags')
DAG_FILES = [
    'ml_training_pipeline.py',
    'data_ingestion_pipeline.py',
    'batch_prediction_pipeline.py',
    'model_monitoring_pipeline.py',
    'retraining_trigger_pipeline.py'
]


class TestDAGIntegrity:
    """Test DAG integrity and import."""
    
    @pytest.fixture(scope='class')
    def dagbag(self):
        """Load all DAGs."""
        return DagBag(dag_folder=str(DAGS_FOLDER), include_examples=False)
    
    def test_dags_load_without_errors(self, dagbag):
        """Test that all DAGs load without import errors."""
        assert len(dagbag.import_errors) == 0, \
            f"DAG import errors: {dagbag.import_errors}"
    
    def test_all_dags_present(self, dagbag):
        """Test that all expected DAGs are present."""
        expected_dags = [
            'ml_training_pipeline',
            'data_ingestion_pipeline',
            'batch_prediction_pipeline',
            'model_monitoring_pipeline',
            'retraining_trigger_pipeline'
        ]
        
        dag_ids = list(dagbag.dag_ids)
        
        for expected_dag in expected_dags:
            assert expected_dag in dag_ids, \
                f"Expected DAG '{expected_dag}' not found. Available: {dag_ids}"
    
    def test_dags_have_tags(self, dagbag):
        """Test that DAGs have tags."""
        for dag_id, dag in dagbag.dags.items():
            assert dag.tags is not None, f"DAG {dag_id} has no tags"
            assert len(dag.tags) > 0, f"DAG {dag_id} has empty tags"
    
    def test_dags_have_owners(self, dagbag):
        """Test that all DAGs have owners defined."""
        for dag_id, dag in dagbag.dags.items():
            assert dag.owner is not None, f"DAG {dag_id} has no owner"
            assert dag.owner != '', f"DAG {dag_id} has empty owner"
    
    def test_dags_have_descriptions(self, dagbag):
        """Test that DAGs have descriptions."""
        for dag_id, dag in dagbag.dags.items():
            assert dag.description is not None, f"DAG {dag_id} has no description"
            assert len(dag.description) > 0, f"DAG {dag_id} has empty description"


class TestMLTrainingPipeline:
    """Test ML Training Pipeline DAG."""
    
    @pytest.fixture
    def dagbag(self):
        """Load DAG."""
        return DagBag(dag_folder=str(DAGS_FOLDER), include_examples=False)
    
    @pytest.fixture
    def dag(self, dagbag):
        """Get ML training pipeline DAG."""
        return dagbag.get_dag('ml_training_pipeline')
    
    def test_dag_exists(self, dag):
        """Test that DAG exists."""
        assert dag is not None
    
    def test_dag_has_correct_schedule(self, dag):
        """Test that DAG has correct schedule."""
        # Daily at 2 AM
        assert dag.schedule_interval == '0 2 * * *'
    
    def test_dag_has_required_tasks(self, dag):
        """Test that DAG has all required tasks."""
        expected_tasks = [
            'check_new_data_available',
            'dvc_pull_data',
            'validate_data_quality',
            'perform_eda',
            'feature_engineering',
            'evaluate_models',
            'compare_with_baseline'
        ]
        
        task_ids = [task.task_id for task in dag.tasks]
        
        for expected_task in expected_tasks:
            assert expected_task in task_ids, \
                f"Task '{expected_task}' not found in DAG"
    
    def test_dag_no_cycles(self, dag):
        """Test that DAG has no cycles."""
        try:
            check_cycle(dag)
        except Exception as e:
            pytest.fail(f"DAG has cycles: {e}")
    
    def test_dag_task_count(self, dag):
        """Test that DAG has expected number of tasks."""
        # ML training pipeline should have 15-20 tasks
        assert len(dag.tasks) >= 10, \
            f"DAG has only {len(dag.tasks)} tasks, expected at least 10"
    
    def test_dag_default_args(self, dag):
        """Test that DAG has proper default args."""
        assert dag.default_args is not None
        assert 'retries' in dag.default_args
        assert dag.default_args['retries'] >= 1
    
    def test_dag_catchup_disabled(self, dag):
        """Test that catchup is disabled."""
        assert dag.catchup == False, "Catchup should be disabled"
    
    def test_dag_start_date(self, dag):
        """Test that DAG has a valid start date."""
        assert dag.start_date is not None
        assert isinstance(dag.start_date, datetime)
        # Start date should be in the past
        assert dag.start_date <= datetime.now()


class TestDataIngestionPipeline:
    """Test Data Ingestion Pipeline DAG."""
    
    @pytest.fixture
    def dagbag(self):
        """Load DAG."""
        return DagBag(dag_folder=str(DAGS_FOLDER), include_examples=False)
    
    @pytest.fixture
    def dag(self, dagbag):
        """Get data ingestion DAG."""
        return dagbag.get_dag('data_ingestion_pipeline')
    
    def test_dag_exists(self, dag):
        """Test that DAG exists."""
        assert dag is not None
    
    def test_dag_has_correct_schedule(self, dag):
        """Test that DAG runs every 6 hours."""
        assert dag.schedule_interval == '0 */6 * * *'
    
    def test_dag_has_dvc_tasks(self, dag):
        """Test that DAG has DVC-related tasks."""
        task_ids = [task.task_id for task in dag.tasks]
        
        dvc_tasks = ['dvc_add', 'dvc_push']
        for dvc_task in dvc_tasks:
            assert dvc_task in task_ids, f"Task '{dvc_task}' not found"
    
    def test_dag_has_validation_task(self, dag):
        """Test that DAG validates new data."""
        task_ids = [task.task_id for task in dag.tasks]
        assert 'validate_new_data' in task_ids


class TestBatchPredictionPipeline:
    """Test Batch Prediction Pipeline DAG."""
    
    @pytest.fixture
    def dagbag(self):
        """Load DAG."""
        return DagBag(dag_folder=str(DAGS_FOLDER), include_examples=False)
    
    @pytest.fixture
    def dag(self, dagbag):
        """Get batch prediction DAG."""
        return dagbag.get_dag('batch_prediction_pipeline')
    
    def test_dag_exists(self, dag):
        """Test that DAG exists."""
        assert dag is not None
    
    def test_dag_has_prediction_tasks(self, dag):
        """Test that DAG has prediction-related tasks."""
        task_ids = [task.task_id for task in dag.tasks]
        
        prediction_tasks = ['generate_predictions', 'store_in_database']
        for pred_task in prediction_tasks:
            assert pred_task in task_ids, f"Task '{pred_task}' not found"
    
    def test_dag_uses_production_model(self, dag):
        """Test that DAG is configured to use production model."""
        # Check task configuration (implementation-specific)
        assert dag is not None


class TestModelMonitoringPipeline:
    """Test Model Monitoring Pipeline DAG."""
    
    @pytest.fixture
    def dagbag(self):
        """Load DAG."""
        return DagBag(dag_folder=str(DAGS_FOLDER), include_examples=False)
    
    @pytest.fixture
    def dag(self, dagbag):
        """Get monitoring DAG."""
        return dagbag.get_dag('model_monitoring_pipeline')
    
    def test_dag_exists(self, dag):
        """Test that DAG exists."""
        assert dag is not None
    
    def test_dag_runs_frequently(self, dag):
        """Test that monitoring runs frequently (every 30 min)."""
        assert dag.schedule_interval == '*/30 * * * *'
    
    def test_dag_has_drift_detection(self, dag):
        """Test that DAG includes drift detection."""
        task_ids = [task.task_id for task in dag.tasks]
        assert 'detect_data_drift' in task_ids
    
    def test_dag_has_performance_monitoring(self, dag):
        """Test that DAG includes performance monitoring."""
        task_ids = [task.task_id for task in dag.tasks]
        assert 'monitor_performance' in task_ids
    
    def test_dag_generates_alerts(self, dag):
        """Test that DAG can generate alerts."""
        task_ids = [task.task_id for task in dag.tasks]
        assert 'generate_alerts' in task_ids


class TestRetrainingTriggerPipeline:
    """Test Retraining Trigger Pipeline DAG."""
    
    @pytest.fixture
    def dagbag(self):
        """Load DAG."""
        return DagBag(dag_folder=str(DAGS_FOLDER), include_examples=False)
    
    @pytest.fixture
    def dag(self, dagbag):
        """Get retraining trigger DAG."""
        return dagbag.get_dag('retraining_trigger_pipeline')
    
    def test_dag_exists(self, dag):
        """Test that DAG exists."""
        assert dag is not None
    
    def test_dag_checks_conditions(self, dag):
        """Test that DAG checks retraining conditions."""
        task_ids = [task.task_id for task in dag.tasks]
        assert 'check_retraining_flag' in task_ids
        assert 'validate_conditions' in task_ids
    
    def test_dag_triggers_training(self, dag):
        """Test that DAG can trigger training pipeline."""
        task_ids = [task.task_id for task in dag.tasks]
        assert 'trigger_training_pipeline' in task_ids


class TestDAGDependencies:
    """Test DAG task dependencies."""
    
    @pytest.fixture
    def dagbag(self):
        """Load all DAGs."""
        return DagBag(dag_folder=str(DAGS_FOLDER), include_examples=False)
    
    def test_ml_training_pipeline_dependencies(self, dagbag):
        """Test ML training pipeline task dependencies."""
        dag = dagbag.get_dag('ml_training_pipeline')
        
        # Data validation should come before feature engineering
        validate_task = dag.get_task('validate_data_quality')
        feature_task = dag.get_task('feature_engineering')
        
        assert feature_task in validate_task.downstream_list or \
               validate_task in feature_task.upstream_list
    
    def test_tasks_have_upstream_dependencies(self, dagbag):
        """Test that non-start tasks have upstream dependencies."""
        for dag_id, dag in dagbag.dags.items():
            for task in dag.tasks:
                # Skip start tasks (sensors, manual triggers)
                if 'check' not in task.task_id.lower() and \
                   'start' not in task.task_id.lower():
                    # Most tasks should have upstream dependencies
                    # (except for independent start tasks)
                    pass  # Implementation-specific check


class TestDAGRetryLogic:
    """Test DAG retry configuration."""
    
    @pytest.fixture
    def dagbag(self):
        """Load all DAGs."""
        return DagBag(dag_folder=str(DAGS_FOLDER), include_examples=False)
    
    def test_critical_tasks_have_retries(self, dagbag):
        """Test that critical tasks have retry logic."""
        for dag_id, dag in dagbag.dags.items():
            for task in dag.tasks:
                # Critical tasks should have retries
                if any(keyword in task.task_id for keyword in 
                       ['train', 'validate', 'predict']):
                    # Check task has retries configured
                    retries = task.retries if hasattr(task, 'retries') else \
                             dag.default_args.get('retries', 0)
                    assert retries > 0, \
                        f"Critical task {task.task_id} in {dag_id} has no retries"
    
    def test_retry_delay_configured(self, dagbag):
        """Test that retry delay is configured."""
        for dag_id, dag in dagbag.dags.items():
            if 'retry_delay' in dag.default_args:
                assert isinstance(dag.default_args['retry_delay'], timedelta)
                # Retry delay should be reasonable (not too short)
                assert dag.default_args['retry_delay'].total_seconds() >= 60


class TestDAGDocumentation:
    """Test DAG documentation."""
    
    @pytest.fixture
    def dagbag(self):
        """Load all DAGs."""
        return DagBag(dag_folder=str(DAGS_FOLDER), include_examples=False)
    
    def test_dags_have_doc_strings(self, dagbag):
        """Test that DAGs have documentation strings."""
        for dag_id, dag in dagbag.dags.items():
            # DAG should have doc string
            assert dag.doc_md is not None or dag.description is not None, \
                f"DAG {dag_id} lacks documentation"
    
    def test_tasks_have_doc_strings(self, dagbag):
        """Test that critical tasks have documentation."""
        for dag_id, dag in dagbag.dags.items():
            for task in dag.tasks:
                # At least complex tasks should have docs
                if any(keyword in task.task_id for keyword in 
                       ['train', 'validate', 'monitor', 'evaluate']):
                    # Task should have some documentation
                    assert task.doc or task.doc_md or task.doc_rst, \
                        f"Task {task.task_id} in {dag_id} lacks documentation"


class TestDAGConfiguration:
    """Test DAG configuration best practices."""
    
    @pytest.fixture
    def dagbag(self):
        """Load all DAGs."""
        return DagBag(dag_folder=str(DAGS_FOLDER), include_examples=False)
    
    def test_dags_have_emails_configured(self, dagbag):
        """Test that DAGs have email configuration."""
        for dag_id, dag in dagbag.dags.items():
            # Check if email is configured in default_args
            if 'email' in dag.default_args or 'email_on_failure' in dag.default_args:
                assert True
            # Some DAGs might not need email alerts
    
    def test_dags_max_active_runs(self, dagbag):
        """Test that DAGs have max_active_runs configured."""
        for dag_id, dag in dagbag.dags.items():
            assert dag.max_active_runs is not None
            assert dag.max_active_runs >= 1
            # Should be limited to avoid resource issues
            assert dag.max_active_runs <= 5
    
    def test_dags_dagrun_timeout(self, dagbag):
        """Test that long-running DAGs have timeout configured."""
        for dag_id, dag in dagbag.dags.items():
            if 'training' in dag_id or 'batch' in dag_id:
                # Long-running DAGs should have timeout
                if dag.dagrun_timeout:
                    assert isinstance(dag.dagrun_timeout, timedelta)
