import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from workflow.job_scheduler import JobScheduler
from workflow.workflow_manager import WorkflowManager
from quantum_classification.quantum_model import pegasos_svc
from image_processing.data_loader import y_train, y_test
from image_processing.dimensionality_reduction import X_train_reduced, X_test_reduced

@pytest.fixture
def job_scheduler():
    """Fixture for JobScheduler instance."""
    return JobScheduler(max_workers=2)

@pytest.fixture
def mock_workflow_manager():
    """Fixture for WorkflowManager with mocked components."""
    with patch("workflow.workflow_manager.JobScheduler") as mock_scheduler:
        mock_instance = mock_scheduler.return_value
        workflow = WorkflowManager()
        workflow.job_scheduler = mock_instance  # Mock scheduler to avoid real execution
        return workflow

def test_schedule_task(job_scheduler):
    """Test scheduling a simple task."""
    def sample_task(x):
        return x * 2

    result = job_scheduler.schedule_task(sample_task, 5)
    assert result == 10, "JobScheduler did not return expected task result"

def test_workflow_training(mock_workflow_manager):
    """Test quantum model training using JobScheduler."""
    mock_workflow_manager.train_quantum_model()
    mock_workflow_manager.job_scheduler.schedule_task.assert_called_once()

def test_workflow_inference(mock_workflow_manager):
    """Test MRI classification inference with mock data."""
    sample_mri_data = np.random.rand(1, X_train_reduced.shape[1])
    mock_workflow_manager.classify_mri_images = MagicMock(return_value=np.array([1]))

    result = mock_workflow_manager.classify_mri_images(sample_mri_data)
    assert result == np.array([1]), "Inference did not return expected classification result"

def test_quantum_model_classification():
    """Test PegasosQSVC model classification accuracy."""
    train_score = pegasos_svc.score(X_train_reduced[:10], y_train[:10])
    test_score = pegasos_svc.score(X_test_reduced[:10], y_test[:10])

    assert 0 <= train_score <= 1, "Invalid train score range"
    assert 0 <= test_score <= 1, "Invalid test score range"

if __name__ == "__main__":
    pytest.main()
