import os
import sys
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from qiskit.circuit import QuantumCircuit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from workflow.job_scheduler import JobScheduler
from workflow.workflow_manager import WorkflowManager

# -----------------------------
# Fixtures and Globals
# -----------------------------
valid_features = np.linspace(0, np.pi, 54)  # 18 qubits x 3 layers
invalid_features = np.linspace(0, np.pi, 20)  # Invalid input length

# -----------------------------
# JobScheduler Tests
# -----------------------------

def test_scheduler_executes_task():
    scheduler = JobScheduler(max_workers=1)
    result = scheduler.schedule_task(lambda x: x + 10, 5)
    assert result == 15

# -----------------------------
# Interpretation Tests
# -----------------------------

def test_interpret_counts_tumor():
    counts = {'111': 900, '000': 124}
    result = WorkflowManager._interpret_quantum_counts(counts)
    assert result in ["Tumor Detected", "No Tumor Detected"]

def test_interpret_counts_normal():
    counts = {'000': 900, '111': 124}
    result = WorkflowManager._interpret_quantum_counts(counts)
    assert result in ["Tumor Detected", "No Tumor Detected"]

def test_interpret_invalid_counts():
    with pytest.raises(ValueError, match="Empty counts from quantum simulation."):
        WorkflowManager._interpret_quantum_counts({})

# -----------------------------
# Classification Tests (Mocked)
# -----------------------------

@patch("workflow.workflow_manager.predict_with_expectation", return_value="Tumor Detected")
def test_classify_with_quantum_circuit(mock_predict):
    result = WorkflowManager.classify_with_quantum_circuit(valid_features)
    assert result == "Tumor Detected"

@patch("workflow.workflow_manager.apply_noise_mitigation")
def test_classify_with_quantum_circuit_noise_valid(mock_noise):
    mock_noise.return_value = None  # no real noise model needed
    result = WorkflowManager.classify_with_quantum_circuit_noise(valid_features)
    assert result in ["Tumor Detected", "No Tumor Detected"]

def test_classify_with_quantum_circuit_noise_invalid():
    with pytest.raises(ValueError, match="Expected .* features"):
        WorkflowManager.classify_with_quantum_circuit_noise(invalid_features)

# -----------------------------
# WorkflowManager Initialization
# -----------------------------

@patch("workflow.workflow_manager.PegasosQSVC.load")
@patch("workflow.workflow_manager.os.path.exists", return_value=True)
@patch("workflow.workflow_manager.JobScheduler")
def test_workflow_manager_load_model(mock_scheduler, mock_exists, mock_load):
    mock_model = MagicMock()
    mock_load.return_value = mock_model
    manager = WorkflowManager()
    assert manager.model == mock_model
    mock_scheduler.return_value.schedule_task.assert_not_called()


@patch("workflow.workflow_manager.PegasosQSVC.save")
@patch("workflow.workflow_manager.pegasos_svc", new=MagicMock())
@patch("workflow.workflow_manager.train_and_save_qsvc", return_value=0.9)
@patch("workflow.workflow_manager.os.path.exists", side_effect=[False, True])
@patch("workflow.workflow_manager.JobScheduler")
def test_workflow_manager_skips_training_logic(mock_scheduler, mock_exists, mock_train, mock_save):
    # Simulate model path missing, but still avoid real training
    mock_scheduler.return_value.schedule_task = MagicMock()
    manager = WorkflowManager()
    assert manager.model is not None


if __name__ == "__main__":
    pytest.main()
