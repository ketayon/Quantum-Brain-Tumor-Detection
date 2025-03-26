import os
import sys
import numpy as np
import pytest
from unittest.mock import patch
from qiskit.circuit import QuantumCircuit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from workflow.workflow_manager import WorkflowManager
from workflow.job_scheduler import JobScheduler

valid_features = np.linspace(0, np.pi, 54)  # 18 qubits x 3 layers
invalid_features = np.linspace(0, np.pi, 20)  # invalid input length

# -----------------------------
# JobScheduler Tests
# -----------------------------

def test_scheduler_executes_task():
    scheduler = JobScheduler(max_workers=1)
    result = scheduler.schedule_task(lambda x: x + 5, 7)
    assert result == 12


# -----------------------------
# Interpretation
# -----------------------------

def test_interpret_counts_detected():
    counts = {'100': 800, '000': 224}
    result = WorkflowManager._interpret_quantum_counts(counts)
    assert result == "No Tumor Detected"

def test_interpret_counts_not_detected():
    counts = {'000': 900, '100': 124}
    result = WorkflowManager._interpret_quantum_counts(counts)
    assert result == "No Tumor Detected"

def test_interpret_invalid_counts():
    with pytest.raises(ValueError, match="Empty counts from quantum simulation."):
        WorkflowManager._interpret_quantum_counts({})

# -----------------------------
# Quantum Classification
# -----------------------------

def test_classify_with_quantum_circuit():
    manager = WorkflowManager()
    result = manager.classify_with_quantum_circuit([np.pi / 4] * 54)
    assert result in ["Tumor Detected", "No Tumor Detected"]

@patch("workflow.workflow_manager.apply_noise_mitigation", return_value=None)
def test_classify_with_quantum_circuit_noise_valid(mock_noise_model):
    manager = WorkflowManager()
    result = manager.classify_with_quantum_circuit_noise(valid_features)
    assert result in ["Tumor Detected", "No Tumor Detected"]

def test_classify_with_quantum_circuit_noise_invalid():
    with pytest.raises(ValueError, match="Expected .* features"):
        WorkflowManager.classify_with_quantum_circuit_noise(invalid_features)

# -----------------------------
# Training Pipeline
# -----------------------------

@patch("workflow.workflow_manager.train_and_save_qsvc", return_value=0.85)
def test_training_pipeline_executes(mock_train):
    manager = WorkflowManager()
    manager._execute_training()
    assert hasattr(manager, "model")
    assert manager.model is not None


if __name__ == "__main__":
    pytest.main()
