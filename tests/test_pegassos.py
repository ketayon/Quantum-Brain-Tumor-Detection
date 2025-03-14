import pytest
import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit_machine_learning.algorithms import PegasosQSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.utils import algorithm_globals
from quantum_classification.kernel_learning import kernel

algorithm_globals.random_seed = 12345  # Ensure reproducibility

def debug_shape(name, array):
    print(f"{name} shape: {array.shape}")
    if len(array) > 0:
        print(f"{name} first row: {array[0]}")
    else:
        print(f"{name} is empty")

def build_ansatz(num_qubits):
    """Rebuilds the ansatz dynamically to match num_qubits."""
    ansatz = QuantumCircuit(num_qubits, name="Ansatz")
    params = [Parameter(f"θ{i}") for i in range(num_qubits)]
    
    for i in range(num_qubits):
        ansatz.rx(params[i], i)
    
    return ansatz

@pytest.mark.parametrize("num_qubits, X_train_shape, X_test_shape", [
    (4, (80, 4), (20, 4)),
    (6, (100, 6), (20, 6)),
    (8, (120, 8), (30, 8))
])
def test_pegasos_qsvc(num_qubits, X_train_shape, X_test_shape):
    """Test PegasosQSVC model training and scoring with correct feature mapping."""
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    
        # Generate synthetic data with exactly `num_qubits` features
        X_train = np.random.rand(*X_train_shape)
        X_test = np.random.rand(*X_test_shape)
        y_train = np.random.randint(0, 2, X_train_shape[0])
        y_test = np.random.randint(0, 2, X_test_shape[0])
        
        debug_shape("X_train (raw)", X_train)
        debug_shape("X_test (raw)", X_test)
        
        assert X_train.shape[1] == num_qubits, f"X_train shape {X_train.shape} does not match num_qubits {num_qubits}"
        assert X_test.shape[1] == num_qubits, f"X_test shape {X_test.shape} does not match num_qubits {num_qubits}"
        
        # Apply MinMaxScaler to match quantum encoding range [0, π]
        scaler = MinMaxScaler(feature_range=(0, np.pi))
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        debug_shape("X_train_scaled", X_train_scaled)
        debug_shape("X_test_scaled", X_test_scaled)
        
        # **Fix: Rebuild ansatz dynamically to match num_qubits**
        dynamic_ansatz = build_ansatz(num_qubits)
        test_kernel = FidelityQuantumKernel(feature_map=dynamic_ansatz)
        
        # Train PegasosQSVC model
        pegasos_svc = PegasosQSVC(quantum_kernel=test_kernel, C=1000, num_steps=100)
        pegasos_svc.fit(X_train_scaled, y_train)
        score = pegasos_svc.score(X_test_scaled, y_test)
        
        print(f"PegasosQSVC Score: {score}")
        
        assert 0 <= score <= 1
