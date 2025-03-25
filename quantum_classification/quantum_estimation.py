import logging
import os
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
from quantum_classification.quantum_circuit import build_ansatz, calculate_total_params


log = logging.getLogger(__name__)

token = os.getenv("QISKIT_IBM_TOKEN")
if not token:
    raise ValueError("QISKIT_IBM_TOKEN environment variable is not set!")

service = QiskitRuntimeService(
    channel="ibm_quantum",
    instance="ibm-q/open/main",
    token=token
)

backend = service.least_busy(operational=True, simulator=False)
log.info(f"Using IBM Quantum Backend: {backend.name}")


def evaluate_ansatz_expectation(features):
    """
    Evaluate expectation ⟨Z⊗Z⊗...Z⟩ on real IBM backend using EstimatorV2.

    Args:
        features (List[float]): Quantum input features of length num_qubits * layers.

    Returns:
        float: Expectation value.
    """
    num_qubits = 18
    layers = 3
    total_params = calculate_total_params(num_qubits, layers)

    if len(features) != total_params:
        raise ValueError(f"Expected {total_params} features, got {len(features)}")

    params = [Parameter(f"θ{i}") for i in range(total_params)]
    circuit = build_ansatz(num_qubits, params)

    observable = SparsePauliOp("Z" * num_qubits)
    log.info(f"Started running on IBM Quantum ....")
    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=1)
    transpiled_circuit = pass_manager.run(circuit)

    isa_observable = observable.apply_layout(transpiled_circuit.layout)

    estimator = Estimator(mode=backend)

    job = estimator.run([(transpiled_circuit, isa_observable, [features])])
    result = job.result()
    value = float(result[0].data.evs)

    log.info(f"🔬 IBM Quantum Expectation Value: {value:.4f}")
    return value


def predict_with_expectation(features):
    """
    Predict label using expectation value thresholding.

    Returns:
        str: "Tumor Detected" or "No Tumor Detected"
    """
    value = evaluate_ansatz_expectation(features)
    return "Tumor Detected" if value > 0.5 else "No Tumor Detected"
