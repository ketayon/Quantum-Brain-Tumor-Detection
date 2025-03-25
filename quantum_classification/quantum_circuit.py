from qiskit.circuit import QuantumCircuit


def conv_layer(num_qubits, label, params):
    """Create a convolutional layer with RX gates."""
    qc = QuantumCircuit(num_qubits, name=label)
    for i in range(num_qubits):
        qc.rx(params[i], i)
    return qc


def pool_layer(qubits, label):
    """Create a pooling layer with CX gates between adjacent qubits."""
    qc = QuantumCircuit(len(qubits), name=label)
    for i in range(0, len(qubits) - 1, 2):
        qc.cx(qubits[i], qubits[i + 1])
    return qc


def build_ansatz(num_qubits, params):
    """Builds the ansatz ensuring equal layers across all qubits."""
    ansatz = QuantumCircuit(num_qubits, name="Ansatz")
    layer_params = iter(params)
    
    for layer in range(1, 4):
        conv_params = [next(layer_params) for _ in range(num_qubits)]
        ansatz.compose(
            conv_layer(num_qubits, f"c{layer}", conv_params),
            range(num_qubits),
            inplace=True
        )
        ansatz.compose(
            pool_layer(range(num_qubits), f"p{layer}"),
            range(num_qubits),
            inplace=True
        )
    return ansatz


def calculate_total_params(num_qubits, layers=3):
    """Calculate the total number of parameters required for the ansatz."""
    total_params = num_qubits * layers
    return total_params