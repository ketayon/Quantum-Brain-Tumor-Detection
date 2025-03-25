from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit.circuit import Parameter
from .quantum_circuit import build_ansatz, calculate_total_params


__all__ = ["kernel", "ansatz"]

num_qubits = 6
total_params = calculate_total_params(num_qubits)
params = [Parameter(f'θ{i}') for i in range(total_params)]
ansatz = build_ansatz(num_qubits, params)

sampler = Sampler()
fidelity = ComputeUncompute(sampler=sampler)
kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=ansatz)