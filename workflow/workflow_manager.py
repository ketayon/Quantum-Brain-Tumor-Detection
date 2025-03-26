import os
import logging
from collections import Counter
from qiskit import transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_machine_learning.algorithms import PegasosQSVC

from quantum_classification.quantum_model import pegasos_svc, train_and_save_qsvc
from quantum_classification.noise_mitigation import apply_noise_mitigation
from workflow.job_scheduler import JobScheduler
from quantum_classification.quantum_circuit import build_ansatz, calculate_total_params
from quantum_classification.quantum_async_jobs import submit_quantum_job, check_quantum_job
from quantum_classification.quantum_estimation import predict_with_expectation

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "PegasosQSVC_Fidelity_quantm_trainer_brain.model")

token = os.getenv("QISKIT_IBM_TOKEN")
if not token:
    raise ValueError("ERROR: QISKIT_IBM_TOKEN environment variable is not set!")

service = QiskitRuntimeService(
    channel="ibm_quantum",
    instance="ibm-q/open/main",
    token=token
)

backend = service.least_busy(operational=True, simulator=False)

class WorkflowManager:
    """Manages the Brain Tumor Quantum Classification Workflow"""

    def __init__(self):
        self.job_scheduler = JobScheduler()
        self.model = None

        log.info("Quantum Brain Tumor Workflow Initialized on Backend: %s", backend)
        self._ensure_model_dir()
        self._load_or_train_model()

    def _ensure_model_dir(self):
        """Ensure the models directory exists, especially in Docker."""
        if not os.path.exists(MODEL_DIR):
            log.warning("Model directory not found. Creating: %s", MODEL_DIR)
            os.makedirs(MODEL_DIR, exist_ok=True)

        # Check if mounted volume exists
        if not os.listdir(MODEL_DIR):
            log.warning("Model directory is empty. If running Docker, mount a volume: -v $(pwd)/models:/app/models")

    def _load_or_train_model(self):
        """Load the trained model if it exists, otherwise train and save"""
        if os.path.exists(MODEL_PATH):
            log.info("📦 Loading pre-trained Quantum Brain Tumor Model...")
            self.model = PegasosQSVC.load(MODEL_PATH)
            log.info("✅ Model loaded successfully.")
        else:
            log.info("⚠️ No pre-trained model found. Training a new model...")
            self.train_quantum_model()
            log.info("💾 Saving trained model to: %s", MODEL_PATH)
            pegasos_svc.save(MODEL_PATH)
            self.model = pegasos_svc
            log.info("✅ Model saved.")

    def train_quantum_model(self):
        log.info("🧠 Scheduling Quantum Model Training...")
        self.job_scheduler.schedule_task(self._execute_training)

    def _execute_training(self):
        log.info("🚀 Executing Quantum Brain Tumor Model Training...")
        accuracy = train_and_save_qsvc()
        self.model = pegasos_svc
        log.info(f"🎯 Training Complete. Accuracy: {accuracy}")

    def classify_brain_mri(self, image_data):
        if self.model is None:
            log.error("❌ No trained model found. Please train the model first.")
            return None
        log.info("📊 Scheduling QSVC-based classification...")
        return self.job_scheduler.schedule_task(self._infer_brain_tumor, image_data)

    def _infer_brain_tumor(self, image_data):
        log.info("🔍 Performing QSVC Classification...")
        prediction = self.model.predict(image_data)
        return prediction

    @staticmethod
    def create_quantum_circuit(features):
        num_qubits = len(features)
        params = [Parameter(f"θ{i}") for i in range(num_qubits)]
        circuit = build_ansatz(num_qubits, params)
        param_dict = dict(zip(params, features))
        qc = circuit.assign_parameters(param_dict)
        qc.measure_all()
        return qc

    @staticmethod
    def run_quantum_classification(qc):
        simulator = AerSimulator()
        transpiled_qc = transpile(qc, simulator)
        result = simulator.run(transpiled_qc, shots=1024).result()
        return result.get_counts()

    @staticmethod
    def _interpret_quantum_counts(counts):
        if not counts:
            raise ValueError("Empty counts from quantum simulation.")
        most_common = Counter(counts).most_common(1)[0][0]
        bit = int(most_common[::-1][0])
        return "Tumor Detected" if bit else "No Tumor Detected"

    @staticmethod
    def classify_with_quantum_circuit_noise(image_features):
        num_qubits, layers = 18, 3
        total_params = num_qubits * layers

        if len(image_features) != total_params:
            raise ValueError(f"Expected {total_params} features, got {len(image_features)}")

        params = [Parameter(f"θ{i}") for i in range(total_params)]
        ansatz = build_ansatz(num_qubits, params)
        qc = ansatz.assign_parameters(dict(zip(params, image_features)))
        qc.measure_all()

        noise_model = apply_noise_mitigation(backend)
        simulator = AerSimulator(noise_model=noise_model)
        transpiled = transpile(qc, simulator)
        result = simulator.run(transpiled, shots=1024).result()
        counts = result.get_counts()

        return WorkflowManager._interpret_quantum_counts(counts)

    @staticmethod
    def classify_with_quantum_circuit(image_features):
        log.info("🧪 Running synchronous expectation-based classification...")
        prediction = predict_with_expectation(image_features)
        log.info(f"🔬 Quantum Estimation Prediction: {prediction}")
        return prediction

    @staticmethod
    def submit_quantum_job_async(image_features):
        log.info("📡 Submitting async quantum job to IBM Quantum...")
        return submit_quantum_job(image_features)

    @staticmethod
    def check_quantum_job_result(job_id):
        log.info(f"🔁 Checking status of job: {job_id}")
        return check_quantum_job(job_id)
