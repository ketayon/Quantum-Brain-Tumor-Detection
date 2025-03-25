from qiskit_aer.noise import NoiseModel
import logging


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def apply_noise_mitigation(backend):
    """Applies noise mitigation techniques for quantum circuits."""
    noise_model = NoiseModel.from_backend(backend)
    return noise_model

log.info("Noise mitigation module initialized.")