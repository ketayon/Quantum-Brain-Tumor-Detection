import logging
from qiskit_machine_learning.algorithms import PegasosQSVC
from .kernel_learning import kernel
from image_processing.dimensionality_reduction import X_train_reduced, X_test_reduced
from image_processing.data_loader import y_train, y_test


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

tau = 100
C = 1000

pegasos_svc = PegasosQSVC(quantum_kernel=kernel, C=1000, num_steps=100)

def train_and_save_qsvc():
    pegasos_svc.fit(X_train_reduced, y_train)
    accuracy = pegasos_svc.score(X_test_reduced, y_test)
    log.info(f"PegasosQSVC classification test score: {accuracy:.2f}")
    pegasos_svc.save('models/PegasosQSVC_Fidelity_quantm_trainer_brain.model')
    return accuracy