import os
import numpy as np
import pytest
from PIL import Image
import warnings
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit_machine_learning.algorithms import PegasosQSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.utils import algorithm_globals
from image_processing.data_loader import load_images_from_folder, load_and_limit_data
from image_processing.image_transformations import apply_grayscale, apply_gaussian_blur, apply_histogram_equalization
from image_processing.dimensionality_reduction import reduce_to_n_dimensions

algorithm_globals.random_seed = 12345

# Define mock dataset paths for Brain Tumor Classification
test_dataset_path = "./tests/mock_dataset"
mock_tumor_path = os.path.join(test_dataset_path, "tumor")
mock_normal_path = os.path.join(test_dataset_path, "normal")

# Ensure test dataset directories exist
os.makedirs(mock_tumor_path, exist_ok=True)
os.makedirs(mock_normal_path, exist_ok=True)

def create_mock_image(file_path):
    """Creates a dummy grayscale MRI image for testing"""
    img = Image.new('RGB', (256, 256), color='gray')
    img.save(file_path)

# Create test images
for i in range(5):
    create_mock_image(os.path.join(mock_tumor_path, f"tumor_{i}.jpg"))
    create_mock_image(os.path.join(mock_normal_path, f"normal_{i}.jpg"))

def debug_shape(name, array):
    print(f"{name} shape: {array.shape}")
    if len(array) > 0:
        print(f"{name} first row: {array[0]}")
    else:
        print(f"{name} is empty")

def build_ansatz(num_qubits):
    """Build a basic RX ansatz for given qubit count"""
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
    """Test PegasosQSVC training and scoring for brain tumor feature vectors."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

    X_train = np.random.rand(*X_train_shape)
    X_test = np.random.rand(*X_test_shape)
    y_train = np.random.randint(0, 2, X_train_shape[0])
    y_test = np.random.randint(0, 2, X_test_shape[0])

    assert X_train.shape[1] == num_qubits
    assert X_test.shape[1] == num_qubits

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    dynamic_ansatz = build_ansatz(num_qubits)
    kernel = FidelityQuantumKernel(feature_map=dynamic_ansatz)
    model = PegasosQSVC(quantum_kernel=kernel, C=1000, num_steps=100)
    model.fit(X_train_scaled, y_train)
    score = model.score(X_test_scaled, y_test)

    assert 0 <= score <= 1

@pytest.mark.parametrize("folder, label", [(mock_tumor_path, 1), (mock_normal_path, 0)])
def test_load_images_from_folder(folder, label):
    """Tests loading MRI images from tumor & normal folders"""
    data, labels = load_images_from_folder(folder, label)
    assert len(data) == 5
    assert len(labels) == 5
    assert all(lbl == label for lbl in labels)
    assert isinstance(data[0], np.ndarray)

@pytest.mark.parametrize("folder, label, num_samples", [(mock_tumor_path, 1, 3), (mock_normal_path, 0, 2)])
def test_load_and_limit_data(folder, label, num_samples):
    """Tests limiting MRI image loading"""
    data, labels = load_and_limit_data(folder, label, num_samples)
    assert len(data) == num_samples
    assert len(labels) == num_samples
    assert all(lbl == label for lbl in labels)
    assert isinstance(data[0], np.ndarray)

def test_apply_grayscale():
    """Test grayscale transformation on MRI image"""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    gray = apply_grayscale(img)
    assert len(gray.shape) == 2
    assert gray.dtype == np.uint8

def test_apply_gaussian_blur():
    """Test Gaussian blur on MRI image"""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    blurred = apply_gaussian_blur(img)
    assert blurred.shape == img.shape
    assert blurred.dtype == np.uint8

def test_apply_histogram_equalization():
    """Test histogram equalization on MRI image"""
    img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    equalized = apply_histogram_equalization(img)
    assert equalized.shape == img.shape
    assert equalized.dtype == np.uint8

def test_reduce_to_n_dimensions():
    """Test feature reduction for MRI data"""
    mock_data = np.random.rand(10, 64)
    reduced = reduce_to_n_dimensions(mock_data, 8)
    assert reduced.shape == (10, 8)

if __name__ == "__main__":
    pytest.main()
