import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(BASE_DIR)

import tempfile
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import cv2
import logging
from werkzeug.utils import secure_filename
from monai.transforms import ScaleIntensity
from flask import Flask, render_template, request, jsonify
from qiskit_machine_learning.algorithms import PegasosQSVC
from sklearn.metrics import confusion_matrix
from image_processing.data_loader import count_images
from image_processing.image_transformations import apply_grayscale, apply_gaussian_blur
from image_processing.dimensionality_reduction import (
    X_train_reduced, X_test_reduced,
    MinMaxScaler, reduce_to_n_dimensions
)
from image_processing.data_loader import (
    y_train, y_test,
    dataset_path_tumor, dataset_path_normal
)
from workflow.workflow_manager import WorkflowManager


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

workflow_manager = WorkflowManager()

MODEL_PATH = "models/PegasosQSVC_Fidelity_quantm_trainer_brain.model"
if os.path.exists(MODEL_PATH):
    loaded_model = PegasosQSVC.load(MODEL_PATH)
    workflow_manager.model = loaded_model
    log.info("Loaded trained QSVC model for Brain Tumor.")
else:
    log.warning("No trained brain tumor model found! Training a new one...")
    workflow_manager.train_quantum_model()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dataset-info")
def dataset_info():
    tumor = count_images(dataset_path_tumor)
    normal = count_images(dataset_path_normal)
    return jsonify({
        "tumor_count": tumor,
        "normal_count": normal
    })


@app.route("/mri-image")
def mri_image():
    if not os.path.exists(dataset_path_tumor):
        return jsonify({"error": "Dataset folder not found!"}), 404

    image_files = [f for f in os.listdir(dataset_path_tumor) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not image_files:
        return jsonify({"error": "No MRI images found!"}), 404

    selected = random.choice(image_files)
    path = os.path.join(dataset_path_tumor, selected)
    log.info(f"Selected MRI Image: {selected}")

    mri_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    scaled = ScaleIntensity(minv=0.0, maxv=1.0)(mri_img.astype(np.float32))
    colored = plt.cm.magma(scaled / np.max(scaled))

    gray_path = os.path.join(STATIC_DIR, "mri_gray.jpg")
    colored_path = os.path.join(STATIC_DIR, "mri_colored.jpg")
    plt.imsave(gray_path, mri_img, cmap="gray")
    plt.imsave(colored_path, colored)

    return jsonify({
        "gray_image": "static/mri_gray.jpg",
        "colored_image": "static/mri_colored.jpg"
    })


@app.route("/pca-plot")
def pca_plot():
    """Generate and save a PCA scatter plot of the dataset."""
    try:
        plt.figure(figsize=(10, 8))
        plt.scatter(
            X_train_reduced[np.where(y_train == 0)[0], 0],
            X_train_reduced[np.where(y_train == 0)[0], 1],
            marker="o",
            color="green",
            label="Normal Brain (Train)"
        )
        plt.scatter(
            X_train_reduced[np.where(y_train == 1)[0], 0],
            X_train_reduced[np.where(y_train == 1)[0], 1],
            marker="x",
            color="red",
            label="Tumor (Train)"
        )

        plt.title("PCA Visualization of Brain MRI Dataset")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend()
        plt.grid(True)

        pca_path = os.path.join(STATIC_DIR, "pca_plot.jpg")
        plt.savefig(pca_path)
        return jsonify({"pca_plot": "static/pca_plot.jpg"})
    except Exception as e:
        log.exception("Error generating PCA plot")
        return jsonify({"error": str(e)}), 500


@app.route("/predict-probabilities")
def predict_probabilities():
    """Generate and save histogram of predicted tumor probabilities."""
    try:
        preds = workflow_manager.model.predict(X_test_reduced[:30])
        probs = torch.tensor(preds, dtype=torch.float32).numpy()

        plt.figure(figsize=(10, 6))
        plt.hist(probs, bins=30, color="purple", alpha=0.7, label="Prediction Scores")
        plt.axvline(0.5, color="black", linestyle="--", label="Decision Threshold (0.5)")
        plt.title("Prediction Probability Distribution")
        plt.xlabel("Prediction Score")
        plt.ylabel("Frequency")
        plt.legend()

        plot_path = os.path.join(STATIC_DIR, "predicted_probs.jpg")
        plt.savefig(plot_path)
        return jsonify({"predicted_probs_plot": "static/predicted_probs.jpg"})
    except Exception as e:
        log.exception("Error generating predicted probabilities")
        return jsonify({"error": str(e)}), 500


@app.route("/confusion-matrix")
def confusion_matrix_plot():
    try:
        preds = workflow_manager.model.predict(X_test_reduced[:30])
        cm = confusion_matrix(y_test[:30], preds)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix - Brain Tumor")
        path = os.path.join(STATIC_DIR, "confusion_matrix.jpg")
        plt.savefig(path)
        return jsonify({"conf_matrix_plot": "static/confusion_matrix.jpg"})
    except Exception as e:
        log.exception("Confusion matrix error")
        return jsonify({"error": str(e)}), 500


@app.route("/classification-score")
def classification_score():
    try:
        train = workflow_manager.model.score(X_train_reduced[:10], y_train[:10])
        test = workflow_manager.model.score(X_test_reduced[:10], y_test[:10])
        return jsonify({
            "train_score": f"{train:.2f}",
            "test_score": f"{test:.2f}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/quantum-circuit-classify", methods=["POST"])
def quantum_circuit_classify():
    try:
        data = request.get_json()
        features = np.array(data.get("features", []), dtype=np.float32)
        pred = workflow_manager.classify_with_quantum_circuit(features)
        return jsonify({"quantum_circuit_prediction": pred})
    except Exception as e:
        log.exception("Quantum Circuit Error")
        return jsonify({"error": str(e)}), 500


@app.route("/classify-image", methods=["POST"])
def classify_uploaded_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files["image"]
    if image.filename == "":
        return jsonify({"error": "No selected image"}), 400

    try:
        temp = os.path.join(tempfile.gettempdir(), secure_filename(image.filename))
        image.save(temp)

        img = cv2.imread(temp)
        if img is None:
            raise ValueError("Invalid or unreadable MRI image.")

        gray = apply_grayscale(img)
        blurred = apply_gaussian_blur(gray)
        resized = cv2.resize(blurred, (256, 256)).flatten().astype(np.float32)

        if np.std(resized) < 1e-3:
            raise ValueError("Uniform image — likely invalid.")

        # Quantum Feature Prep
        num_qubits = 18
        layers = 3
        total_params = num_qubits * layers

        reduced = reduce_to_n_dimensions(resized.reshape(1, -1), num_qubits)
        scaled = MinMaxScaler((0, np.pi)).fit_transform(reduced).flatten()
        features = np.tile(scaled, layers)[:total_params]

        if len(features) != total_params:
            raise ValueError(f"Expected {total_params} features, got {len(features)}")

        prediction = workflow_manager.classify_with_quantum_circuit(features)
        return jsonify({"quantum_prediction": prediction})

    except Exception as e:
        log.exception("Image classification failed")
        return jsonify({"error": f"Invalid MRI image: {str(e)}"}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
