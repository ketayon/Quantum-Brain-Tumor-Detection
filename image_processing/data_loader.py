import os
import numpy as np
import logging
from PIL import Image
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

dataset_path_tumor = "brain_tumor_dataset/yes"
dataset_path_normal = "brain_tumor_dataset/no"

# Clean up .DS_Store if exists
for folder in [dataset_path_tumor, dataset_path_normal]:
    ds_store_path = os.path.join(folder, ".DS_Store")
    if os.path.exists(ds_store_path):
        os.remove(ds_store_path)
        log.info(f"Removed .DS_Store from {folder}")

img_size = (256, 256)


def load_images_from_folder(folder, label, target_size=img_size):
    data, labels = [], []
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        try:
            img = Image.open(img_path).convert("RGB").resize(target_size)
            data.append(np.array(img).flatten())
            labels.append(label)
        except Exception as e:
            log.error(f"Error loading image {img_name}: {e}")
    return data, labels


def load_and_limit_data(path, label, num_samples, target_size=img_size):
    data, labels = load_images_from_folder(path, label, target_size)
    indices = np.random.choice(len(data), min(num_samples, len(data)), replace=False)
    data = [data[i] for i in indices]
    labels = [labels[i] for i in indices]
    return data, labels


tumor_data, tumor_labels = load_and_limit_data(dataset_path_tumor, label=1, num_samples=155)
normal_data, normal_labels = load_and_limit_data(dataset_path_normal, label=0, num_samples=98)

if len(tumor_data) == 0 and len(normal_data) == 0:
    log.error("No images found! Cannot proceed with train-test split.")
    exit(1)

all_data = tumor_data + normal_data
all_labels = tumor_labels + normal_labels
log.info(f"Total images loaded: {len(all_data)}")

if len(all_data) > 1:
    X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.2, random_state=42)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    log.info(f"Dataset loaded successfully: {len(X_train)} train samples, {len(X_test)} test samples.")
else:
    log.error("Not enough images to split. Ensure the dataset is correctly located.")
    exit(1)


def count_images(directory):
    return len([f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
