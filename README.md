# 🧠 Quantum AI Brain Tumor Detection

This project is an **end-to-end solution** that integrates **Quantum AI, Computer Vision, and Hybrid Quantum-Classical Models** to detect **brain tumors from MRI scans**. It leverages **Quantum Kernel Learning, Quantum Support Vector Classifiers (QSVC), and Neural Networks** for enhanced medical diagnosis.

---

## 🚀 Features

- 🧠 **Real MRI Image Processing & Augmentation**
- ⚛️ **Quantum-Classical Hybrid Computation**
- ☁️ **IBM Quantum Cloud Integration**
- 🖼️ **Web UI with Real-Time MRI Upload + Visualization**
- 🧪 **CLI Support for Direct Model Execution**
- 🐳 **Dockerized for Seamless Deployment**

---

## 🏗 Solution Architecture

### 🔬 End-to-End Processing Pipeline

1. **MRI Preprocessing**  
   - Loads MRI images and applies grayscale & blur enhancements.  
   - Reduces noise and standardizes shape for analysis.

2. **Quantum Feature Extraction**  
   - Applies custom reduction (or PCA) to condense features.  
   - Scales and maps features to quantum rotation angles.

3. **Quantum Model Training & Classification**  
   - Uses **Fidelity Quantum Kernels** via **QSVC**.  
   - Trained on IBM Quantum backend, with simulation fallback.

4. **Automated Execution**  
   - Modular task management via `JobScheduler` + `WorkflowManager`.

5. **Interactive Web Interface**  
   - Upload your own MRI image and classify it live.  
   - View PCA, predictions, histograms, confusion matrix & more.

---

## 🛠️ Installation Guide

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-repo/Quantum-Brain-Tumor-Detection
cd Quantum-Brain-Tumor-Detection
```

### 2️⃣ **Setup Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate   # MacOS/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## 🔥 Running the System

### **1️⃣ CLI Mode**
```bash
python interfaces/cli.py --dataset-info
python interfaces/cli.py --model-score
python interfaces/cli.py --predict
```
✅ **Output Example:**  
`Quantum QSVC on the training dataset: 0.89`
`Quantum QSVC on the test dataset: 0.82`

---

### **2️⃣ Web Interface**
```bash
python interfaces/web_app/app.py
```
🖥 **Access Web App:**  
🔍 Open **`http://127.0.0.1:5000/`** in a browser.
`Web UI Features:`
`Upload and analyze MRI images`
`View Quantum Model Predictions`
`Visualize MRI Scans and Tumor Probability Heatmaps`

---

## 🐳 Deploying with Docker

### **1️⃣ Build Docker Image**
```bash
docker build -t quantum-brain-tumor .
```

### **2️⃣ Run Container**
```bash
docker run -p 5000:5000 quantum-brain-tumor

if echo "QISKIT_IBM_TOKEN=your_ibm_quantum_token_here" > .env
docker run --env-file .env -p 5000:5000 quantum-brain-tumor
```

🖥 **Access Web App:**  
🔍 Open **`http://127.0.0.1:5000/`**

---

## 🛠️ Development & Testing

### **Run PyTests**
```bash
pytest tests/
```

---

## 💼 IBM Quantum Cloud Integration

**Setup IBM Quantum Account**  
1. Create an account at [IBM Quantum](https://quantum-computing.ibm.com/)
2. Get your API **Token** from **My Account**
3. Set it in your environment:
```bash
export QISKIT_IBM_TOKEN="your_ibm_quantum_token"
```

---
