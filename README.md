# 🩺 Chest X-ray Pneumonia Detection

A deep learning–based medical imaging project to detect **Pneumonia** from **Chest X-ray images** using **Convolutional Neural Networks (CNNs)**.

---

## 📌 Overview

Pneumonia is a potentially life-threatening lung infection where **early detection is critical**.  
This project builds an **end-to-end image classification pipeline** that classifies chest X-ray images into:

- **NORMAL**
- **PNEUMONIA**

The project is designed for **educational and research purposes**, focusing on correct ML practices rather than just high accuracy.

---

## 🎯 Objectives

- Understand and preprocess medical image data  
- Build a **CNN from scratch** (no transfer learning in Version-1)  
- Prioritize **Recall** to minimize missed pneumonia cases  
- Perform evaluation and error analysis  
- Maintain a clean, reproducible ML project structure  

---

## 📁 Project Structure

```
chest-xray-pneumonia-detection/
│
├── data/               # Dataset (excluded from git)
├── notebooks/          # EDA, preprocessing, training, evaluation
├── src/                # Reusable training & evaluation scripts
├── models/             # Saved trained models (excluded)
├── reports/            # Metrics, plots, confusion matrices
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

This project uses the **Chest X-Ray Images (Pneumonia)** dataset.

### Classes
- **NORMAL**
- **PNEUMONIA**

⚠️ The dataset is **not included** in this repository due to size constraints.

---

## 📥 Dataset Setup

### Download from Kaggle (Recommended)

1. Visit:  
   https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

2. Download and extract the dataset

3. Place it inside the `data/` directory as shown below:

```
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

---

## 🛠️ Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy, Pandas  
- Matplotlib  
- OpenCV  

---

## 📈 Evaluation Metrics

- Accuracy  
- **Recall (PNEUMONIA)** — prioritized due to medical risk  
- Confusion Matrix  

---

## 🚧 Project Status

**Version-1 (CNN from scratch): In progress** 🚧  

Planned future versions:
- **Version-2:** Transfer learning + Grad-CAM explainability  
- **Version-3:** Decision-support framing with risk scoring  

---

## 📜 Disclaimer

This project is intended **strictly for educational and research purposes**.  
It should **not** be used for real-world medical diagnosis without professional validation.
