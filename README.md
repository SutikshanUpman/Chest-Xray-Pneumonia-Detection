# 🩺 Chest X-ray Pneumonia Detection

A deep learning–based medical imaging project to classify **Chest X-ray images** as **NORMAL** or **PNEUMONIA** using CNNs built from scratch. The project emphasizes correct ML practices, reproducibility, and honest evaluation over chasing raw accuracy.

---

## 📌 Overview

Pneumonia is a potentially life-threatening lung infection where early detection is critical. This project builds an end-to-end image classification pipeline and systematically compares three strategies for handling class imbalance — a problem native to this dataset (1:3 Normal-to-Pneumonia ratio).

**Version 1 (Current) — CNN from scratch, no transfer learning.**

---

## 🎯 Objectives

- Understand and preprocess real-world medical image data
- Build a CNN from scratch without transfer learning
- Systematically compare three class-imbalance strategies
- Prioritize **Recall** to minimize missed pneumonia cases
- Document findings honestly, including failures

---

## 📁 Project Structure

```
chest-xray-pneumonia-detection/
│
├── data/
│   └── raw/
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       ├── val/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       └── test/
│           ├── NORMAL/
│           └── PNEUMONIA/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA + val set reconstruction
│   ├── 02_model_training_class_weights.ipynb
│   ├── 02_model_training_OverSample.ipynb
│   ├── 02_model_training_k_fold.ipynb
│   └── 03_evaluation.ipynb
│
├── src/                                # (planned) reusable scripts
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── data_loader.py
│
├── models/                             # Saved .h5 files (excluded from git)
├── reports/                            # Metrics, plots, confusion matrices
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

**Chest X-Ray Images (Pneumonia)** — Kaggle  
→ https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

### Class Distribution

| Split | NORMAL | PNEUMONIA |
|-------|--------|-----------|
| Train | 1,199  | 3,480     |
| Val   | 150    | 403       |
| Test  | 234    | 390       |

> ⚠️ The original Kaggle val set had only 16 images per class. In `01_data_exploration.ipynb`, 10% of the training set was moved to reconstruct a representative validation set before any model training.

---

## 📥 Dataset Setup

```
data/
└── raw/
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

Download from Kaggle, extract, and run `01_data_exploration.ipynb` first — it will automatically reconstruct the val split.

---

## 🏗️ Version 1 — Architecture

All three models share the same 4-block CNN backbone:

```
Input (224×224×1 grayscale)
  → Conv2D(32) + BN + ReLU + MaxPool + Dropout(0.1)
  → Conv2D(64) + BN + ReLU + MaxPool + Dropout(0.2)
  → Conv2D(128) + BN + ReLU + MaxPool + Dropout(0.2)
  → Conv2D(256) + BN + ReLU + MaxPool + Dropout(0.3)
  → GlobalAveragePooling2D
  → Dense(256, relu) + Dropout(0.4)
  → Dense(128, relu) + Dropout(0.3)
  → Dense(1, sigmoid)
```

Optimizer: Adam (lr=0.0001) | Loss: Binary Crossentropy  
Callbacks: EarlyStopping + ReduceLROnPlateau + ModelCheckpoint (all monitoring `val_loss`)

The K-Fold model uses a lighter backbone (2 conv blocks + Flatten) since it trains 3 separate folds.

---

## ⚖️ Imbalance Strategies Compared

### Strategy 1 — Class Weights
Adjusted class weights (Normal: 1.6, Pneumonia: 0.8) passed to `model.fit()`. Monitors `val_loss`.

### Strategy 2 — Oversampling
Normal images duplicated (with no augmentation) to match 3,480 Pneumonia images. Online augmentation applied during training. Monitors `val_loss`.

### Strategy 3 — K-Fold with Balanced Batches (3-fold)
Custom generator that samples equal Normal/Pneumonia images per batch. No dataset-level balancing. 3 models trained via `StratifiedKFold`; soft-voted ensemble at inference.

---

## 📈 Results — Phase 1

All evaluated on the same held-out test set (234 Normal, 390 Pneumonia) using best F1 threshold search.

| Strategy | Normal Recall | Pneumonia Recall | Accuracy | F1 (Pneumonia) |
|----------|:---:|:---:|:---:|:---:|
| Class Weights | 0.48 | 0.95 | 0.77 | 0.84 |
| Oversampling | 0.02 | 1.00 | 0.63 | 0.77 |
| **K-Fold Ensemble** | **0.82** | **0.92** | **0.88** | **0.91** |

### Key Finding

> *Naive oversampling by duplicating the minority class performed worst (Normal recall: 0.02), consistent with published findings on this dataset. Duplicated images lack visual diversity — the model learns that Normal images are "repetitive" and deprioritizes them. Class weights improved Normal recall to 0.48 but remained biased toward Pneumonia. The K-Fold ensemble with batch-level balancing achieved the best balance, confirming that batch-level balancing is more effective than dataset-level balancing for this problem.*

---

## 🛠️ Tech Stack

- Python 3.10
- TensorFlow / Keras 2.20.0
- NumPy, Pandas, Matplotlib
- OpenCV (cv2)
- scikit-learn

---

## 🚀 Roadmap

### ✅ Version 1 — CNN from Scratch (Complete)
- [x] EDA + val set reconstruction
- [x] Class Weights model
- [x] Oversampling model
- [x] K-Fold ensemble model
- [x] Evaluation with threshold search

### 🔜 Version 2 — Transfer Learning + Explainability
- [ ] Fine-tune a pretrained backbone (MobileNetV2 / EfficientNetB0) on grayscale X-rays
- [ ] Grad-CAM visualizations — highlight regions the model focuses on
- [ ] Compare against Version 1 metrics
- [ ] Structured `src/` scripts (data_loader, model, train, evaluate)

### 🔮 Version 3 — Decision Support
- [ ] Risk scoring output instead of binary classification
- [ ] Confidence thresholding with "refer to specialist" zone
- [ ] Basic Streamlit or Gradio demo

---

## 📜 Disclaimer

This project is strictly for **educational and research purposes**. It must not be used for real-world medical diagnosis without professional validation.
