# 🩺 Chest X-ray Pneumonia Detection

A deep learning–based medical imaging project to classify **Chest X-ray images** as **NORMAL** or **PNEUMONIA** using CNNs built from scratch. The project emphasizes correct ML practices, reproducibility, and honest evaluation over chasing raw accuracy.

---

## 📌 Overview

Pneumonia is a potentially life-threatening lung infection where early detection is critical. This project builds an end-to-end image classification pipeline and systematically compares four strategies for handling class imbalance — a problem native to this dataset (~1:3 Normal-to-Pneumonia ratio in training).

**Phase 1 (Completed — CNN from scratch, no transfer learning.**

---

## 🎯 Objectives

- Understand and preprocess real-world medical image data
- Reconstruct a representative validation set from the original Kaggle data
- Build CNNs from scratch without transfer learning
- Systematically compare four class-imbalance handling strategies
- Prioritize **Recall (Pneumonia)** to minimise missed diagnoses, while preserving **Normal recall** to avoid over-prediction
- Document findings honestly, including failure modes

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
│   ├── 01_data_exploration.ipynb              # EDA + val set reconstruction
│   ├── 02_model_training_class_weights.ipynb  # Strategy 1 — weighted loss
│   ├── 02_model_training_OverSample.ipynb     # Strategy 2 — disk-level oversampling
│   ├── 02_model_training_k_fold.ipynb         # Strategy 3 — StratifiedKFold ensemble
│   ├── 02_model_training_k_fold_custom.ipynb  # Strategy 4 — Custom KFold implementation
│   └── 03_evaluation.ipynb                    # Unified evaluation & comparison table
│
├── src/                                       # (Phase 2) reusable scripts
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── data_loader.py
│
├── models/                                    # Saved .h5 files (excluded from git)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

**Chest X-Ray Images (Pneumonia)** — Kaggle
→ https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

### Class Distribution (after val reconstruction)

| Split | NORMAL | PNEUMONIA |
|-------|--------|-----------|
| Train | 1,199  | 3,480     |
| Val   | 150    | 403       |
| Test  | 234    | 390       |

> ⚠️ **Val set note:** The original Kaggle `val/` directory had only 16 images per class — too small for reliable callback signals (EarlyStopping, ReduceLROnPlateau). In `01_data_exploration.ipynb`, 10% of the training set was stratified-split and moved to `val/` before any model training. The class imbalance in val is intentional — it mirrors real-world distribution.

---

## 📥 Dataset Setup

Download from Kaggle and extract into:

```
data/raw/train/NORMAL/
data/raw/train/PNEUMONIA/
data/raw/val/NORMAL/
data/raw/val/PNEUMONIA/
data/raw/test/NORMAL/
data/raw/test/PNEUMONIA/
```

Run `01_data_exploration.ipynb` first — it automatically reconstructs the val split from the training set.

---

## 🏗️ Phase 1 — Architecture

### Strategies 1 & 2 (Class Weights, Oversampling)
A deeper 4-block CNN with Global Average Pooling:

```
Input (224×224×1 grayscale)
  → Conv2D(32)  + BN + ReLU + MaxPool + Dropout(0.1)
  → Conv2D(64)  + BN + ReLU + MaxPool + Dropout(0.2)
  → Conv2D(128) + BN + ReLU + MaxPool + Dropout(0.2)
  → Conv2D(256) + BN + ReLU + MaxPool + Dropout(0.3)
  → GlobalAveragePooling2D
  → Dense(256, relu) + Dropout(0.4)
  → Dense(128, relu) + Dropout(0.3)
  → Dense(1, sigmoid)
```

### Strategies 3 & 4 (K-Fold variants)
A lighter 2-block CNN with Flatten (3 instances trained per strategy):

```
Input (224×224×1 grayscale)
  → Conv2D(32) + BN + ReLU + MaxPool + Dropout(0.1)
  → Conv2D(64) + BN + ReLU + MaxPool + Dropout(0.2)
  → Flatten
  → Dense(128, relu) + Dropout(0.3)
  → Dense(1, sigmoid)
```

**Shared training config:**
- Optimizer: Adam (lr = 0.0001)
- Loss: Binary Crossentropy
- Metrics: Accuracy, Precision, Recall
- Callbacks: EarlyStopping + ReduceLROnPlateau + ModelCheckpoint

---

## ⚖️ Imbalance Strategies Compared

### Strategy 1 — Class Weights
Adjusted class weights `{Normal: 1.6, Pneumonia: 0.8}` passed to `model.fit()`. All callbacks monitor `val_loss`. The weights were manually tuned down from the fully-balanced sklearn defaults (~1.95 / 0.67) to avoid over-correcting.

### Strategy 2 — Disk-level Oversampling
Augmented copies of Normal images written to disk using OpenCV + `ImageDataGenerator` until Normal count matched Pneumonia (1,199 → 3,480). Online augmentation applied during training via a separate `train_datagen`. Augmented files deleted after training to keep the repo clean.

### Strategy 3 — K-Fold with Balanced Batch Generator (sklearn StratifiedKFold)
A custom generator samples exactly `batch_size/2` Positives and Negatives per batch (with replacement), removing dataset-level imbalance without copying files. Three folds trained via `sklearn.StratifiedKFold`. Final predictions are the soft-vote average of all three fold models.

### Strategy 4 — K-Fold with Custom StratifiedKFold
Identical training setup to Strategy 3 but with a hand-written `custom_stratified_kfold_split()` function replacing `sklearn.StratifiedKFold`. Implemented to validate understanding of the stratification algorithm and confirm that results are reproducible without the sklearn dependency.

---

## 📈 Phase 1 Results

All models evaluated on the same held-out test set (234 Normal, 390 Pneumonia). Classification threshold optimised per model by searching `[0.20, 0.80]` to maximise Pneumonia F1-score.

| Strategy | Threshold | Normal Recall | Pneumonia Recall | Accuracy | F1 (Pneumonia) | Macro F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Class Weights | 0.79 | 0.48 | 0.95 | 0.77 | 0.84 | 0.73 |
| Oversampling | 0.66 | 0.58 | 0.96 | 0.82 | 0.87 | 0.79 |
| KFold Ensemble (sklearn) | 0.79 | 0.82 | 0.92 | 0.88 | 0.91 | 0.87 |
| **KFold Ensemble (custom)** | **0.76** | **0.83** | **0.93** | **0.89** | **0.91** | **0.88** |

### Key Findings

> **1. Validation set size is foundational.** The original 16-image val set caused EarlyStopping and ReduceLROnPlateau to react to noise rather than signal. Reconstructing a properly-sized val set was the most impactful fix in the entire project.

> **2. Dataset-level balancing alone is insufficient.** Class Weights and Oversampling both improved Pneumonia recall significantly, but at the cost of Normal recall (0.48 and 0.58 respectively). The model sees more Pneumonia examples relative to the signal it gets from Normal, regardless of dataset-level adjustments.

> **3. Batch-level balancing is more effective.** The K-Fold strategies, which guarantee equal Normal/Pneumonia sampling *within each batch*, achieved the best balance — Normal recall 0.82–0.83, Pneumonia recall 0.92–0.93 — without requiring any copies to be written to disk.

> **4. Custom KFold matched sklearn KFold exactly.** Strategy 4 reproduced Strategy 3's results within 0.001, confirming correctness of the hand-written stratification logic.

---

## 🛠️ Tech Stack

- Python 3.10
- TensorFlow / Keras 2.20.0
- NumPy, Pandas, Matplotlib
- OpenCV (cv2)
- Pillow (PIL)
- scikit-learn

---

## 🚀 Roadmap

### ✅ Phase 1 — CNN from Scratch (Complete)
- [x] EDA + val set reconstruction from original Kaggle data
- [x] Class Weights model (4-block CNN, GAP)
- [x] Disk-level Oversampling model (4-block CNN, GAP)
- [x] K-Fold ensemble with balanced batch generator (sklearn split)
- [x] K-Fold ensemble with custom stratified split implementation
- [x] Unified evaluation: threshold search, classification report, confusion matrix, comparison table

---

### 🔜 Phase 2 — Transfer Learning + Explainability (Planned)

**Goal:** Replace the scratch CNN backbone with a pretrained feature extractor and understand *what* the model is actually looking at.

- [ ] Adapt a pretrained backbone (MobileNetV2 or EfficientNetB0) for single-channel X-ray input
- [ ] Fine-tune with progressive unfreezing (freeze base → train head → unfreeze top blocks)
- [ ] Grad-CAM heatmaps to visualise which lung regions drive predictions
- [ ] Head-to-head comparison against Phase 1 metrics (same test set, same threshold search)
- [ ] Refactor training logic into reusable `src/` scripts (`data_loader.py`, `model.py`, `train.py`, `evaluate.py`)

---

### 🔮 Phase 3 — PyTorch + Vision Transformers (Planned)

**Goal:** Reproduce the best Phase 2 result in PyTorch and explore attention-based architectures.

- [ ] Port the dataset pipeline and evaluation loop to PyTorch / torchvision
- [ ] Implement a Vision Transformer (ViT) for patch-based X-ray classification
- [ ] Compare CNN vs. ViT on the same test set
- [ ] Explore cross-architecture ensemble (CNN + ViT soft-vote)

---

### 🌍 Phase 4 — Geospatial Land Classification (Planned)

**Goal:** Extend the capstone to a multi-class remote sensing task using satellite imagery.

- [ ] Geospatial land-use / land-cover classification dataset (e.g. EuroSAT or UC Merced)
- [ ] Adapt multi-channel (RGB + NIR) input pipelines
- [ ] Apply transfer learning and compare with the medical imaging findings
- [ ] Evaluate with macro F1 and per-class breakdowns across 10+ classes

---

## 📜 Disclaimer

This project is strictly for **educational and research purposes**. It must not be used for real-world medical diagnosis without rigorous clinical validation and regulatory approval.
