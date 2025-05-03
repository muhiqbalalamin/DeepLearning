#  Fish Image Classification & Tabular MLP Regression/Classification using PyTorch

This repository contains two deep learning pipelines using PyTorch:
1. **Image Classification of Fish Species** using CNN & Transfer Learning (MobileNetV2)
2. **Regression and Classification** using MLP on tabular data (RegresiUTSTelkom.csv)

---

## 📁 Directory Structure
```
.
├── fish_classification/      # Image classification pipeline
├── mlp_tabular_pipeline/     # MLP for regression and classification
└── README.md
```

---

##  1. Fish Image Classification with CNN & MobileNetV2

### ✅ Features
- Image augmentations (rotation, crop, flip, color jitter)
- Transfer Learning with MobileNetV2 (fine-tuned classifier head)
- Early Stopping and Learning Rate Scheduler
- Evaluation with Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- Visualization of training curves

### 📦 Dataset Structure (ImageFolder Format)
```
FishImgDataset/
├── train/
├── val/
└── test/
```

### 🧠 Models
- **CustomCNN**
- **MobileNetV2** with frozen feature extractor and custom classifier

### 🏋️ Training
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Batch Size: 32
- Input Size: 224x224
- Epochs: 10 with EarlyStopping

### 📊 Evaluation Metrics
- Accuracy
- Precision (macro)
- Recall (macro)
- F1-Score (macro)
- Confusion Matrix

---

##  2. Tabular Data Regression & Classification with MLP

### 🔢 Dataset
- `RegresiUTSTelkom.csv`
- Contains both numerical and categorical features

### ⚙️ Pipeline
- Preprocessing:
  - Missing value imputation
  - Label encoding for classification target
  - StandardScaler normalization
- Split: Train, Validation, Test (60/20/20)

### 🧠 MLP Structure
```python
nn.Sequential(
  nn.Linear(input_dim, 128),
  nn.ReLU(),
  nn.Dropout(0.2),
  nn.Linear(128, 64),
  nn.ReLU(),
  nn.Dropout(0.2),
  nn.Linear(64, 32),
  nn.ReLU(),
  nn.Linear(32, output_dim)
)
```

### 📊 Visualizations
- Training & validation loss
- Confusion matrix (classification)
- Bar chart comparison for evaluation metrics

---


## 🧪 Example Results
### Fish Image Classification (MobileNetV2)
```
Test Accuracy: 0.81875
Precision: 0.8166698700408601
Recall: 0.7868843344317382
F1 Score: 0.7897284544149252
```

### MLP Regression
```
RMSE: 8.6936
MAE: 5.9649
R²: 0.3632
```

### MLP Classification
```
Accuracy: 0.7322
Precision: 0.7134
Recall: 0.7313
F1 Score: 0.7223
AUC-ROC: 0.8069
```

## 📌 Notes
- All models are optimized for Google Colab with T4 GPU
