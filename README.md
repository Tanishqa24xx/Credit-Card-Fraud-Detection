# Credit Card Fraud Detection

An end-to-end ML pipeline to detect fraudulent credit card transactions from a highly imbalanced dataset. Benchmarks three classifiers using precision-recall metrics and deploys the best model as a real-time fraud risk-scoring module.

**Dataset:** [Credit Card Fraud Detection 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023) - Kaggle (`creditcard_2023.csv`)

---

## Problem Statement

Financial institutions process millions of transactions daily. Even a small fraud rate translates to massive losses. This project builds a binary classifier to flag fraudulent transactions using **28 anonymised PCA features (V1–V28)**, transaction amount, and related attributes.

**Key challenge:** Features are PCA components - no domain-specific feature engineering is possible. Model selection and threshold tuning matter more than feature crafting.

---

## Screenshots

### Model Performance Metrics
<p align="center">
  <img width="300" alt="model-performance" src="https://github.com/user-attachments/assets/c3a364c6-7b1d-4229-8697-6554ace64f7c" />
</p>

### Confusion Matrix
<p align="center">
  <img width="300" alt="confusion-matrix" src="https://github.com/user-attachments/assets/34ed325e-624e-459b-a955-8569d54a7717" />
</p>

### ROC Curve
<p align="center">
  <img width="300" alt="roc-curve" src="https://github.com/user-attachments/assets/1009e32c-9a85-4539-92d4-acb850233b97" />
</p>

### Precision-Recall Curve
<p align="center">
  <img width="300" alt="precision-recall-curve" src="https://github.com/user-attachments/assets/2448ba62-ee15-4283-b3aa-31c5ebd9cb9e" />
</p>

### Feature Importance (Permutation-based)
<p align="center">
  <img width="300" alt="feature-imp" src="https://github.com/user-attachments/assets/bc4a46f5-9a5b-46af-9455-d5b61898ac8f" />
</p>

---

## Results

| Metric | HistGradientBoosting |
|---|---|
| Accuracy | 0.9943 |
| Precision | 0.9949 |
| Recall | 0.9937 |
| F1-Score | 0.9943 |
| ROC-AUC | 0.9998 |
| Train ROC-AUC | 0.9998 |
| Train/Test gap | < 0.05 (good generalisation) |

**Confusion Matrix:**

| | Predicted Legitimate | Predicted Fraud |
|---|---|---|
| Actual Legitimate | TN: 56,571 | FP: 292 |
| Actual Fraud | FN: 359 | TP: 56,504 |

---

## Model Benchmarking

Three models were evaluated on the same stratified train/test split using **ROC-AUC** as the primary metric, given the class imbalance.

| Model | Notes | Status |
|---|---|---|
| Logistic Regression | Baseline; fast, interpretable, linear decision boundary | Benchmarked |
| Random Forest | Handles imbalance well; slower on large datasets | Benchmarked |
| HistGradientBoosting | Fastest of the boosting family; native support for missing values | **Deployed** |

**Why HistGradientBoosting won:**
- Standard GradientBoosting was too slow for hyperparameter search across this dataset size
- HistGradientBoosting uses histogram-based splits - significantly faster with no accuracy loss
- Tuned via `RandomizedSearchCV` (5 combinations, 3-fold CV) - GridSearch would have required 729 combinations

**Best hyperparameters**:
```
max_iter:      100
learning_rate: 0.1
max_depth:     5
Best CV ROC-AUC: 0.9998
```

---

## Pipeline

```
Raw CSV (creditcard_2023.csv)
   │
   ▼  Class distribution check · null check · duplicate removal · drop id
Cleaning
   │  Stratified train/test split (80/20) · RobustScaler (handles outliers)
   ▼
Preprocessing
   │  Three models trained · RandomizedSearchCV on HistGradientBoosting
   ▼
Model Training & Tuning
   │  Accuracy · Precision · Recall · F1 · ROC-AUC · Permutation importance
   ▼
Evaluation
   │  Overfitting check (train vs test ROC-AUC gap)
   ▼
Risk Scoring Module
   │  fraud_probability → Low / Moderate / High / CRITICAL tier
   ▼
Fraud Detection Report (per transaction)
```

---

## Dataset

| Property | Value |
|---|---|
| Source | Kaggle - creditcard_2023.csv |
| Features | V1–V28 (PCA-anonymised) + Amount + Class |
| Target | Class (0 = Legitimate, 1 = Fraud) |
| Class imbalance | Heavily skewed toward legitimate transactions |
| Preprocessing | RobustScaler (robust to outliers in Amount) · Stratified split |

---

## Fraud Risk-Scoring Module

The deployed module scores any transaction and returns a 4-tier risk label:

```python
def get_risk_tier(probability):
    if probability < 0.10:   return "🟢 Low Risk"
    elif probability < 0.50: return "🟡 Moderate Risk"
    elif probability < 0.90: return "🟠 High Risk"
    else:                    return "🔴 CRITICAL RISK"
```

**Sample output:**
```
------------------------------------------------------------
🛑 FRAUD DETECTION REPORT
------------------------------------------------------------
ACTUAL STATUS:       FRAUD (True Label: 1)
MODEL PREDICTION:    FRAUD
FRAUD PROBABILITY:   94.23%
RISK TIER:           🔴 CRITICAL RISK
------------------------------------------------------------
Transaction Snapshot (Key Scaled Features):
V14: -3.2411 | V4: 1.8832 | V12: -2.9104
------------------------------------------------------------
```

---

## Tech Stack

| | |
|---|---|
| Language | Python 3 |
| ML | scikit-learn (HistGradientBoosting, LogisticRegression, RandomForest) |
| Tuning | RandomizedSearchCV · 3-fold CV |
| Evaluation | ROC-AUC · Precision-Recall · Confusion Matrix · Permutation Importance |
| EDA | pandas · seaborn · matplotlib |
| Notebook | Jupyter / Google Colab |

---

## Project Structure

```
credit-card-fraud-detection/
├── notebooks/
│   └── fraud_detection_pipeline.ipynb   # Full pipeline - EDA → training → evaluation
├── docs/
│   ├── metrics_bar_chart.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   └── feature_importance.png
├── presentation/
│   └── fraud_detection_slides.pdf       # Project presentation
├── requirements.txt
└── README.md
```

---

## Getting Started

```bash
# 1. Clone
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the dataset
#    Download creditcard_2023.csv from Kaggle and place it in the project root
#    https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023

# 4. Run the notebook
jupyter notebook notebooks/fraud_detection_pipeline.ipynb
```

---

## Requirements

```
scikit-learn
pandas
numpy
matplotlib
seaborn
jupyter
```

---

## Key Findings

- **RobustScaler over StandardScaler:** Transaction `Amount` contains outliers - RobustScaler uses median/IQR and is more stable for fraud data.
- **Stratified split is non-negotiable:** Random splitting on an imbalanced dataset leaks class distribution; stratified split preserves the fraud ratio in both train and test sets.
- **ROC-AUC over accuracy:** With heavy class imbalance, a model predicting "all legitimate" can achieve >99% accuracy. ROC-AUC and precision-recall are the only meaningful metrics.
- **Permutation importance over built-in:** HistGradientBoosting doesn't expose `feature_importances_` directly - permutation importance on the test set is more reliable and less biased anyway.
- **Train/test gap <0.05:** Confirms the model generalises well and isn't overfitting to training fraud patterns.

---

## Project Context

| | |
|---|---|
| Subject | Internal project as part of Data Science Club|
| Period | December 2025 |
| Dataset | Kaggle - Credit Card Fraud Detection 2023 |
