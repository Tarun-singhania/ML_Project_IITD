# 🍷 Red Wine Quality Prediction using Machine Learning

This repository contains a complete machine learning analysis of the **Red Wine Quality dataset**, covering both **regression** and **classification** tasks.  
The project demonstrates data preprocessing, model implementation, evaluation, hyperparameter tuning, and handling of imbalanced datasets.

---

## 📌 Project Overview

The objective of this project is to predict and analyze wine quality based on physicochemical properties such as acidity, alcohol content, and sulphates.

The project is divided into two parts:

1. **Regression** – Predicting the exact wine quality score  
2. **Classification** – Classifying wine as *Good* or *Poor* quality  

Both parts include custom implementations and standard machine learning models.

---

## 📂 Repository Structure
├── ML_Part_1_red_wine.py # Regression analysis
├── ML_Part_2_red_wine.py # Classification analysis
├── train.csv # dataset
├── Report # Overall Report
├── README.md # Project documentation


---

## 📊 Dataset Information

- **Dataset:** Red Wine Quality Dataset (UCI ML Repository)
- **Original Size:** 1599 samples × 12 features
- **After Cleaning:** 1359 samples (240 duplicate rows removed)
- **Target Variable:**
  - Regression: Wine quality score (0–10)
  - Classification:
    - `1` → Good Quality (quality ≥ 7)
    - `0` → Poor Quality (quality < 7)

---

## 🧪 Part 1: Regression Analysis (`ML_Part_1_red_wine.py`)

### 🔧 Models Implemented
- Custom **KNN Regressor** (from scratch)
- **Linear Regression**
- **Ridge Regression**
- **Random Forest Regressor**

### ⚙️ Preprocessing Steps
- Duplicate removal
- Train–Test Split: **70:30**
- Feature normalization using **PowerTransformer (Yeo–Johnson)**

### 📐 Evaluation Metrics
- R² Score
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- 5-Fold Cross-Validation Score

### 🏆 Key Results
- **Best Overall Model:** Random Forest Regressor
- **Best Distance-Based Model:** KNN Regressor
- **Best Linear Model:** Ridge Regression
- Hyperparameter tuning significantly improved KNN performance

---

## 🧠 Part 2: Classification Analysis (`ML_Part_2_red_wine.py`)

### 🔧 Models Implemented
- Custom **KNN Classifier** (from scratch)
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Support Vector Classifier (SVC)**

### ⚙️ Preprocessing Steps
- Binary label creation (Good / Poor)
- Train–Test Split: **80:20**
- Feature transformation using **PowerTransformer**
- Dimensionality reduction using **PCA**
- Class imbalance handling using **SMOTE + ENN**

### 📐 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Cross-Validation Score

### 🏆 Key Results after handled imbalanced dataset
- **Highest Accuracy:** Random Forest Classifier
- **Best ROC-AUC:** Random Forest Classifier
- **KNN Performance:** Improved after applying PCA
- **After SMOTE + ENN:**
  - Accuracy: **87.13%**
  - Macro F1-score: **0.76**
  - Recall for minority class improved significantly

---

## ⚖️ Class Imbalance Analysis

The dataset is highly imbalanced:

- Poor Quality: ~86%
- Good Quality: ~14%

After applying **SMOTE + ENN**:
- Minority class recall increased significantly
- Overall accuracy and F1-score decreased
- Highlights the trade-off between accuracy and minority-class sensitivity

---

## 🚀 How to Run the Code

```bash
# Run regression analysis
python ML_Part_1_red_wine.py

# Run classification analysis
python ML_Part_2_red_wine.py

