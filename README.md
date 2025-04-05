# Diabetes Prediction Pipeline - Project Overview

## 🌟 Objective
Predict whether a person has diabetes based on medical features using various machine learning models with the help of a clean and scalable pipeline.

---

## 📈 Dataset Used
- **Name:** Pima Indians Diabetes Dataset
- **Source:** UCI Machine Learning Repository
- **Target Variable:** `Outcome` (1 = Diabetic, 0 = Not Diabetic)
- **Features Include:**
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI
  - Diabetes Pedigree Function
  - Age

---

## 📅 Steps Followed

### 1. **Data Loading**
Data loaded using `pandas.read_csv()`

### 2. **Exploratory Data Analysis (EDA)**
- Used `seaborn` and `matplotlib` for visualization
- Checked for nulls, distributions, and correlation

### 3. **Data Preprocessing**
- Standardization using `StandardScaler`
- Dimensionality Reduction using `PCA`

### 4. **Train-Test Split**
- Used `train_test_split` with 80-20 ratio

### 5. **Model Building with Pipeline**
- Created machine learning pipelines using `sklearn.pipeline.Pipeline`
- Models used:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - AdaBoost
  - Gradient Boosting
  - XGBoost
  - K-Nearest Neighbors (KNN)

### 6. **Model Evaluation**
- Compared accuracy scores
- Plotted model-wise accuracy using bar chart

### 7. **Best Model Selection**
- Selected the best-performing model based on accuracy score

---

## 🤖 Tools and Libraries Used
- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- xgboost

---

## ⚡ How to Run
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run the notebook:
```bash
jupyter notebook pipeline.ipynb
```

---

## 🚀 Outcomes
- Built a scalable ML pipeline for medical diagnosis.
- Achieved high accuracy with ensemble models.
- Improved performance using feature scaling and PCA.

---

## 📄 Author
**Sundram Tiwari**

