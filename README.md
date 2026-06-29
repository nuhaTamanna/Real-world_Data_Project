# Real-world Data Project: Heart Disease Prediction Using Machine Learning

## Introduction

Heart disease is one of the leading causes of death worldwide and remains a significant public health concern. Early identification of individuals at risk can help healthcare professionals provide timely diagnosis and treatment, ultimately improving patient outcomes.

This project explores the UCI Heart Disease dataset through data preprocessing, exploratory data analysis (EDA), visualization, and predictive modeling. The primary objective is to analyze the factors associated with heart disease and develop machine learning models capable of predicting the presence of heart disease based on patient medical information.

The project demonstrates essential data science skills including:

* Data collection and preprocessing
* Handling missing values
* Exploratory Data Analysis (EDA)
* Data visualization
* Feature scaling
* Machine Learning model development
* Model evaluation and comparison
* Interpretation of feature importance
* Healthcare data analysis

---

# Dataset Information

**Dataset:** UCI Heart Disease Dataset

**Source:** UCI Machine Learning Repository

The dataset contains medical information collected from patients undergoing heart disease diagnosis.

### Dataset Features

| Feature  | Description                                    |
| -------- | ---------------------------------------------- |
| age      | Age of the patient                             |
| sex      | Gender (1 = Male, 0 = Female)                  |
| cp       | Chest pain type                                |
| trestbps | Resting blood pressure                         |
| chol     | Serum cholesterol level                        |
| fbs      | Fasting blood sugar                            |
| restecg  | Resting electrocardiographic results           |
| thalach  | Maximum heart rate achieved                    |
| exang    | Exercise-induced angina                        |
| oldpeak  | ST depression induced by exercise              |
| slope    | Slope of peak exercise ST segment              |
| ca       | Number of major vessels colored by fluoroscopy |
| thal     | Thalassemia type                               |
| num      | Diagnosis of heart disease (Target Variable)   |

### Dataset Size

* Rows: 303
* Columns: 14

---

# Tools and Technologies Used

## Programming Language

* Python

## Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* UCI ML Repository (ucimlrepo)

## Development Environment

* Jupyter Notebook

---

# Requirements

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn ucimlrepo
```

---

# Project Structure

```
Heart_Disease_Prediction/
│
├── Heart_Disease_Analysis.ipynb
├── README.md
└── images/
    ├── heart_disease_distribution.png
    ├── age_distribution.png
    ├── gender_distribution.png
    ├── chest_pain_distribution.png
    ├── heart_disease_by_gender.png
    ├── cholesterol_boxplot.png
    ├── blood_pressure_boxplot.png
    ├── maximum_heart_rate_boxplot.png
    ├── correlation_heatmap.png
    ├── pairplot.png
    ├── feature_importance.png
    ├── logistic_confusion_matrix.png
    ├── randomforest_confusion_matrix.png
    └── model_accuracy_comparison.png
```

---

# Data Preprocessing

## Step 1: Loading the Dataset

The dataset was imported directly from the UCI Machine Learning Repository using the `ucimlrepo` package.

```python
heart_disease = fetch_ucirepo(id=45)
```

---

## Step 2: Dataset Overview

The dataset dimensions, feature names, data types, and descriptive statistics were examined to understand the overall structure of the data.

The following methods were used:

```python
df.info()
df.describe()
```

---

## Step 3: Missing Value Analysis

Missing values were identified using:

```python
df.isnull().sum()
```

### Observation

The dataset contained a small number of missing values, primarily in:

* ca
* thal

Since only a few observations were missing, median imputation was used to preserve the dataset.

---

## Step 4: Handling Missing Values

Missing numerical values were replaced with the median value.

```python
df.fillna(df.median(numeric_only=True), inplace=True)
```

---

## Step 5: Duplicate Detection

Duplicate records were checked using:

```python
df.duplicated().sum()
```

### Observation

No duplicate records were found in the dataset.

---

## Step 6: Target Variable Transformation

The original target variable contained five categories (0–4).

To simplify the prediction task, the target variable was converted into binary classes:

* 0 → No Heart Disease
* 1–4 → Heart Disease

```python
df["num"] = df["num"].apply(lambda x: 0 if x == 0 else 1)
```

---

# Exploratory Data Analysis

## 1. Heart Disease Distribution

### Visualization

Count Plot

### Observation

The dataset contains both healthy individuals and patients diagnosed with heart disease.

### Insight

The dataset is reasonably balanced for binary classification.

---

## 2. Age Distribution

### Visualization

Histogram

### Observation

Most patients fall within the middle-aged to older adult population.

### Insight

Heart disease risk tends to increase with age.

---

## 3. Gender Distribution

### Visualization

Count Plot

### Observation

Male patients represent a larger proportion of the dataset.

### Insight

The dataset reflects the higher prevalence of heart disease among males.

---

## 4. Chest Pain Type

### Visualization

Count Plot

### Observation

Different chest pain categories appear with varying frequencies.

### Insight

Chest pain type is an important clinical indicator of heart disease.

---

## 5. Heart Disease by Gender

### Visualization

Count Plot

### Observation

Heart disease is more frequently observed among male patients.

### Insight

Gender appears to influence heart disease prevalence.

---

## 6. Cholesterol Analysis

### Visualization

Box Plot

### Observation

Patients diagnosed with heart disease generally exhibit greater variability in cholesterol levels.

### Insight

High cholesterol is an important cardiovascular risk factor.

---

## 7. Resting Blood Pressure

### Visualization

Box Plot

### Observation

Blood pressure values vary across both classes.

### Insight

Elevated resting blood pressure may contribute to increased heart disease risk.

---

## 8. Maximum Heart Rate

### Visualization

Box Plot

### Observation

Patients without heart disease generally achieve higher maximum heart rates during exercise.

### Insight

Reduced exercise capacity may indicate cardiovascular problems.

---

## 9. Correlation Analysis

### Visualization

Correlation Heatmap

### Observation

Several clinical variables show meaningful relationships with the target variable.

### Insight

Features such as chest pain type, oldpeak, exercise-induced angina, and maximum heart rate demonstrate stronger correlations with heart disease.

---

## 10. Pairplot

### Visualization

Pairplot

### Observation

Relationships among age, cholesterol, blood pressure, maximum heart rate, and oldpeak become visually apparent.

### Insight

Pairwise analysis helps identify separability between healthy individuals and heart disease patients.

---

# Machine Learning

## Feature Scaling

Numerical features were standardized using StandardScaler to improve Logistic Regression performance.

---

## Logistic Regression

A Logistic Regression classifier was trained using the processed dataset.

### Evaluation Metrics

* Accuracy
* Classification Report
* Confusion Matrix

---

## Random Forest Classifier

A Random Forest classifier was developed to compare predictive performance with Logistic Regression.

### Evaluation Metrics

* Accuracy
* Classification Report
* Confusion Matrix

---

## Feature Importance

Random Forest feature importance was analyzed to identify the variables contributing most significantly to prediction.

---

# Model Comparison

Both machine learning models were evaluated using accuracy scores.

The comparison provides insights into which algorithm performs better for predicting heart disease.

---

# Key Findings

* The dataset required minimal preprocessing.
* Missing values were successfully handled using median imputation.
* No duplicate records were identified.
* Age is an important factor associated with heart disease.
* Chest pain type showed a strong relationship with the diagnosis.
* Patients with lower maximum heart rates were more likely to have heart disease.
* Exercise-induced angina and ST depression were influential predictors.
* Random Forest achieved slightly higher prediction accuracy than Logistic Regression.
* Feature importance analysis identified chest pain type, oldpeak, thalach, and age as some of the most influential features.

---

# Conclusion

This project demonstrates a complete end-to-end healthcare data science workflow using the UCI Heart Disease dataset. Data preprocessing, exploratory data analysis, visualization, feature engineering, and machine learning techniques were successfully applied to understand the factors associated with heart disease.

Both Logistic Regression and Random Forest classifiers were evaluated for predictive performance, with Random Forest generally achieving higher accuracy. The analysis revealed that clinical variables such as chest pain type, maximum heart rate, exercise-induced angina, and ST depression are among the strongest predictors of heart disease.

Overall, this project highlights the practical application of data science techniques in healthcare and demonstrates how predictive models can assist medical professionals in identifying individuals at higher risk of heart disease. While these models provide valuable insights, they should be considered decision-support tools rather than replacements for clinical expertise.
