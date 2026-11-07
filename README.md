
# Breast Cancer Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to develop and compare five machine learning classification algorithms to classify breast cancer cases as Benign or Malignant.

The project also provides an interactive Streamlit application for evaluating trained models using uploaded test data.

## 2. Dataset Description

Dataset: Breast Cancer Wisconsin (Diagnostic)

Source: UCI Machine Learning Repository

Source URL:
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

- Total Instances: 569
- Total Features: 30
- Target Classes: 2
- Benign: 357
- Malignant: 212
- Training Records: 455
- Testing Records: 114
- Train-Test Split: 80:20

Target encoding:
- 0 = Benign
- 1 = Malignant

## 3. GitHub Repository Link

REPLACE_WITH_YOUR_GITHUB_REPOSITORY_URL

## 4. Machine Learning Models and Evaluation

Five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

### Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Model Performance Observations

| Model | Observation |
|---|---|
| Logistic Regression | Achieved 96.49% accuracy and excellent AUC of 0.9960. |
| Decision Tree | Achieved 92.11% accuracy with comparatively lower recall. |
| KNN | Delivered strong performance with 95.61% accuracy. |
| Naive Bayes | Achieved perfect precision but comparatively lower recall. |
| Support Vector Machine | Achieved the highest accuracy, F1 score and MCC among the five models. |

### Overall Winner

Support Vector Machine (SVM) is the best-performing model on the selected test dataset.

- Accuracy: 97.37%
- AUC: 0.9947
- Precision: 1.0000
- Recall: 0.9286
- F1 Score: 0.9630
- MCC: 0.9442

## 5. Streamlit Application

The Streamlit application provides:

- CSV test dataset upload.
- Classification model selection.
- Evaluation metric display.
- Confusion matrix.
- Classification report.
- Model comparison.

The application loads previously trained models without retraining on uploaded test data.

Live Application:
REPLACE_WITH_YOUR_STREAMLIT_APP_URL

## 6. Project Structure

```text
ML_Classification_Assignment_2/
|
|-- app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- ML_Classification_Models.ipynb
|
|-- model/
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- naive_bayes.joblib
|   |-- support_vector_machine.joblib
|
|-- results/
    |-- comparison.csv
