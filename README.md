# Breast Cancer Classification Using Machine Learning

M.Tech AIML | Machine Learning Assignment 2 | BITS Pilani

## Live Demo

- Streamlit application: https://ml-classification-assignment-2-jbb27mfj5nxxgxqjyypts3.streamlit.app/
- GitHub repository: https://github.com/Venugopal-14/ML-Classification-Assignment-2

To try the app, open the link, choose a model in the sidebar and upload `test_data.csv` from this repository. If the app has been idle, Streamlit Cloud may show a "wake up" screen first.

## 1. Problem Statement

The objective of this project is to develop and compare five machine learning classification algorithms to classify breast cancer cases as Benign or Malignant.

The project also provides an interactive Streamlit application for evaluating trained models using uploaded test data.

## 2. Dataset Description

Dataset: Breast Cancer Wisconsin (Diagnostic)

Source: UCI Machine Learning Repository

Source URL:
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

- Total Instances: 569
- Total Features: 30 (numeric, computed from digitized images of breast mass cell nuclei)
- Target Classes: 2
- Benign: 357
- Malignant: 212
- Training Records: 455
- Testing Records: 114
- Train-Test Split: 80:20 (`random_state=42`, stratified)

Target encoding:
- 0 = Benign
- 1 = Malignant

`test_data.csv` contains the 114 test records: 30 feature columns plus a `target` column.

## 3. Machine Learning Models and Evaluation

Five classification algorithms were implemented in `ML_Classification_Models.ipynb`:

| # | Model | Configuration |
|---|---|---|
| 1 | Logistic Regression | StandardScaler + LogisticRegression (`max_iter=2000`) |
| 2 | Decision Tree Classifier | `max_depth=5`, `random_state=42` |
| 3 | K-Nearest Neighbors | StandardScaler + KNN (`n_neighbors=5`) |
| 4 | Gaussian Naive Bayes | default parameters |
| 5 | Support Vector Machine | StandardScaler + SVC (RBF kernel, `probability=True`, `random_state=42`) |

### Evaluation Metrics

Computed on the 114-record test set (source: `results/comparison.csv`).

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

## 4. Streamlit Application

The Streamlit application (`app.py`) provides:

- CSV test dataset upload.
- Classification model selection (sidebar).
- Evaluation metric display (Accuracy, AUC, Precision, Recall, F1, MCC).
- Confusion matrix.
- Classification report.

The application loads the previously trained models from `model/` and does not retrain them on uploaded data.

### Input requirements

- A CSV file with exactly 30 feature columns in the same order as the training data, plus a `target` column.
- `target` must contain only 0 (Benign) and 1 (Malignant), with no missing values.

## 5. Project Architecture

```text
Notebook (training)                    Streamlit app (inference only)
-------------------                    ------------------------------
Load dataset (569 x 30)                Upload CSV (features + target)
  -> 80:20 stratified split              -> validate columns / target
  -> train 5 models                      -> load model/<name>.joblib
  -> evaluate on test set                -> predict + predict_proba
  -> save models (model/*.joblib)        -> metrics, confusion matrix,
  -> save results/comparison.csv            classification report
```

## 6. Project Structure

```text
ML_Classification_Assignment_2/
|
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
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
```

## 7. Installation

Requires Python 3.11 or 3.12 (the pinned `numpy==1.26.4` does not support newer versions).

```bash
git clone https://github.com/Venugopal-14/ML-Classification-Assignment-2.git
cd ML-Classification-Assignment-2

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

## 8. Usage

Run the Streamlit app locally:

```bash
streamlit run app.py
```

Then:

1. Select a classification model in the sidebar.
2. Upload `test_data.csv` (or your own CSV in the same format).
3. Review the metrics, confusion matrix and classification report.

To reproduce the training and evaluation, open `ML_Classification_Models.ipynb` in Jupyter and run all cells. This is optional and is not needed to run the app.
