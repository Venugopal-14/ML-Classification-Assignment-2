
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Breast Cancer Classification Dashboard")
st.write(
    "An interactive comparison of five machine learning "
    "classification algorithms."
)

st.sidebar.header("Model Configuration")

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

selected_model = st.sidebar.selectbox(
    "Select Classification Model",
    list(model_files.keys())
)

st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.write("Uploaded Dataset Preview")
    st.dataframe(data.head())

    if "target" not in data.columns:
        st.error("CSV must contain the target column.")
        st.stop()

    X = data.drop(columns=["target"])
    y = data["target"]

    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        st.error("Target must contain only 0 and 1 without missing values.")
        st.stop()

    expected_features = 30

    if X.shape[1] != expected_features:
        st.error("Dataset must contain exactly 30 input features.")
        st.stop()

    model_path = os.path.join(
        "model",
        model_files[selected_model]
    )

    if not os.path.exists(model_path):
        st.error("Selected trained model file was not found.")
        st.stop()

    model = joblib.load(model_path)

    try:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        accuracy = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_prob)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y, y_pred)

        st.success(f"{selected_model} evaluated successfully!")

        st.subheader("Evaluation Metrics")

        c1, c2, c3 = st.columns(3)
        c4, c5, c6 = st.columns(3)

        c1.metric("Accuracy", f"{accuracy:.4f}")
        c2.metric("AUC Score", f"{auc:.4f}")
        c3.metric("Precision", f"{precision:.4f}")
        c4.metric("Recall", f"{recall:.4f}")
        c5.metric("F1 Score", f"{f1:.4f}")
        c6.metric("MCC Score", f"{mcc:.4f}")

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y, y_pred, labels=[0, 1])

        st.dataframe(
            pd.DataFrame(
                cm,
                index=["Actual Benign", "Actual Malignant"],
                columns=["Predicted Benign", "Predicted Malignant"]
            )
        )

        st.subheader("Classification Report")

        report = classification_report(
            y,
            y_pred,
            target_names=["Benign", "Malignant"],
            output_dict=True,
            zero_division=0
        )

        st.dataframe(pd.DataFrame(report).transpose())

    except Exception as error:
        st.error(f"Evaluation failed: {error}")

else:
    st.info("Upload test_data.csv to evaluate your selected model.")

st.divider()
st.caption("M.Tech AIML | Machine Learning Assignment 2 | BITS Pilani")
