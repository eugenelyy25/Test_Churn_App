import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel

st.set_page_config(layout="wide", page_title="Telco Churn Analysis")

st.title("Telco Customer Churn Prediction")

# Sidebar file uploader
uploaded_file = st.file_uploader("Upload Telco-Customer-Churn.csv", type=["csv"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

if uploaded_file:
    df = load_data(uploaded_file)

    tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "⚙Preprocessing", "Logistic Regression", "Feature Importance"])

    with tab1:
        st.subheader("Raw Data")
        st.dataframe(df.head())
        st.write("Shape:", df.shape)

        st.subheader("Churn Distribution")
        churn_counts = df["Churn"].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=churn_counts.index, y=churn_counts.values, palette="pastel", ax=ax)
        st.pyplot(fig)

with tab2:
    st.subheader("Encoding and Cleaning")

    df_clean = df.copy()

    # Convert all blanks to NaN
    df_clean.replace(" ", np.nan, inplace=True)

    # Drop rows with any NaN
    df_clean.dropna(inplace=True)

    # Drop customerID
    df_clean.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges to numeric (in case it's object due to blanks)
    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
    df_clean.dropna(inplace=True)  # Drop again if conversion introduced NaNs

    # Encode target
    df_clean["Churn"] = df_clean["Churn"].map({"Yes": 1, "No": 0})

    # Binary encode Yes/No features
    for col in df_clean.columns:
        if df_clean[col].nunique() == 2 and df_clean[col].dtype == "object":
            df_clean[col] = df_clean[col].map({"Yes": 1, "No": 0})

    # One-hot encode remaining categorical features
    df_encoded = pd.get_dummies(df_clean)

    # Final safety check for NaNs
    df_encoded.dropna(inplace=True)

    st.write("No missing values left:", df_encoded.isnull().sum().sum() == 0)
    st.write("Encoded Data Shape:", df_encoded.shape)
    st.dataframe(df_encoded.head())


    with tab3:
        st.subheader("Logistic Regression with SMOTE")

        X = df_encoded.drop("Churn", axis=1)
        y = df_encoded["Churn"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_res, y_res)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {acc:.4f}")

        st.write("Confusion Matrix:")
        st.text(confusion_matrix(y_test, y_pred))

        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

    with tab4:
        st.subheader("Top 10 Most Important Features")

        feature_importance = pd.Series(np.abs(model.coef_[0]), index=X.columns)
        top10 = feature_importance.sort_values(ascending=False).head(10)

        fig, ax = plt.subplots()
        sns.barplot(x=top10.values, y=top10.index, palette="Blues_r", ax=ax)
        ax.set_xlabel("Absolute Coefficient")
        ax.set_title("Top 10 Logistic Regression Features")
        st.pyplot(fig)

else:
    st.warning("Please upload Telco-Customer-Churn.csv to start.")
