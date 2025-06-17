import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Telco Churn Streamlit App", layout="wide")
st.title("📊 Telco Customer Churn Analysis – Group 12")

# Upload Data
uploaded_file = st.sidebar.file_uploader("Upload telco_data.csv", type=["csv"])

if uploaded_file:
    telco_df = pd.read_csv(uploaded_file)

    tab1, tab2 = st.tabs(["📊 EDA", "🤖 Modeling"])

    with tab1:
        st.header("Exploratory Data Analysis")

        st.subheader("Dataset Preview")
        st.dataframe(telco_df.head())

        st.write(f"**Shape:** {telco_df.shape[0]} rows × {telco_df.shape[1]} columns")
        st.write("**Missing Values:**")
        st.write(telco_df.isnull().sum())

        telco_df['TotalCharges'] = pd.to_numeric(telco_df['TotalCharges'], errors='coerce')
        telco_df.dropna(inplace=True)
        telco_df['SeniorCitizen'] = telco_df['SeniorCitizen'].replace({1: 'Yes', 0: 'No'})
        telco_df.drop('customerID', axis=1, inplace=True)

        # Encoding for correlation
        encoder = OrdinalEncoder()
        encoded = telco_df.copy()
        encoded[encoded.select_dtypes(include='object').columns] = encoder.fit_transform(
            encoded.select_dtypes(include='object'))

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.heatmap(encoded.corr(), cmap='coolwarm')
        st.pyplot(fig)

        st.subheader("Monthly Charges by Churn")
        fig2, ax2 = plt.subplots()
        sns.kdeplot(data=telco_df, x="MonthlyCharges", hue="Churn", fill=True, common_norm=False, alpha=0.5)
        st.pyplot(fig2)

        st.subheader("Churn Pie Chart")
        fig3 = go.Figure(data=[go.Pie(labels=telco_df['Churn'].value_counts().index,
                                      values=telco_df['Churn'].value_counts().values, hole=0.4)])
        fig3.update_layout(title_text="Churn Distribution")
        st.plotly_chart(fig3)

    with tab2:
        st.header("Modeling with Logistic Regression and SMOTE")

        # Encode target
        df_model = telco_df.copy()
        label_enc = LabelEncoder()
        df_model['Churn'] = label_enc.fit_transform(df_model['Churn'])

        # One-hot encode for features
        X = pd.get_dummies(df_model.drop('Churn', axis=1), drop_first=True)
        y = df_model['Churn']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)

        # SMOTE oversampling
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)

        # Train Logistic Regression
        model = LogisticRegression(max_iter=1000)
        model.fit(X_res, y_res)

        # Predictions and evaluation
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.subheader("Model Evaluation")
        st.write(f"**Accuracy:** {acc:.4f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

        st.subheader("Top 10 Most Important Features")
        coef_df = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': model.coef_[0]
        }).sort_values(by='Coefficient', key=abs, ascending=False)

        st.dataframe(coef_df.head(10).style.background_gradient(cmap='RdBu', axis=0))
else:
    st.warning("📂 Please upload the `telco_data.csv` file to proceed.")
