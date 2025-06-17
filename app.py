import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

st.set_page_config(layout="wide")
st.title("Customer Churn Prediction App")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Telco-Customer-Churn.csv")
    return df

df = load_data()

# Preprocessing
@st.cache_data
def preprocess_data(df):
    df = df.drop(['customerID'], axis=1)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})
    df_dummies = pd.get_dummies(df.drop('Churn', axis=1), drop_first=True)
    return df_dummies, df['Churn']

X, y = preprocess_data(df)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Exploration", "Modeling", "Top 10 Features", "Feature Group Summary"])

# Tab 1: Exploration
with tab1:
    st.subheader("Raw Data")
    st.dataframe(df.head())

# Tab 2: Modeling
with tab2:
    st.subheader("Train Logistic Regression Model with SMOTE")
    test_size = st.slider("Test size (%)", 10, 50, 30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_res, y_res)

    y_pred = model.predict(X_test)
    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))
    st.text("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

# Tab 3: Top 10 Features
with tab3:
    st.subheader("Top 10 Important Features (Logistic Regression Coefficients)")
    coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_[0]})
    coef_df["AbsCoefficient"] = coef_df["Coefficient"].abs()
    top10 = coef_df.sort_values("AbsCoefficient", ascending=False).head(10)
    st.dataframe(top10, use_container_width=True)

    # Plot
    fig, ax = plt.subplots()
    sns.barplot(data=top10, x="AbsCoefficient", y="Feature", palette="viridis", ax=ax)
    ax.set_title("Top 10 Features by Absolute Coefficient")
    st.pyplot(fig)

# Tab 4: Grouped Feature Importance
with tab4:
    st.subheader("Grouped Feature Importance from Logistic Regression")

    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_[0]
    })
    coef_df["Group"] = coef_df["Feature"].apply(lambda x: x.split('_')[0])

    grouped = (
        coef_df.groupby("Group")
        .agg(
            MeanAbsCoefficient=("Coefficient", lambda x: np.mean(np.abs(x))),
            MaxAbsCoefficient=("Coefficient", lambda x: np.max(np.abs(x))),
            SumAbsCoefficient=("Coefficient", lambda x: np.sum(np.abs(x))),
            Count=("Coefficient", "count")
        )
        .sort_values("MeanAbsCoefficient", ascending=False)
    )

    st.dataframe(grouped.style.background_gradient(cmap='Blues'), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=grouped.reset_index(), x="MeanAbsCoefficient", y="Group", palette="Blues_r", ax=ax)
    ax.set_title("Grouped Feature Importance (Mean Abs Coefficient)")
    st.pyplot(fig)
