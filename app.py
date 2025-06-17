import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

st.title("🔍 Customer Churn Prediction App")

# --- Step 1: Upload CSV ---
file = st.file_uploader("📁 Upload your CSV file", type=["csv"])
if file:
    df = pd.read_csv(file)
    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.head())

    # --- Step 2: Initial checks ---
    st.write("✅ Shape of dataset:", df.shape)
    st.write("✅ Missing values:", df.isnull().sum().sum())

    # --- Step 3: Target encoding ---
    if 'Churn' not in df.columns:
        st.error("❌ Target column 'Churn' not found.")
        st.stop()
    le = LabelEncoder()
    df['Churn'] = le.fit_transform(df['Churn'])

    # --- Step 4: One-hot encoding for categorical features ---
    df_encoded = pd.get_dummies(df, drop_first=True)
    st.write("✅ Encoded dataset shape:", df_encoded.shape)

    # --- Step 5: Split features and label ---
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']

    # --- Step 6: Impute missing values ---
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # --- Step 7: Train-Test Split ---
    if X_imputed.shape[0] == 0:
        st.error("❌ No rows after preprocessing. Check your data.")
        st.stop()
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.25, random_state=42)

    # --- Step 8: Apply SMOTE to balance classes ---
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    # --- Step 9: Train Logistic Regression model ---
    model = LogisticRegression(max_iter=1000)
    model.fit(X_res, y_res)

    # --- Step 10: Evaluate model ---
    y_pred = model.predict(X_test)
    st.subheader("📈 Model Evaluation")
    st.write("✅ Accuracy:", accuracy_score(y_test, y_pred))
    st.text("Confusion Matrix:")
    st.text(confusion_matrix(y_test, y_pred))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # --- Step 11: Churn Distribution Plot ---
    churn_counts = df['Churn'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=churn_counts.index, y=churn_counts.values, hue=churn_counts.index,
                palette="pastel", ax=ax, legend=False)
    ax.set_title("Churn Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # --- Optional: Show feature importance ---
    st.subheader("🧠 Feature Importance")
    importance = pd.Series(model.coef_[0], index=X.columns)
    st.bar_chart(importance.sort_values(ascending=False))
