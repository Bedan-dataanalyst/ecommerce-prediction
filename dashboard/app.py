import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Auto ML Analytics Engine", layout="wide")

st.title("🚀 Auto ML & Analytics Engine")
st.markdown("Upload any dataset and get instant insights + ML predictions")

# ----------------------------
# FILE UPLOAD
# ----------------------------
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.write("Shape:", df.shape)

    # ----------------------------
    # AUTO DATA CLEANING
    # ----------------------------
    df = df.dropna()

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    st.subheader("🔍 Column Detection")
    st.write("Numeric Columns:", numeric_cols)
    st.write("Categorical Columns:", categorical_cols)

    # ----------------------------
    # AUTO EDA CHARTS
    # ----------------------------
    st.subheader("📊 Auto Data Insights")

    if len(numeric_cols) > 0:
        col = numeric_cols[0]
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)

    if len(numeric_cols) >= 2:
        fig2 = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title="Feature Relationship")
        st.plotly_chart(fig2, use_container_width=True)

    # ----------------------------
    # OPTIONAL ML SECTION
    # ----------------------------
    st.subheader("🤖 Auto Machine Learning (Optional)")

    target = st.selectbox("Select Target Column (for prediction)", df.columns)

    if st.button("Run Auto ML Model"):

        data = df.copy()

        # encode categorical
        for col in categorical_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

        X = data.drop(columns=[target])
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)

        st.success(f"Model Accuracy: {accuracy:.2f}")

        st.write("Feature Importance:")

        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        fig3 = px.bar(importance, x="Feature", y="Importance", title="Feature Importance")
        st.plotly_chart(fig3, use_container_width=True)