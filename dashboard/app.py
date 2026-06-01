import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Ecommerce Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# LOAD DATA
# ----------------------------
model = joblib.load("models/customer_model.pkl")
rfm = pd.read_csv("data/rfm_segmented.csv")

# ----------------------------
# SIDEBAR - PREDICTION TOOL
# ----------------------------
st.sidebar.title("🧠 Customer Prediction Tool")

spent = st.sidebar.number_input("Total Spent", min_value=0.0)
items = st.sidebar.number_input("Total Items", min_value=0)
orders = st.sidebar.number_input("Total Orders", min_value=0)

if st.sidebar.button("Predict Customer Type"):
    input_data = np.array([[spent, items, orders]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.sidebar.success("🔥 High Value Customer")
    else:
        st.sidebar.warning("⚠️ Regular Customer")

# ----------------------------
# TITLE
# ----------------------------
st.title("🛒 Ecommerce Analytics & Customer Intelligence Dashboard")

st.markdown("Real-time insights from ecommerce transaction data")

# ----------------------------
# KPI CARDS
# ----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("👥 Total Customers", len(rfm))
col2.metric("💰 Total Revenue", f"${rfm['Monetary'].sum():,.0f}")
col3.metric("📦 Avg Spend", f"${rfm['Monetary'].mean():,.2f}")
col4.metric("🔁 Avg Orders", f"{rfm['Frequency'].mean():.1f}")

st.divider()

# ----------------------------
# CUSTOMER SEGMENTS CHART
# ----------------------------
st.subheader("📊 Customer Segments Overview")

segment_counts = rfm["Segment"].value_counts().reset_index()
segment_counts.columns = ["Segment", "Count"]

fig1 = px.bar(
    segment_counts,
    x="Segment",
    y="Count",
    title="Customer Segment Distribution"
)

st.plotly_chart(fig1, use_container_width=True)

# ----------------------------
# REVENUE DISTRIBUTION
# ----------------------------
st.subheader("💰 Revenue Distribution")

fig2 = px.histogram(
    rfm,
    x="Monetary",
    nbins=30,
    title="Customer Spending Distribution"
)

st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# TOP CUSTOMERS
# ----------------------------
st.subheader("🏆 Top Customers")

top_customers = rfm.sort_values("Monetary", ascending=False).head(10)

fig3 = px.bar(
    top_customers,
    x="CustomerID",
    y="Monetary",
    title="Top 10 Customers by Revenue"
)

st.plotly_chart(fig3, use_container_width=True)

# ----------------------------
# RFM RELATIONSHIP VIEW
# ----------------------------
st.subheader("📈 Customer Behavior Map (RFM Analysis)")

fig4 = px.scatter(
    rfm,
    x="Frequency",
    y="Monetary",
    color="Segment",
    size="Monetary",
    title="RFM Customer Segmentation"
)

st.plotly_chart(fig4, use_container_width=True)

# ----------------------------
# DATA TABLES
# ----------------------------
st.subheader("⚠️ At Risk Customers")

at_risk = rfm[rfm["Segment"] == "At Risk"]
st.dataframe(at_risk.head(10))

st.subheader("📋 Full Dataset Preview")

st.dataframe(rfm.head(20))