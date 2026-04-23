import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Cost Optimization Dashboard", layout="wide")

st.title("📊 Cost Optimization Dashboard")

# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("⚙️ Controls")

file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Load dataset
if file:
    df = pd.read_csv(file, encoding='latin1')
else:
    st.warning("Using default dataset (add your CSV for real use)")
    df = pd.read_csv("data.csv", encoding='latin1')

# ==============================
# DATA PREPROCESSING
# ==============================
df['Profit Margin'] = df['Profit'] / df['Sales']

if 'Order Date' in df.columns:
    df['Order Date'] = pd.to_datetime(df['Order Date'])

# ==============================
# SIDEBAR FILTERS
# ==============================
if 'Region' in df.columns:
    region = st.sidebar.multiselect("Region", df['Region'].unique(), default=df['Region'].unique())
    df = df[df['Region'].isin(region)]

if 'Category' in df.columns:
    category = st.sidebar.multiselect("Category", df['Category'].unique(), default=df['Category'].unique())
    df = df[df['Category'].isin(category)]

# ==============================
# KPI METRICS
# ==============================
st.header("📌 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Sales", f"{df['Sales'].sum():,.0f}")
col2.metric("Total Profit", f"{df['Profit'].sum():,.0f}")
col3.metric("Total Orders", df.shape[0])
col4.metric("Profit Margin", f"{(df['Profit'].sum()/df['Sales'].sum())*100:.2f}%")

# ==============================
# TABS
# ==============================
tab1, tab2, tab3, tab4 = st.tabs(["📈 EDA", "📊 Analysis", "🤖 ML Model", "💡 Insights"])

# ==============================
# TAB 1: EDA
# ==============================
with tab1:
    st.subheader("Distributions")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df['Sales'], kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.histplot(df['Profit'], kde=True, ax=ax)
        st.pyplot(fig)

    st.subheader("Profit Margin Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Profit Margin'], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df[['Sales','Profit','Discount']].corr(), annot=True, ax=ax)
    st.pyplot(fig)

# ==============================
# TAB 2: ANALYSIS
# ==============================
with tab2:
    st.subheader("Discount vs Profit")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='Discount', y='Profit', hue='Category', ax=ax)
    st.pyplot(fig)

    st.subheader("Profit by Category")
    cat_profit = df.groupby('Category')['Profit'].sum().sort_values()
    fig, ax = plt.subplots()
    cat_profit.plot(kind='barh', ax=ax)
    st.pyplot(fig)

    st.subheader("Region Performance")
    region_perf = df.groupby('Region')[['Sales','Profit']].sum()
    fig, ax = plt.subplots()
    region_perf.plot(kind='bar', ax=ax)
    st.pyplot(fig)

    st.subheader("Top 10 Profitable Products")
    top_products = df.groupby('Product Name')['Profit'].sum().nlargest(10)
    st.dataframe(top_products)

    st.subheader("Top 10 Loss-Making Products")
    worst_products = df.groupby('Product Name')['Profit'].sum().nsmallest(10)
    st.dataframe(worst_products)

    if 'Order Date' in df.columns:
        st.subheader("Profit Over Time")
        time_data = df.groupby(df['Order Date'].dt.to_period('M'))['Profit'].sum()
        fig, ax = plt.subplots()
        time_data.plot(ax=ax)
        st.pyplot(fig)

# ==============================
# TAB 3: ML MODEL
# ==============================
with tab3:
    st.subheader("Profit Prediction")

    features = ['Sales', 'Discount']
    target = 'Profit'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model_choice = st.selectbox("Choose Model", ["Linear Regression", "Random Forest"])

    if model_choice == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor()

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    st.write(f"Model R² Score: {score:.2f}")

    st.subheader("Predict Profit")

    sales_input = st.number_input("Sales", min_value=0.0)
    discount_input = st.slider("Discount", 0.0, 1.0, 0.1)

    if st.button("Predict"):
        prediction = model.predict([[sales_input, discount_input]])
        st.success(f"Predicted Profit: {prediction[0]:.2f}")

# ==============================
# TAB 4: INSIGHTS
# ==============================
with tab4:
    st.subheader("Business Insights")

    if df['Discount'].mean() > 0.2:
        st.warning("High discounts detected → may reduce profitability")

    loss_count = df[df['Profit'] < 0].shape[0]
    st.write(f"Loss-making transactions: {loss_count}")

    top_region = df.groupby('Region')['Profit'].sum().idxmax()
    st.success(f"Most profitable region: {top_region}")

    worst_category = df.groupby('Category')['Profit'].sum().idxmin()
    st.error(f"Worst performing category: {worst_category}")

    st.subheader("Recommendations")
    st.write("""
    - Reduce excessive discounts
    - Focus on high-margin categories
    - Re-evaluate loss-making products
    - Invest more in profitable regions
    """)