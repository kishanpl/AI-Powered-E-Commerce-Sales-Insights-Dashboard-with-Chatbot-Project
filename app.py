import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import duckdb

# --- PAGE CONFIG ---
st.set_page_config(page_title="Free AI Sales Dashboard", layout="wide")

# --- 1. DATA LOADING & CLEANING (Requirement: Data Cleaning) ---
@st.cache_data
def load_data():
    df = pd.read_csv("superstore.csv", encoding='latin1')
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    # Fix column names for SQL compatibility (spaces to underscores)
    df.columns = [c.replace(' ', '_') for c in df.columns]
    return df

try:
    df = load_data()
except:
    st.error("Missing 'superstore.csv'. Please upload it to GitHub.")
    st.stop()

# --- 2. SIDEBAR FILTERS (Requirement: Filters) ---
st.sidebar.header("Dashboard Filters")
region = st.sidebar.multiselect("Region", df['Region'].unique(), default=df['Region'].unique())
category = st.sidebar.multiselect("Category", df['Category'].unique(), default=df['Category'].unique())

filtered_df = df[(df['Region'].isin(region)) & (df['Category'].isin(category))]

# --- 3. KPI CARDS (Requirement: KPI Cards) ---
st.title("📊 AI E-Commerce Insights (Gemini Powered)")
k1, k2, k3 = st.columns(3)
k1.metric("Total Sales", f"${filtered_df['Sales'].sum():,.0f}")
k2.metric("Total Profit", f"${filtered_df['Profit'].sum():,.0f}")
k3.metric("Total Orders", f"{len(filtered_df):,}")

# --- 4. CHARTS (Requirement: Visualizations/Power BI Style) ---
st.divider()
c1, c2 = st.columns(2)

with c1:
    # Sales by Category
    fig_cat = px.bar(filtered_df.groupby('Category')['Sales'].sum().reset_index(), 
                     x='Category', y='Sales', title="Sales by Category", color='Category')
    st.plotly_chart(fig_cat, use_container_width=True)

    # Sales by Region
    fig_reg = px.pie(filtered_df, values='Sales', names='Region', title="Sales Distribution")
    st.plotly_chart(fig_reg, use_container_width=True)

with c2:
    # SQL Query for Trend (Requirement: SQL Analysis)
    # Using DuckDB to run SQL on the dataframe
    trend_data = duckdb.query("SELECT Order_Date, SUM(Sales) as Sales FROM filtered_df GROUP BY 1 ORDER BY 1").df()
    fig_trend = px.line(trend_data, x='Order_Date', y='Sales', title="Daily Sales Trend")
    st.plotly_chart(fig_trend, use_container_width=True)

# --- 5. FREE AI CHATBOT (Requirement: AI Insights & Chatbot) ---
st.divider()
st.subheader("🤖 Ask the AI Business Analyst")

# Setup Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-1.5-flash')

if prompt := st.chat_input("Ex: Give me a 3-point summary of this data."):
    st.chat_message("user").write(prompt)
    
    # Context for the AI
    data_context = f"""
    You are a business analyst. Here is the current dashboard data summary:
    - Total Sales: ${filtered_df['Sales'].sum():,.2f}
    - Total Profit: ${filtered_df['Profit'].sum():,.2f}
    - Highest Selling Category: {filtered_df.groupby('Category')['Sales'].sum().idxmax()}
    """
    
    with st.chat_message("assistant"):
        response = model.generate_content(f"{data_context}\n\nUser Question: {prompt}")
        st.write(response.text)
