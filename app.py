import streamlit as st
import pandas as pd
import plotly.express as px
from google import genai  # Latest 2026 SDK
import duckdb
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Sales Intelligence 2026", layout="wide", page_icon="📈")

# --- 2. DATA ENGINE (Requirement: Data Cleaning & SQL) ---
@st.cache_data
def get_processed_data():
    try:
        # Load Raw Data
        df = pd.read_csv("superstore.csv", encoding='latin1')
        
        # Preprocessing: Fix Dates and Names for SQL
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df.columns = [c.replace(' ', '_').replace('-', '_') for c in df.columns]
        
        # Feature Engineering
        df['Profit_Margin'] = (df['Profit'] / df['Sales']) * 100
        return df
    except FileNotFoundError:
        st.error("❌ 'superstore.csv' not found. Please upload it to your GitHub repository.")
        st.stop()

df = get_processed_data()

# --- 3. INTERACTIVE FILTERS (Requirement: Filters) ---
st.sidebar.header("Global Filters")
region_list = df['Region'].unique().tolist()
category_list = df['Category'].unique().tolist()

selected_region = st.sidebar.multiselect("Select Region", region_list, default=region_list)
selected_cat = st.sidebar.multiselect("Select Category", category_list, default=category_list)

# Data Filtering Logic
filtered_df = df[(df['Region'].isin(selected_region)) & (df['Category'].isin(selected_cat))]

# --- 4. EXECUTIVE DASHBOARD (Requirement: KPI Cards & Visuals) ---
st.title("🚀 E-Commerce AI Strategy Dashboard")
st.markdown("### Executive Overview")

# KPI Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Total Sales", f"${filtered_df['Sales'].sum():,.0f}")
with kpi2:
    st.metric("Total Profit", f"${filtered_df['Profit'].sum():,.0f}")
with kpi3:
    st.metric("Profit Margin", f"{filtered_df['Profit_Margin'].mean():.1f}%")
with kpi4:
    st.metric("Orders", f"{len(filtered_df):,}")

st.divider()

# Charts (Requirement: Category, Region, Trends)
col_left, col_right = st.columns(2)

with col_left:
    # 1. Sales by Category (Interactive Bar)
    cat_sales = filtered_df.groupby('Category')['Sales'].sum().reset_index()
    fig_cat = px.bar(cat_sales, x='Category', y='Sales', title="Revenue by Category", 
                     color='Category', template="plotly_dark")
    st.plotly_chart(fig_cat, use_container_width=True)

    # 2. Regional Distribution (Pie Chart)
    fig_pie = px.pie(filtered_df, values='Sales', names='Region', title="Sales Weight by Region",
                     hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    # 3. SQL-Based Trend Analysis (Requirement: SQL Analysis)
    # Using DuckDB to simulate complex SQL group-bys
    sql_trend = duckdb.query("""
        SELECT Order_Date, SUM(Sales) as Daily_Sales 
        FROM filtered_df 
        GROUP BY 1 
        ORDER BY 1
    """).df()
    
    fig_trend = px.line(sql_trend, x='Order_Date', y='Daily_Sales', title="Sales Velocity (Timeline)",
                        line_shape="spline", render_mode="svg")
    st.plotly_chart(fig_trend, use_container_width=True)

# --- 5. THE AI CHATBOT (Requirement: Gemini 3 AI & Chatbot) ---
st.divider()
st.subheader("🤖 Chat with your Data")

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User asks a question
if prompt := st.chat_input("Ex: Which category is performing worst?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Context for AI
    data_summary = f"Total Sales: {filtered_df['Sales'].sum()}, Profit: {filtered_df['Profit'].sum()}"

    with st.chat_message("assistant"):
        # --- ADD THE SPINNER HERE ---
        with st.spinner("Analysing sales data..."):
            try:
                # The AI call happens inside the spinner
                response = client.models.generate_content(
                    model='gemini-3-flash-preview',
                    contents=f"Context: {data_summary}\n\nQuestion: {prompt}"
                )
                
                if response.text:
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                else:
                    st.warning("AI returned an empty response. Try rephrasing.")
                    
            except Exception as e:
                st.error(f"AI Error: {e}")

# Secure API Key Check
if "GEMINI_API_KEY" not in st.secrets:
    st.warning("⚠️ Please add your 'GEMINI_API_KEY' to Streamlit Secrets to enable the AI Analyst.")
else:
    # Initialize 2026 Gemini Client
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Interaction
    if prompt := st.chat_input("Ask about sales trends, margins, or regional performance..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Build Context (Requirement: AI-Generated Insights)
        context_summary = f"""
        Dataset Facts:
        - Filters applied: Region: {selected_region}, Category: {selected_cat}
        - Current Sales: ${filtered_df['Sales'].sum():,.0f}
        - Current Profit: ${filtered_df['Profit'].sum():,.0f}
        - Top Selling Product: {filtered_df.groupby('Product_Name')['Sales'].sum().idxmax()}
        """

        with st.chat_message("assistant"):
            # Using Gemini 3 Flash Preview (2026 Model)
            response = client.models.generate_content(
                model='gemini-3-flash-preview',
                contents=f"SYSTEM: You are a Senior Retail Analyst. Use this context: {context_summary}\nUSER: {prompt}"
            )
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
