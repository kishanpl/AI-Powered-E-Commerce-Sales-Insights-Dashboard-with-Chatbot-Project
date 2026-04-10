import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
import duckdb

st.set_page_config(page_title="AI E-Commerce Dashboard", layout="wide")

# --- DATA PREPROCESSING (Requirement: Data Cleaning) ---
@st.cache_data
def load_and_clean():
    df = pd.read_csv("superstore.csv", encoding='latin1')
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df.columns = [c.replace(' ', '_') for c in df.columns] # SQL-friendly names
    return df

df = load_and_clean()

# --- SIDEBAR FILTERS (Requirement: Filters) ---
st.sidebar.header("Dashboard Filters")
region = st.sidebar.multiselect("Select Region", df['Region'].unique(), default=df['Region'].unique())
category = st.sidebar.multiselect("Select Category", df['Category'].unique(), default=df['Category'].unique())

filtered_df = df[(df['Region'].isin(region)) & (df['Category'].isin(category))]

# --- KPI CARDS (Requirement: KPI Cards) ---
st.title("📊 Sales Insights Dashboard")
k1, k2, k3 = st.columns(3)
k1.metric("Total Sales", f"${filtered_df['Sales'].sum():,.0f}")
k2.metric("Total Profit", f"${filtered_df['Profit'].sum():,.0f}")
k3.metric("Orders", len(filtered_df))

# --- CHARTS (Requirement: Visualizations) ---
c1, c2 = st.columns(2)
with c1:
    fig_cat = px.bar(filtered_df.groupby('Category')['Sales'].sum().reset_index(), 
                     x='Category', y='Sales', title="Sales by Category", color_discrete_sequence=['#0083B8'])
    st.plotly_chart(fig_cat, use_container_width=True)

    fig_reg = px.pie(filtered_df, values='Sales', names='Region', title="Sales by Region")
    st.plotly_chart(fig_reg, use_container_width=True)

with c2:
    # SQL inside Python (Requirement: SQL Analysis)
    trend_data = duckdb.query("SELECT Order_Date, SUM(Sales) as Sales FROM filtered_df GROUP BY 1 ORDER BY 1").df()
    fig_trend = px.line(trend_data, x='Order_Date', y='Sales', title="Monthly Sales Trend")
    st.plotly_chart(fig_trend, use_container_width=True)

# --- AI CHATBOT (Requirement: ChatGPT API & Chatbot) ---
st.divider()
st.subheader("🤖 AI Data Assistant")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if prompt := st.chat_input("Ask: What is the most profitable category?"):
    st.chat_message("user").write(prompt)
    
    # Send data context to AI (Requirement: AI Insights)
    context = f"Total Sales: {filtered_df['Sales'].sum()}, Top Category: {filtered_df.groupby('Category')['Sales'].sum().idxmax()}"
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": f"Analyze this data: {context}"}, {"role": "user", "content": prompt}]
    )
    st.chat_message("assistant").write(response.choices[0].message.content)