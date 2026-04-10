import streamlit as st
import pandas as pd
import plotly.express as px
from google import genai
import duckdb

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Sales Intelligence 2026", 
    layout="wide", 
    page_icon="💹"
)

# --- 2. DATA ENGINE (Fixed: Duplicate Column Prevention) ---
@st.cache_data
def load_and_clean_data():
    try:
        # Load the dataset
        df = pd.read_csv("superstore.csv", encoding='latin1')
        
        # STEP A: Clean column names FIRST to prevent duplicates
        # This turns "Order Date" into "Order_Date" immediately
        df.columns = [c.replace(' ', '_').replace('-', '_') for c in df.columns]
        
        # STEP B: Convert the cleaned column to datetime in-place
        if 'Order_Date' in df.columns:
            df['Order_Date'] = pd.to_datetime(df['Order_Date'])
        
        # STEP C: Remove any accidental duplicate columns just in case
        df = df.loc[:, ~df.columns.duplicated()].copy()
        
        return df
    except Exception as e:
        st.error(f"❌ Data Error: {e}")
        st.stop()

df = load_and_clean_data()

# --- 3. SIDEBAR FILTERS ---
st.sidebar.header("Control Panel")
all_regions = df['Region'].unique().tolist()
all_cats = df['Category'].unique().tolist()

selected_regions = st.sidebar.multiselect("Region", all_regions, default=all_regions)
selected_cats = st.sidebar.multiselect("Category", all_cats, default=all_cats)

# Filter logic
filtered_df = df[
    (df['Region'].isin(selected_regions)) & 
    (df['Category'].isin(selected_cats))
]

# --- 4. EXECUTIVE DASHBOARD ---
st.title("🛍️ E-Commerce Strategic Dashboard")

if filtered_df.empty:
    st.warning("Please adjust filters to view data.")
else:
    # KPI Row
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Sales", f"${filtered_df['Sales'].sum():,.0f}")
    k2.metric("Total Profit", f"${filtered_df['Profit'].sum():,.0f}")
    k3.metric("Orders", f"{len(filtered_df):,}")

    st.divider()

    # Visuals Row
    c1, c2 = st.columns(2)

    with c1:
        # Visual 1: Bar Chart
        cat_data = filtered_df.groupby('Category')['Sales'].sum().reset_index()
        fig_cat = px.bar(cat_data, x='Category', y='Sales', title="Revenue by Category", 
                         color='Category', template="plotly_white")
        st.plotly_chart(fig_cat, use_container_width=True)

        # Visual 2: Pie Chart (The one that caused the error)
        fig_pie = px.pie(filtered_df, values='Sales', names='Region', title="Sales Distribution",
                         hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        # Visual 3: SQL Trend Analysis
        trend_query = "SELECT Order_Date, SUM(Sales) as Daily_Sales FROM filtered_df GROUP BY 1 ORDER BY 1"
        trend_data = duckdb.query(trend_query).df()
        
        fig_trend = px.line(trend_data, x='Order_Date', y='Daily_Sales', title="Sales Velocity (SQL-Calculated)")
        st.plotly_chart(fig_trend, use_container_width=True)

# --- 5. AI CHATBOT (Gemini 3) ---
st.divider()
st.subheader("🤖 AI Business Analyst")

if "GEMINI_API_KEY" not in st.secrets:
    st.info("Add your GEMINI_API_KEY to Streamlit Secrets.")
else:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about sales..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    summary = f"Total Sales: {filtered_df['Sales'].sum()}, Profit: {filtered_df['Profit'].sum()}"
                    response = client.models.generate_content(
                        model='gemini-3-flash-preview',
                        contents=prompt,
                        config={'system_instruction': f"You are a Sales Analyst. Data: {summary}"}
                    )
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"AI Error: {e}")
