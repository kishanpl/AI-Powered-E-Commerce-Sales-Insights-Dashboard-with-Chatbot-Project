import streamlit as st
import pandas as pd
import plotly.express as px
from google import genai
import duckdb

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Sales Analytics 2026", 
    layout="wide", 
    page_icon="💹"
)

# --- 2. DATA ENGINE (Requirement: Data Cleaning & SQL) ---
@st.cache_data
def load_and_clean_data():
    try:
        # Load the dataset
        df = pd.read_csv("superstore.csv", encoding='latin1')
        
        # Data Cleaning: Fix Dates and Column Names for SQL
        df['Order_Date'] = pd.to_datetime(df['Order Date'])
        # Replace spaces/hyphens with underscores for SQL compatibility
        df.columns = [c.replace(' ', '_').replace('-', '_') for c in df.columns]
        
        return df
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}. Ensure 'superstore.csv' is on GitHub.")
        st.stop()

df = load_and_clean_data()

# --- 3. SIDEBAR FILTERS (Requirement: Filters) ---
st.sidebar.header("Control Panel")
all_regions = df['Region'].unique().tolist()
all_cats = df['Category'].unique().tolist()

selected_regions = st.sidebar.multiselect("Region", all_regions, default=all_regions)
selected_cats = st.sidebar.multiselect("Category", all_cats, default=all_cats)

# Dynamic Filtering
filtered_df = df[
    (df['Region'].isin(selected_regions)) & 
    (df['Category'].isin(selected_cats))
]

# --- 4. EXECUTIVE DASHBOARD (Requirement: KPI Cards & Visuals) ---
st.title("🛍️ E-Commerce Strategic Dashboard")
st.markdown("### Real-time Performance Metrics")

if filtered_df.empty:
    st.warning("Please select at least one Region and Category to view data.")
else:
    # KPI Row
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Revenue", f"${filtered_df['Sales'].sum():,.0f}")
    k2.metric("Total Profit", f"${filtered_df['Profit'].sum():,.0f}")
    k3.metric("Order Volume", f"{len(filtered_df):,}")

    st.divider()

    # Visuals Row
    c1, c2 = st.columns(2)

    with c1:
        # Visual 1: Sales by Category (Bar Chart)
        cat_data = filtered_df.groupby('Category')['Sales'].sum().reset_index()
        fig_cat = px.bar(cat_data, x='Category', y='Sales', title="Revenue by Category", 
                         color='Category', template="plotly_white")
        st.plotly_chart(fig_cat, use_container_width=True)

        # Visual 2: Sales by Region (Pie Chart)
        fig_pie = px.pie(filtered_df, values='Sales', names='Region', title="Sales Distribution",
                         hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        # Visual 3: SQL Trend Analysis (Requirement: SQL-based Analysis)
        # We use DuckDB to run actual SQL on the dataframe
        trend_query = """
            SELECT Order_Date, SUM(Sales) as Daily_Sales 
            FROM filtered_df 
            GROUP BY 1 
            ORDER BY 1
        """
        trend_data = duckdb.query(trend_query).df()
        
        fig_trend = px.line(trend_data, x='Order_Date', y='Daily_Sales', title="Sales Velocity (SQL-Calculated)",
                            line_shape="linear", markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)

# --- 5. AI CHATBOT (Requirement: Gemini 3 AI & Chatbot) ---
st.divider()
st.subheader("🤖 AI Business Analyst (Powered by Gemini 3 Flash)")

# Security Check for API Key
if "GEMINI_API_KEY" not in st.secrets:
    st.info("💡 To enable the AI Analyst, add your GEMINI_API_KEY to Streamlit Secrets.")
else:
    # Initialize the 2026 SDK Client
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User Input
    if prompt := st.chat_input("Ask a question about your sales performance..."):
        # Stop if data is empty to prevent API errors
        if filtered_df.empty:
            st.error("No data available for analysis. Please adjust your filters.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Context construction for the AI
            summary = f"""
            The user is viewing a sales dashboard. 
            - Total Sales: ${filtered_df['Sales'].sum():,.2f}
            - Total Profit: ${filtered_df['Profit'].sum():,.2f}
            - Best Selling Category: {filtered_df.groupby('Category')['Sales'].sum().idxmax()}
            - Regions filtered: {selected_regions}
            """

            with st.chat_message("assistant"):
                # Professional Loading Spinner
                with st.spinner("Analyzing business metrics..."):
                    try:
                        # 2026 Model: gemini-3-flash-preview
                        response = client.models.generate_content(
                            model='gemini-3-flash-preview',
                            contents=prompt,
                            config={
                                'system_instruction': f"You are a Senior Business Analyst. Answer based on this data: {summary}"
                            }
                        )

                        if response.text:
                            st.markdown(response.text)
                            st.session_state.messages.append({"role": "assistant", "content": response.text})
                        else:
                            st.warning("The AI could not generate a response. Please try rephrasing your question.")

                    except Exception as e:
                        # Professional Error Handling for 2026 APIs
                        if "429" in str(e):
                            st.error("⏳ Rate limit exceeded. Please wait a minute before asking again.")
                        elif "400" in str(e):
                            st.error("🔍 Request Error: Ensure the data is not too large for the model.")
                        else:
                            st.error(f"📡 System Communication Error: {e}")
