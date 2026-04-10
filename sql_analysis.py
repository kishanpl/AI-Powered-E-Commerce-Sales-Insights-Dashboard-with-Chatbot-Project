import duckdb
import pandas as pd

def run_sql_queries(csv_path):
    # Connect to an in-memory database
    con = duckdb.connect(database=':memory:')
    df = pd.read_csv(csv_path, encoding='latin1')
    con.register('sales_data', df)
    
    # 1. SQL for Monthly Trend
    monthly_sql = con.execute("""
        SELECT "Order Date", SUM(Sales) as Total_Sales 
        FROM sales_data 
        GROUP BY 1 
        ORDER BY 1
    """).df()
    
    # 2. SQL for Category Insights
    category_sql = con.execute("""
        SELECT Category, SUM(Sales) as Sales, SUM(Profit) as Profit
        FROM sales_data 
        GROUP BY Category
    """).df()
    
    return monthly_sql, category_sql