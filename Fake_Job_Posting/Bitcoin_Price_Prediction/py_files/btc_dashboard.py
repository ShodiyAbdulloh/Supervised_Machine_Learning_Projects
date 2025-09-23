import streamlit as st
import pandas as pd
import plotly.express as px

# Title
st.title("📈 Interactive Bitcoin Price Dashboard")

# Load CSV and automatically select necessary columns
required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

btc = pd.read_csv("bitcoin.csv")

# Keep only necessary columns that exist in the CSV
btc = btc[[col for col in required_columns if col in btc.columns]]

# Convert 'Date' column to datetime, ignore errors
btc['Date'] = pd.to_datetime(btc['Date'], errors='coerce')

# Drop rows with invalid dates
btc = btc.dropna(subset=['Date'])

# Sidebar - date selection
start_date = st.sidebar.date_input("Start Date", btc['Date'].min())
end_date = st.sidebar.date_input("End Date", btc['Date'].max())

# Filter data based on selected dates
btc_filtered = btc[(btc['Date'] >= pd.to_datetime(start_date)) & 
                   (btc['Date'] <= pd.to_datetime(end_date))]

# Show filtered table
st.subheader("Filtered BTC Data")
st.dataframe(btc_filtered)

# Interactive Plotly chart
st.subheader("BTC Closing Price Chart")
fig = px.line(
    btc_filtered, 
    x='Date', 
    y='Close', 
    title='Bitcoin Closing Prices',
    labels={'Close': 'Price (USD)'},
    hover_data=[col for col in ['Open','High','Low','Volume'] if col in btc_filtered.columns]
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("BTC Prices Overview (Open, High, Low, Close)")

fig_all = px.line(
    btc_filtered,
    x='Date',
    y=['Open','High','Low','Close'],
    labels={'value':'Price (USD)', 'variable':'Price Type'},
    title='BTC Price Overview'
)
st.plotly_chart(fig_all, use_container_width=True)
btc_filtered['MA7'] = btc_filtered['Close'].rolling(7).mean()
btc_filtered['MA30'] = btc_filtered['Close'].rolling(30).mean()

st.subheader("BTC Closing Price with Moving Averages")
fig_ma = px.line(
    btc_filtered,
    x='Date',
    y=['Close','MA7','MA30'],
    labels={'value':'Price (USD)', 'variable':'Type'},
    title='BTC Closing Price with 7 & 30-Day MA'
)
st.plotly_chart(fig_ma, use_container_width=True)
