# btc_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load BTC dataset
# -----------------------------
btc = pd.read_csv("bitcoin.csv")  # Your CSV file
btc['Date'] = pd.to_datetime(btc['Date'], errors='coerce')  # Convert Date to datetime
btc = btc.dropna(subset=['Date', 'Close'])  # Drop rows with missing Date or Close
btc = btc.sort_values('Date')  # Sort by date

# -----------------------------
# Streamlit App Title
# -----------------------------
st.title("📈 Bitcoin Price Analysis App")
st.subheader("Visualize BTC Closing Prices over Time")

# -----------------------------
# Sidebar: Date selection
# -----------------------------
min_date = btc['Date'].min().date()
max_date = btc['Date'].max().date()

start_date = st.sidebar.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End date", max_date, min_value=min_date, max_value=max_date)

# Filter dataset by selected date
mask = (btc['Date'].dt.date >= start_date) & (btc['Date'].dt.date <= end_date)
btc_filtered = btc.loc[mask]

# -----------------------------
# Plot BTC Closing Price
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(btc_filtered['Date'], btc_filtered['Close'], color='orange', label='BTC Close Price')
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("BTC Closing Price Over Time")
plt.legend()
plt.grid(True)

st.pyplot(plt)

# -----------------------------
# Show Data
# -----------------------------
st.subheader("Filtered BTC Data")
st.dataframe(btc_filtered)