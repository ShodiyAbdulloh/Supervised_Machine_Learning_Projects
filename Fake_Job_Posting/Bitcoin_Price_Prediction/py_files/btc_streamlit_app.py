# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# ML
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("bitcoin.csv")

# Show summary info about the Data
df.info()
# Display summary statistics for numerical columns (count, mean, std, min, 25%, 50%, 75%, max)
df.describe()
df.shape
# Count the number of missing (null) values in each column
df.isnull().sum()
# List of numeric columns to fill
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
# Forward fill first, then backward fill for any remaining missing values
df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
# Check missing values after filling
print("Missing values after filling:\n", df.isnull().sum())
df.info()
# Plot closing price
plt.figure(figsize=(12,6))
plt.plot(btc.index, btc['Close'], label='BTC Close Price')
plt.title("Bitcoin Closing Price Over Time")
plt.legend()
plt.show()
# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(btc.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()
# Convert date column to datetime if exists
if 'Date' in btc.columns:
    btc['Date'] = pd.to_datetime(btc['Date'])
    btc.set_index('Date', inplace=True)
# Convert Date/Time column to datetime
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.set_index('Date', inplace=True)
elif 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df.set_index('Timestamp', inplace=True)
else:
    print("No Date or Timestamp column found. Time-series analysis may not work.")
# Convert price/volume columns to numeric
numeric_cols = ["Price", "Close", "High", "Low", "Open", "Volume"]
for col in numeric_cols:
    if col in df.columns:
        # Remove commas and spaces, convert to numeric
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "").str.strip(), errors="coerce")
df.info()
# Save cleaned dataset
df.to_csv("bitcoin_cleaned.csv", index=False)
# Features and target
X = btc.drop(columns=['Close', 'Date'], errors='ignore')
y = btc['Close']
# Train-test split (time series: shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=200, max_depth=10),
    "XGBoost": XGBRegressor(random_state=42, n_estimators=200, learning_rate=0.1, max_depth=5)
}

# Function to evaluate model
def evaluate_model(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "MAE": mean_absolute_error(y_true, y_pred)
    }

# Train, predict, evaluate
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores = evaluate_model(y_test, y_pred)
    scores["Model"] = name
    results.append(scores)
# Convert to DataFrame for better display
results_df = pd.DataFrame(results)[["Model", "R2", "RMSE", "MAE"]].sort_values(by="R2", ascending=False)
print(results_df)
# Feature Engineering
# Lag of 1,2,3,7,14 days
for lag in [1,2,3,7,14]:
    btc[f'Close_lag_{lag}'] = btc['Close'].shift(lag)

# Rolling mean and standard deviation of past prices.
for window in [3,7,14,30]:
    btc[f'Close_roll_mean_{window}'] = btc['Close'].rolling(window).mean()
    btc[f'Close_roll_std_{window}'] = btc['Close'].rolling(window).std()
# Difference between current price and previous prices.
btc['Momentum_1'] = btc['Close'] - btc['Close'].shift(1)  # daily momentum
btc['Momentum_7'] = btc['Close'] - btc['Close'].shift(7)  # weekly momentum

# Difference between short-term and long-term moving averages.
btc['MA_diff'] = btc['Close_roll_mean_7'] - btc['Close_roll_mean_30']

# Rolling averages and momentum of trading volume.
btc['Volume_roll_mean_7'] = btc['Volume'].rolling(7).mean()
btc['Volume_momentum_1'] = btc['Volume'] - btc['Volume'].shift(1)

# Drop rows with NaN created by feature engineering
btc.dropna(inplace=True)

btc.to_csv("btc_2_feature_engineered.csv", index=False)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, shuffle=False, test_size=0.2)

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42)
}

model_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_scores[name] = {
        "R2": r2_score(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred, squared=False),
        "MAE": mean_absolute_error(y_test, y_pred)
    }

print(model_scores)

# Convert to DataFrame
scores_df = pd.DataFrame(model_scores)

# Save to CSV
scores_df.to_csv("btc_model_scores.csv", index=False)
print("Model scores saved to btc_model_scores.csv")
# Hyperparameter Tuning
# Ridge hyperparameter tuning
ridge = Ridge()
params_ridge = {"alpha": [0.1, 1.0, 10, 100]}
grid_ridge = GridSearchCV(ridge, params_ridge, cv=5)
grid_ridge.fit(X_train, y_train)

print("Best Ridge params:", grid_ridge.best_params_)

# Best Ridge model
best_ridge = grid_ridge.best_estimator_
y_pred_ridge = best_ridge.predict(X_test)

# Evaluate
r2 = r2_score(y_test, y_pred_ridge)
rmse = mean_squared_error(y_test, y_pred_ridge, squared=False)
mae = mean_absolute_error(y_test, y_pred_ridge)
print(f"Ridge -> R2: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
rf = RandomForestRegressor(random_state=42)
params_rf = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5]
}
grid_rf = GridSearchCV(rf, params_rf, cv=3)
grid_rf.fit(X_train, y_train)

print("Best RF params:", grid_rf.best_params_)

best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)

# Evaluate
r2 = r2_score(y_test, y_pred_rf)
rmse = mean_squared_error(y_test, y_pred_rf, squared=False)
mae = mean_absolute_error(y_test, y_pred_rf)
print(f"Random Forest -> R2: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")

from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

xgb = XGBRegressor(random_state=42)
params_xgb = {
    "n_estimators":[100, 200],
    "learning_rate":[0.01, 0.1],
    "max_depth":[3, 5, 7]
}

rand_xgb = RandomizedSearchCV(xgb, params_xgb, cv=3, n_iter=5, random_state=42)
rand_xgb.fit(X_train, y_train)

print("Best XGB params:", rand_xgb.best_params_)

best_xgb = rand_xgb.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)

# Evaluate
r2 = r2_score(y_test, y_pred_xgb)
rmse = mean_squared_error(y_test, y_pred_xgb, squared=False)
mae = mean_absolute_error(y_test, y_pred_xgb)
print(f"XGBoost -> R2: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")


results = [
    {"Model":"Ridge", "R2":r2_score(y_test, y_pred_ridge), "RMSE":mean_squared_error(y_test, y_pred_ridge, squared=False), "MAE":mean_absolute_error(y_test, y_pred_ridge)},
    {"Model":"Random Forest", "R2":r2_score(y_test, y_pred_rf), "RMSE":mean_squared_error(y_test, y_pred_rf, squared=False), "MAE":mean_absolute_error(y_test, y_pred_rf)},
    {"Model":"XGBoost", "R2":r2_score(y_test, y_pred_xgb), "RMSE":mean_squared_error(y_test, y_pred_xgb, squared=False), "MAE":mean_absolute_error(y_test, y_pred_xgb)}
]

pd.DataFrame(results).to_csv("btc_tuned_model_scores.csv", index=False)

import joblib

joblib.dump(best_ridge, "best_ridge_model.pkl")
joblib.dump(best_rf, "best_rf_model.pkl")
joblib.dump(best_xgb, "best_xgb_model.pkl")


# Save as streamlit_app.py
import streamlit as st

# Load data and model
btc = pd.read_csv("btc_processed.csv")
model = joblib.load("best_ridge_model.pkl")  # example

st.title("Bitcoin Price Prediction")

import streamlit as st
import pandas as pd

# Example: load your BTC dataset

btc = pd.read_csv("bitcoin.csv")

# Ensure Date column exists and is datetime
if 'Date' in btc.columns:
    btc['Date'] = pd.to_datetime(btc['Date'], errors='coerce')
    btc.set_index('Date', inplace=True)
else:
    st.error("No Date column found!")
# Now btc.index is datetime, you can safely use .date()
start_date = st.date_input("Start date", btc.index.min().date())
end_date = st.date_input("End date", btc.index.max().date())
# Filter BTC data by selected date range
mask = (btc.index >= pd.to_datetime(start_date)) & (btc.index <= pd.to_datetime(end_date))
filtered_btc = btc.loc[mask]
# Display filtered BTC data
st.subheader(f"BTC Data from {start_date} to {end_date}")
st.dataframe(filtered_btc)
# Predicted BTC price for a specific date
selected_date = st.date_input(
    "Select a date to view predicted BTC price",
    value=btc.index.max().date(),
    min_value=btc.index.min().date(),
    max_value=btc.index.max().date()
)
selected_date_ts = pd.to_datetime(selected_date)
if selected_date_ts in btc.index and 'Predicted_Close' in btc.columns:
    pred_price = btc.loc[selected_date_ts, 'Predicted_Close']
    st.write(f"Predicted BTC price on {selected_date}: ${pred_price:,.2f}")
else:
    st.write("Prediction for the selected date is not available.")