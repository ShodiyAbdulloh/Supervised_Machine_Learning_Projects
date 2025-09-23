# # Web title
import pandas as pd
import streamlit as st
import numpy as np
import joblib

# Load the saved linear_regression model
model = joblib.load("linear_regression_model.pkl")
# Read
with open("accuracy_scores.txt", "r") as f:
    accuracy = f.read()


feature_names = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront',
    'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built',
    'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'year',
    'month', 'day', 'weekday', 'house_age', 'price_per_sqft'
]




st.title("Linear Regression modeli yordamida uy narxini aniqlash uchun tuzilgan aqlli dastur")
st.write("Malumotlarni kiriting va natijani oling! .")
st.sidebar.write(f"### Model Accuracy: {accuracy}%")







# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Set background color */
    body {
        background-color: #F5F5F5;
    }

    /* Customize title */
    .title {
        font-size: 36px !important;
        color: blue;
        text-align: center;
        font-weight: bold;
    }

    /* Customize subtitle */
    .subtitle {
        font-size: 20px;
        color: #117A65;
        text-align: center;
        font-style: italic;
    }

    /* Customize sidebar */
    .sidebar-title {
        font-size: 24px;
        color: red;
        text-align: center;
        font-weight: bold;
    }
    
    /* Center align content */
    .main-content {
        max-width: 800px;
        margin: auto;
        padding: 20px;
        background-color: #FFFFFF;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Subtitle with Custom Styling
st.markdown('<p class="title">KC House Price Prediction Using Linear Regression</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">📊 Enter details below to get a prediction</p>', unsafe_allow_html=True)

# Sidebar with accuracy
st.sidebar.markdown('<p class="sidebar-title">📈 Model Performance</p>', unsafe_allow_html=True)
st.sidebar.write(f"### 🎯 Model Accuracy: **{accuracy}%**")

# Main Content Box
st.markdown('<div class="main-content">', unsafe_allow_html=True)
st.write("### Please enter the required details below and click 'Predict'")

# Input fields for each feature
input_data = []
for feature in feature_names:
    value = st.number_input(f"Enter value for {feature}", value=0.0)  # Treat all as numeric
    input_data.append(value)



# Predict when the user clicks the button
if st.button("Predict"):
    # Convert input data to a NumPy array
    input_array = np.array([input_data])

    # Make predictions
    prediction = model.predict(input_array)[0]

    # Show Prediction Result
    st.write(f"### Predicted House Price: ${prediction:,.2f}")

#streamlit run app.py
import streamlit as st

st.title("🚀 Streamlit is Running!")
st.write("If you see this message, Streamlit is working correctly.")

if st.button("Click Me"):
    st.write("✅ Button Clicked!")