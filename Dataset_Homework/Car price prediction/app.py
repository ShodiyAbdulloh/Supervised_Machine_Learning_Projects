# # Web title

import streamlit as st
import numpy as np
import joblib

# Load the saved Decision Tree model
model = joblib.load("decision_tree_model.pkl")
# Read
with open("accuracy.txt", "r") as f:
    accuracy = f.read()


# Define the feature names (excluding the target column "prediction")
feature_names = [
    'country', 'region', 'population', 'incidence_rate', 'mortality_rate',
    'age', 'survival_rate', 'cost_of_treatment', 'gender_female', 'gender_male',
    'alcohol_consumption_high', 'alcohol_consumption_low', 'alcohol_consumption_moderate',
    'smoking_status_non_smoker', 'smoking_status_smoker', 'hepatitis_b_status_negative',
    'hepatitis_b_status_positive', 'hepatitis_c_status_negative', 'hepatitis_c_status_positive',
    'obesity_normal', 'obesity_obese', 'obesity_overweight', 'obesity_underweight',
    'diabetes_no', 'diabetes_yes', 'rural_or_urban_rural', 'rural_or_urban_urban',
    'seafood_consumption_high', 'seafood_consumption_low', 'seafood_consumption_medium',
    'herbal_medicine_use_no', 'herbal_medicine_use_yes', 'healthcare_access_good',
    'healthcare_access_moderate', 'healthcare_access_poor', 'screening_availability_available',
    'screening_availability_not_available', 'treatment_availability_available',
    'treatment_availability_not_available', 'liver_transplant_access_no', 'liver_transplant_access_yes',
    'ethnicity_african', 'ethnicity_asian', 'ethnicity_caucasian', 'ethnicity_hispanic',
    'ethnicity_mixed', 'preventive_care_good', 'preventive_care_moderate', 'preventive_care_poor'
]   
st.title("Decision Tree modeli yordamida jigar o'smani aniqlash uchun tuzilgan aqlli dastur")
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
st.markdown('<p class="title">🩺 Decision Tree Model for Liver Disease Prediction</p>', unsafe_allow_html=True)
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
    if feature in ["country", "region"]:  
        value = st.text_input(f"Enter {feature}")  # Text input for categorical features
    else:
        value = st.number_input(f"Enter value for {feature}")  # Numeric input fields
    
    input_data.append(value)



# Predict when the user clicks the button
if st.button("Predict"):
    # Convert input data to a NumPy array
    input_array = np.array([input_data])

    # Make predictions
    prediction = model.predict(input_array)[0]

    # Show Prediction Result
    st.write(f"### Prediction: {'Liver Disease (1)' if prediction == 1 else 'No Liver Disease (0)'}")

#streamlit run app.py
import streamlit as st

st.title("🚀 Streamlit is Running!")
st.write("If you see this message, Streamlit is working correctly.")

if st.button("Click Me"):
    st.write("✅ Button Clicked!")
