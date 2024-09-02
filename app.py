import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

# Load the trained model and scaler
model = load_model('diabetes_deep_learning_model.h5')
scaler = joblib.load('scaler.pkl')

# Set up the Streamlit app
st.title('Diabetes Prediction App')

# Input fields for user data
st.write("Enter the following health indicators:")

# Create input fields for each feature, matching the HTML form

# HighBP
HighBP = st.selectbox('High Blood Pressure:', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# HighChol
HighChol = st.selectbox('High Cholesterol:', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# BMI
BMI = st.number_input('BMI:', min_value=10.0, max_value=50.0, step=0.1)

# Smoker
Smoker = st.selectbox('Smoker:', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# PhysActivity
PhysActivity = st.selectbox('Physical Activity:', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# GenHlth
GenHlth = st.selectbox('General Health (1=Excellent, 5=Poor):', options=[1, 2, 3, 4, 5])

# Age
Age = st.selectbox('Age Category:', options=list(range(1, 14)), format_func=lambda x: [
    "18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", 
    "60-64", "65-69", "70-74", "75-79", "80 or older"][x - 1])

# Sex
Sex = st.selectbox('Sex:', options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')

# Collect the inputs into a numpy array
user_input = np.array([[HighBP, HighChol, BMI, Smoker, PhysActivity, GenHlth, Age, Sex]])

# Debugging: Print user input and its shape
st.write("User input:", user_input)
st.write("User input shape:", user_input.shape)

try:
    # Scale the input data
    user_input_scaled = scaler.transform(user_input)

    # Make a prediction using the loaded model
    prediction = model.predict(user_input_scaled)

    # Display the result
    if prediction > 0.5:
        st.write("The model predicts that the individual is likely to have diabetes.")
    else:
        st.write("The model predicts that the individual is unlikely to have diabetes.")
except ValueError as e:
    st.error(f"Error: {e}")
