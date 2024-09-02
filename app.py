import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

# Load the trained model and scaler
try:
    model = load_model('diabetes_deep_learning_model.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()  # Stop execution if model loading fails

try:
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"Error loading scaler: {e}")
    st.stop()

# Set up the Streamlit app
st.title('Diabetes Prediction App')

# Input fields for user data
st.write("Enter the following health indicators:")

# Create sliders for input features where appropriate

HighBP = st.selectbox('High Blood Pressure:', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
HighChol = st.selectbox('High Cholesterol:', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
BMI = st.slider('BMI:', min_value=10.0, max_value=50.0, step=0.1)
Smoker = st.selectbox('Smoker:', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
PhysActivity = st.selectbox('Physical Activity:', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
GenHlth = st.slider('General Health (1=Excellent, 5=Poor):', min_value=1, max_value=5, step=1)
Age = st.slider('Age Category:', min_value=1, max_value=13, step=1, format="%d", 
                help="Select age category (1=18-24, 2=25-29, ..., 13=80 or older)")
Sex = st.selectbox('Sex:', options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')

# Display the button
if st.button('Predict'):
    # Collect the inputs into a numpy array
    user_input = np.array([[HighBP, HighChol, BMI, Smoker, PhysActivity, GenHlth, Age, Sex]])

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
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
