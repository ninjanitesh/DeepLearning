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

# Create input fields for each feature
# Adjust the names of the features according to your dataset
# Example:
feature_1 = st.number_input('Feature 1', value=0.0)
feature_2 = st.number_input('Feature 2', value=0.0)
feature_3 = st.number_input('Feature 3', value=0.0)
# Add more features as required
# ...

# Collect the inputs into a numpy array
user_input = np.array([[feature_1, feature_2, feature_3]])  # Add more features here

# Scale the input data
user_input_scaled = scaler.transform(user_input)

# Make a prediction using the loaded model
prediction = model.predict(user_input_scaled)

# Display the result
if prediction > 0.5:
    st.write("The model predicts that the individual is likely to have diabetes.")
else:
    st.write("The model predicts that the individual is unlikely to have diabetes.")
