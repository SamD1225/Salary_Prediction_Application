import streamlit as st
import pickle
import numpy as np

# Load model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("ML Model Deployment")

st.write("Enter input data as comma-separated values (example: 1,2,3,4)")

# User input
user_input = st.text_input("Input Features")

if st.button("Predict"):
    try:
        # Convert input string to numpy array
        input_list = [float(x.strip()) for x in user_input.split(",")]
        input_array = np.array(input_list).reshape(1, -1)

        # Prediction
        prediction = model.predict(input_array)

        st.success(f"Prediction: {prediction[0]}")

    except Exception as e:
        st.error(f"Error: {e}")
