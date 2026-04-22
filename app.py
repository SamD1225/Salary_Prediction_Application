import streamlit as st
import pickle
import numpy as np

# Load the model
def load_model():
    with open('Model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Streamlit UI
st.title("Salary Prediction App")
st.write("Enter the years of experience to predict the salary based on the trained linear regression model.")

# User Input
years_exp = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=1.0, step=0.5)

if st.button("Predict"):
    # Reshape input for the model
    input_data = np.array([[years_exp]])
    prediction = model.predict(input_data)
    
    st.success(f"The predicted salary for {years_exp} years of experience is: ${prediction[0]:,.2f}")

st.divider()
st.info("This app uses a Scikit-Learn Linear Regression model loaded from a pickle file.")
