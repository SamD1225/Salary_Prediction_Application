import streamlit as st
import pickle
import numpy as np
import sklearn

# Page configuration
st.set_page_config(page_title="Salary Predictor", layout="centered")

# Display version for debugging
st.sidebar.write(f"Running scikit-learn version: {sklearn.__version__}")

@st.cache_resource
def load_model():
    try:
        with open('Model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model:
    st.title("💼 Salary Prediction App")
    st.write("Predict estimated salary based on years of professional experience.")

    # User Input
    years_exp = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=2.0, step=0.5)

    if st.button("Calculate Prediction"):
        # The model expects a 2D array (YearsExperience) 
        input_data = np.array([[years_exp]])
        
        try:
            prediction = model.predict(input_data)
            # Accessing the first element of the prediction array 
            st.balloons()
            st.success(f"### Predicted Salary: ${prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.warning("Model file could not be loaded. Please check your GitHub repository for 'Model.pkl'.")
