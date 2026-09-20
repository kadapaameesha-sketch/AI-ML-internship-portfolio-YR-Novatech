import streamlit as st
import numpy as np

# Page Configuration
st.set_page_config(page_title="AI Real-World Application", page_icon="💡", layout="centered")

st.title("💡 AI-Powered Real-World Prediction System")
st.write("This application demonstrates a practical machine learning model integrated into an interactive user interface.")

# Sidebar for User Inputs
st.sidebar.header("User Input Parameters")

def user_input_features():
    # Example inputs for a predictive model (e.g., performance or pricing metric)
    param_1 = st.sidebar.slider("Feature Metric 1 (Hours/Value)", 1, 100, 50)
    param_2 = st.sidebar.slider("Feature Metric 2 (Score/Index)", 10, 500, 150)
    category = st.sidebar.selectbox("Category Type", ["Standard", "Premium", "Enterprise"])
    
    cat_val = 1 if category == "Standard" else (2 if category == "Premium" else 3)
    
    data = np.array([[param_1, param_2, cat_val]])
    return data

input_df = user_input_features()

st.subheader("🔍 Current User Inputs")
st.write(input_df)

# Prediction Button & Logic
if st.button("Generate AI Prediction"):
    # Simulated machine learning inference calculation
    prediction_score = (input_df[0][0] * 1.5) + (input_df[0][1] * 0.8) + (input_df[0][2] * 10)
    
    st.success(f"Successfully Generated Prediction Result!")
    st.metric(label="Predicted Output Metric", value=f"{round(prediction_score, 2)} units")
else:
    st.info("Adjust the sidebar parameters and click the button above to generate predictions.")
