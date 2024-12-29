import altair as alt
import pandas as pd
import streamlit as st
import pickle

# Set the page configuration
st.set_page_config(page_title="Cervical Mielopathy Outcome Prediction", page_icon="🔄")
st.title("🔄 Cervical Mielopathy Outcome Prediction")
st.write(
    """

    """
)

# Load the models from .pkl files
@st.cache_resource
def load_model(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

model_1 = load_model("Neck_VAS_model.pkl")
model_2 = load_model("mJOA_model.pkl")
model_3 = load_model("Arm_VAS_model.pkl")

# Collect inputs for all required features
st.subheader("Input Variables")

categorical_features = [
    
]

numerical_features = [
    
]

# Create widgets for each feature
inputs = {}
for feature in categorical_features:
    if feature == 'BMI':
        inputs[feature] = st.selectbox(f"{feature}", options=[0, 1, 2, 3, 4])
    else:
        inputs[feature] = st.selectbox(f"{feature}", options=[0, 1])

for feature in numerical_features:
    inputs[feature] = st.number_input(f"{feature}", value=0.0)

# Convert inputs into a DataFrame
input_df = pd.DataFrame([inputs])

# Apply models and display probabilities
st.subheader("Model Predictions")

col1, col2, col3 = st.columns(3)
with col1:
    st.write("**Model 1**")
    prediction_1 = model_1.predict_proba(input_df)[:, 1][0]
    st.write(f"**Probability of Class 1:** {prediction_1:.2f}")

with col2:
    st.write("**Model 2**")
    prediction_2 = model_2.predict_proba(input_df)[:, 1][0]
    st.write(f"**Probability of Class 1:** {prediction_2:.2f}")

with col3:
    st.write("**Model 3**")
    prediction_3 = model_3.predict_proba(input_df)[:, 1][0]
    st.write(f"**Probability of Class 1:** {prediction_3:.2f}")

# Display the inputs and predictions for review
st.write("### Input Values")
st.dataframe(input_df)
