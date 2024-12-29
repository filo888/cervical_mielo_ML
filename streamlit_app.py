import altair as alt
import pandas as pd
import streamlit as st
import pickle

# Set the page configuration
st.set_page_config(page_title="Model Comparison", page_icon="🔄")
st.title("🔄 Model Comparison App")
st.write(
    """
    This app allows you to manually input variables for each feature, process them through three different models 
    (loaded from `.pkl` files), and compare the outputs. Each model applies a different pipeline and outputs the 
    probability of class 1.
    """
)

# Load the models from .pkl files
@st.cache_resource
def load_model(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

model_1 = load_model("model_1.pkl")
model_2 = load_model("model_2.pkl")
model_3 = load_model("model_3.pkl")

# Collect inputs for all required features
st.subheader("Input Variables")

categorical_features = [
    'ASA score', 'BMI', 'Preoperative Grade of Myelopathy (mJOA)',
    'Levels of cervical pathology', 'Levels of radiological myelopathy',
    'Extent of Compression', 'Sex', 'Type of approach', 'Smoke (Yes/no)',
    'Previous Cervical Surgery', 'Predominance of Site of Compression',
    'Cervical Alignment', 'Type of Compression'
]

numerical_features = [
    'Age', 'Symptoms Duration', 'Neck VAS', 'Arm VAS', 'NDI', 'Charlson'
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
