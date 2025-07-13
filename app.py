# app.py
import streamlit as st
import pandas as pd
import joblib

# Load model and training column names
model = joblib.load("health_model.pkl")
#model_columns = joblib.load("model_columns.pkl")  # <-- Load saved column names
columns = [
    'Age',
    'Condition_COVID-19',
    'Condition_Dengue',
    'Condition_Heart Attack',
    'Stage_Moderate',
    'Stage_Severe'
]

st.title("🏥 Health Care Recovery Prediction App")
st.write("Predict whether a patient will recover based on their condition and stage.")

age = st.slider("Age", 10, 90, 30)
condition = st.selectbox("Condition", ['Accident', 'COVID-19', 'Dengue', 'Heart Attack'])
stage = st.selectbox("Stage", ['Early', 'Moderate', 'Severe'])

def prepare_input(age, condition, stage):
    # These are all the columns your model was trained on
    columns = [
        'Age',
        'Condition_COVID-19',
        'Condition_Dengue',
        'Condition_Heart Attack',
        'Stage_Moderate',
        'Stage_Severe'
    ]
    
    # Initialize all to 0
    input_data = dict.fromkeys(columns, 0)
    input_data['Age'] = age
    
    # Set the correct condition and stage
    if condition != 'Accident':  # Accident was dropped as base category during get_dummies
        input_data[f'Condition_{condition}'] = 1
    if stage != 'Early':         # Early was dropped as base category
        input_data[f'Stage_{stage}'] = 1
    
    return pd.DataFrame([input_data])


if st.button("Predict Recovery"):
    input_df = prepare_input(age, condition, stage)
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("✅ Patient is likely to recover.")
    else:
        st.error("⚠️ Patient may not recover. Critical care advised.")
