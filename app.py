import streamlit as st
import pandas as pd
import joblib
from pydantic import BaseModel


# Load the saved models
pipeline_logistic = joblib.load('pipeline_logistic.pkl')
pipeline_random_forest = joblib.load('pipeline_random_forest.pkl')
pipeline_stacking = joblib.load('pipeline_stacking.pkl')

# Streamlit application
st.title("Loan Default Prediction")

st.write("""
This application predicts the likelihood of a loan default using three different models:
Logistic Regression, Random Forest, and Stacking.
""")

# Collect user input
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
Annual_Income = st.number_input("Annual Income", min_value=10000, max_value=1000000, value=50000)
Credit_Score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
Employment_Years = st.number_input("Employment Years", min_value=0, max_value=50, value=5)
Loan_Amount_Requested = st.number_input("Loan Amount Requested", min_value=1000, max_value=500000, value=15000)

if st.button("Predict"):
    try:
        # Convert input data into a DataFrame
        data = {
            'Age': [Age],
            'Annual_Income': [Annual_Income],
            'Credit_Score': [Credit_Score],
            'Employment_Years': [Employment_Years],
            'Loan_Amount_Requested': [Loan_Amount_Requested]
        }
        df = pd.DataFrame(data)

        # Make predictions and probabilities
        predictions = {}

        # Logistic Regression
        logistic_pred = pipeline_logistic.predict(df)
        logistic_proba = pipeline_logistic.predict_proba(df)
        predictions['Logistic Regression'] = {
            'Loan_Default': int(logistic_pred[0]),
            'Probability_of_Default': float(logistic_proba[0][1])
        }

        # Random Forest
        rf_pred = pipeline_random_forest.predict(df)
        rf_proba = pipeline_random_forest.predict_proba(df)
        predictions['Random Forest'] = {
            'Loan_Default': int(rf_pred[0]),
            'Probability_of_Default': float(rf_proba[0][1])
        }

        # Stacking
        stacking_pred = pipeline_stacking.predict(df)
        stacking_proba = pipeline_stacking.predict_proba(df)
        predictions['Stacking'] = {
            'Loan_Default': int(stacking_pred[0]),
            'Probability_of_Default': float(stacking_proba[0][1])
        }

        st.write("### Predictions:")
        st.json(predictions)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

