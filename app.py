import streamlit as st
import pandas as pd
import joblib

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
age_input = st.text_input("Age (18-100)")
annual_income_input = st.text_input("Annual Income (10000-1000000)")
credit_score_input = st.text_input("Credit Score (300-850)")
employment_years_input = st.text_input("Employment Years (0-50)")
loan_amount_requested_input = st.text_input("Loan Amount Requested (1000-500000)")

def convert_input(value, min_value, max_value):
    try:
        value = int(value)
        if value < min_value or value > max_value:
            raise ValueError
        return value
    except ValueError:
        st.error(f"Please enter a valid number between {min_value} and {max_value}.")
        return None

if st.button("Predict"):
    # Convert inputs to integers and validate ranges
    Age = convert_input(age_input, 18, 100)
    Annual_Income = convert_input(annual_income_input, 10000, 1000000)
    Credit_Score = convert_input(credit_score_input, 300, 850)
    Employment_Years = convert_input(employment_years_input, 0, 50)
    Loan_Amount_Requested = convert_input(loan_amount_requested_input, 1000, 500000)

    if None not in (Age, Annual_Income, Credit_Score, Employment_Years, Loan_Amount_Requested):
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
