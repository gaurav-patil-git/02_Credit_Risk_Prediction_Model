import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Load the model and its components
model = joblib.load('D:/[01] SKILLS/[00] Projects/02 Probability of Default (PD) Model/model/model.joblib')
scaler_obj = joblib.load('D:/[01] SKILLS/[00] Projects/02 Probability of Default (PD) Model/model/scaler_obj.joblib')

scaler = scaler_obj['scaler']
cols_to_scale = scaler_obj['cols_to_scale']

def prepare_input(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                    delinquency_ratio, credit_utilization_ratio, num_open_accounts, residence_type,
                    loan_purpose, loan_type):
    # Create a dictionary with input values and dummy values for missing features
    input_data = {
        'age': age,
        'loan_tenure_months': loan_tenure_months,
        'number_of_open_accounts': num_open_accounts,
        'credit_utilization_ratio': credit_utilization_ratio,
        'loan_to_income': loan_amount / income if income > 0 else 0,
        'delinquency_ratio': delinquency_ratio,
        'avg_dpd_per_delinquency': avg_dpd_per_delinquency,
        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented': 1 if residence_type == 'Rented' else 0,
        'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,
        # additional dummy fields just for scaling purpose
        'number_of_dependants': 1,  # Dummy value
        'years_at_current_address': 1,  # Dummy value
        'zipcode': 1,  # Dummy value
        'sanction_amount': 1,  # Dummy value
        'processing_fee': 1,  # Dummy value
        'gst': 1,  # Dummy value
        'net_disbursement': 1,  # Computed dummy value
        'principal_outstanding': 1,  # Dummy value
        'bank_balance_at_application': 1,  # Dummy value
        'number_of_closed_accounts': 1,  # Dummy value
        'enquiry_count': 1  # Dummy value
    }

    # Ensure all columns for features and cols_to_scale are present
    df = pd.DataFrame([input_data])

    # Ensure only required columns for scaling are scaled
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    return df


def predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
            delinquency_ratio, credit_utilization_ratio, num_open_accounts,
            residence_type, loan_purpose, loan_type):
    # Prepare input data
    input_df = prepare_input(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                             delinquency_ratio, credit_utilization_ratio, num_open_accounts, residence_type,
                             loan_purpose, loan_type)

    probability, credit_score, rating = calculate_credit_score(input_df)

    return probability, credit_score, rating


def calculate_credit_score(input_df, base_score=300, scale_length=600):
    x = np.dot(input_df.values, model.coef_.T) + model.intercept_

    # Apply the logistic function to calculate the probability
    default_probability = 1 / (1 + np.exp(-x))

    non_default_probability = 1 - default_probability

    # Convert the probability to a credit score, scaled to fit within 300 to 900
    credit_score = base_score + non_default_probability.flatten() * scale_length

    # Determine the rating category based on the credit score
    def get_rating(score):
        if 300 <= score < 500:
            return 'Poor'
        elif 500 <= score < 650:
            return 'Average'
        elif 650 <= score < 750:
            return 'Good'
        elif 750 <= score <= 900:
            return 'Excellent'
        else:
            return 'Undefined'  # in case of any unexpected score

    rating = get_rating(credit_score[0])

    return default_probability.flatten()[0], int(credit_score[0]), rating

# Set the page configuration and title
st.set_page_config(page_title="Credit Risk Modelling", page_icon="📊")
st.title("Credit Risk Modelling")

# Create rows of three columns each
row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
row4 = st.columns(3)

# Assign inputs to the first row with default values
with row1[0]:
    age = st.number_input('Age', min_value=18, step=1, max_value=100, value=28)
with row1[1]:
    income = st.number_input('Income', min_value=0, value=1200000)
with row1[2]:
    loan_amount = st.number_input('Loan Amount', min_value=0, value=2560000)

# Calculate Loan to Income Ratio and display it
loan_to_income_ratio = loan_amount / income if income > 0 else 0
with row2[0]:
    st.text("Loan to Income Ratio:")
    st.text(f"{loan_to_income_ratio:.2f}")  # Display as a text field

# Assign inputs to the remaining controls
with row2[1]:
    loan_tenure_months = st.number_input('Loan Tenure (months)', min_value=0, step=1, value=36)
with row2[2]:
    avg_dpd_per_delinquency = st.number_input('Avg DPD', min_value=0, value=20)

with row3[0]:
    delinquency_ratio = st.number_input('Delinquency Ratio', min_value=0, max_value=100, step=1, value=30)
with row3[1]:
    credit_utilization_ratio = st.number_input('Credit Utilization Ratio', min_value=0, max_value=100, step=1, value=30)
with row3[2]:
    num_open_accounts = st.number_input('Open Loan Accounts', min_value=1, max_value=4, step=1, value=2)


with row4[0]:
    residence_type = st.selectbox('Residence Type', ['Owned', 'Rented', 'Mortgage'])
with row4[1]:
    loan_purpose = st.selectbox('Loan Purpose', ['Education', 'Home', 'Auto', 'Personal'])
with row4[2]:
    loan_type = st.selectbox('Loan Type', ['Unsecured', 'Secured'])


# Button to calculate risk
if st.button('Calculate Risk'):
    # Call the predict function from the helper module
    # print((age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
    #                                             delinquency_ratio, credit_utilization_ratio, num_open_accounts,
    #                                             residence_type, loan_purpose, loan_type))
    probability, credit_score, rating = predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                                                delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                                                residence_type, loan_purpose, loan_type)

    # Display the results
    st.write(f"Deafult Probability: {probability:.2%}")
    st.write(f"Credit Score: {credit_score}")
    st.write(f"Rating: {rating}")

# Footer
# st.markdown('_Project From Codebasics ML Course_')
