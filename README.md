# 🏥 Default Risk Assessment Model | BFSI and NBFC

### Developed a default risk assessment ML model to predict probability of default (PD) with custom credit score based on loan borrow's historic data.

## 📌 Table of Contents
- <a href="#overview">Overview</a>
- <a href="#model-preview">Model Preview</a>
- <a href="#dataset">Dataset</a>
- <a href="#tools-technologies">Tools & Technologies</a>
- <a href="#project-structure">Project Structure</a>
- <a href="#data-cleaning-preparation">Data Cleaning & Preparation</a>
- <a href="#target-analysis">Target Analysis</a>
- <a href="#model-development">Model Development</a>
- <a href="#streamlit-app">Streamlit App</a>
- <a href="#author-contact">Author & Contact</a>

<h2><a class="anchor" id="overview"></a>📝 Overview</h2>

This project aims to develop a machine learning product to help loan officers in identifying potential loan defaulters and assign custom credit scores for the applicant. This product streamlines the loan approval process, reduces risk of loan defaults and enables data-driven decision making.

- Develop a classification predictive model
- Ensure model recall is above 90% for positive class
- Deploy a most viable product (MVP) using Streamlit application.

<h2><a class="anchor" id="model-preview"></a>🔗 Model Preview</h2>

<h2><a class="anchor" id="credits"></a>🪪 Credits</h2>

This capstone project is a part of the “_Master Machine Learning for Data Science & AI: Beginner to Advanced_” course offered by **Codebasics** - All rights reserved.

- **Course Instructor**: Mr. Dhaval Patel
- **Platform**: codebasics.io – All rights reserved.

All education content and dataset used as learning resources belong to Codebasics and are protected under their respective rights and terms of use.

<h2><a class="anchor" id="dataset"></a>📊 Dataset</h2>

`.xlsx` files located in `/data/` folder


#### Customer Data (collected from Internal CRM Team):
| Feature                  | Description                             | Data Type |
|--------------------------|-----------------------------------------|-----------|
| cust_id                  | Unique customer identifier              | String    |
| age                      | Age of the customer                     | Integer   |
| gender                   | Gender (M/F)                            | String    |
| marital_status           | Marital status (Married/Single)         | String    |
| employment_status        | Employment type (Salaried/Self-Employed) | String    |
| income                   | Annual income (in Indian Rupees)        | Integer   |
| number_of_dependants     | Number of dependents                    | Integer   |
| residence_type           | Type of residence (Owned/Mortgage)      | String    |
| years_at_current_address | Duration at current address (years)     | Integer   |
| city                     | City of residence                       | String    |
| state                    | State of residence                      | String    |
| zipcode                  | Postal/ZIP code                         | Integer   |

#### Loan Data (collected from Loan Operations Team):
| Feature                        | Description                                                   | Data Type |
|--------------------------------|---------------------------------------------------------------|-----------|
| loan_id                        | Unique identifier for the loan                                | String    |
| cust_id                        | Customer ID linked to the loan (foreign key)                  | String    |
| loan_purpose                   | Purpose of the loan (e.g., Car, Home, Personal, Education)    | String    |
| loan_type                      | Type of loan (Secured/Unsecured)                              | String    |
| sanction_amount                | Total loan amount approved (in INR)                           | Integer   |
| loan_amount                    | Actual loan amount disbursed (in INR)                         | Integer   |
| processing_fee                 | Fee charged for loan processing (in INR)                      | Integer   |
| gst                            | GST applied on processing fee (in INR)                        | Integer   |
| net_disbursement               | Final amount disbursed after deductions (in INR)              | Integer   |
| loan_tenure_months             | Loan repayment period (in months)                             | Integer   |
| principal_outstanding          | Remaining principal amount to be repaid (in INR)              | Integer   |
| bank_balance_at_application | Customer’s bank balance at the time of loan application (in INR) | Integer   |
| disbursal_date                 | Date when the loan was disbursed (DD-MM-YYYY)                 | Date      |
| installment_start_dt           | Date when the first EMI payment is due (DD-MM-YYYY)           | Date      |
| default                        | Whether the loan defaulted (TRUE/FALSE)                       | Boolean   |

#### Bureau Data (collected from CIBIL via Credit Underwriting Team):
| Feature                   | Description                                     | Data Type |
|---------------------------|-------------------------------------------------|-----------|
| cust_id                   | Unique customer identifier (foreign key)        | String    |
| number_of_open_accounts   | Number of currently active credit accounts     | Integer   |
| number_of_closed_accounts | Number of closed credit accounts                | Integer   |
| total_loan_months         | Cumulative duration of all loans (in months)    | Integer   |
| delinquent_months         | Number of months with late/missed payments     | Integer   |
| total_dpd                 | Total days past due (DPD) across all loans     | Integer   |
| enquiry_count             | Number of recent credit inquiries (hard checks) | Integer   |
| credit_utilization_ratio  | Percentage of available credit being used (0-100%) | Integer   |


<h2><a class="anchor" id="tools-technologies"></a>🛠️ Tools & Technologies</h2>

| Task                 | Libraries Used                      |
|----------------------|-------------------------------------|
| Data Preprocessing   | Pandas                              |
| Data Visualization   | Matplotlib, Seaborn                 |
| Feature Engineering  | Pandas, Statsmodels, Scikit-learn   |
| Model Training       | Scikit-learn, XGBoost               |
| Model Fine Tuning    | Optuna                              |
| UI Frontend          | Streamlit                           |

<h2><a class="anchor" id="project-structure"></a>📁 Project Structure</h2>

```
02_Credit_Risk_Prediction_Model/
│
├── data/
│   ├── raw/               # Original, immutable data dumps
│   ├── processed/         # Cleaned & feature-engineered datasets
│
├── documents/             # Scope of work
│
├── models/                # Saved model and scaler objects 
│
├── mvp/                   # Minimum Viable Product (Streamlit app)
│
├── notebooks/             # Jupyter notebooks organized by purpose
│
├── visuals/               # Mockups and model preview
│
├── README.md              # High-level project overview
├── .gitignore             # Ignore data, models, logs if using Git

```

<h2><a class="anchor" id="data-cleaning-preparation"></a>🧼 Data Cleaning & Preparation</h2>

### **Data Cleaning**
- Detected negligible (0.124 %) no. of rows with missing values and imputed them with mode.
- Detected and handled anomalies :
  - Principal Outstanding : -1 (min value) - Filtered the data
  - Income, Sanction Amount, Loan Amount, Processing Fee, GST, Net Disbursement, Bank Balance at Application : 0 (min value) - Filtered the data
- Processing Fee (PF): Validated records using industry norms - PF = 3% * Loan Amount
- GST: Validated records using industry norms - PF = 18% * Loan Amount 
- Corrected categories of Loan Purpose : 'Personaal'
- Handled Data Leaking (Train Test contamination, Target Leakage)

### **Feature Engineering**
- Derived three new features using existing ones
- Applied feature scaling
- Used VIF to detect mutlicolinearity and eliminate those with higher value
- Used WoE and IV to detech categorical features with low predictive power and eliminated them
- Applied One Hot Encoding on 3 features
- Applied Label Encoding on 4 columns

<h2><a class="anchor" id="target-analysis"></a>🎯 Target Analysis</h2>

- Through target analysis we realised that there is a class imbalance: defaulter class is ~8.61%
- Using KDE plots we identified 10 out of 33 features have distinct patterns to distinguish defaulters and non-defaulters
- We also identified 5 categories from respective features that highlight patterns to distinguish defaulters and non-defaulters
- Exported processed data and scaler object

<h2><a class="anchor" id="model-development"></a>🤖 Model Development</h2>

### **Model Training**
Performance table of different models with test rank:
| Classifier | 0 | | 1 | | Test Rank |
| :--- | :--- | :--- | :--- | :--- | :--- |
| | recall | f1-score | recall | f1-score | |
| Logistic Regression | 0.93 | 0.96 | 0.94 | 0.71 | 1 |
| SVM | 0.94 | 0.97 | 0.94 | 0.74 | 1 |
| Random Forest | 0.96 | 0.97 | 0.84 | 0.76 | 3 |
| XGBoost | 0.98 | 0.98 | 0.8 | 0.78 | 4 |
| Decision Tree | 0.96 | 0.97 | 0.76 | 0.69 | 5 |

- Since Logistic Regression performed better than others, we moved ahead with it for hyperparameter fine tuning using Optuna.

Best Parameters for Logistic Regression:
- 'C': 12.490728236643456,
- 'solver': 'newton-cg'
- F1 Score: 0.9525497338300815

<h2><a class="anchor" id="streamlit-app"></a>📱 Streamlit App</h2>

- Exported model using `joblib`
- Developed and deployed a most viable product (MVP) using `streamlit`
- This MVP will be utilized by underwriters for 3 to 6 months for feedback and improvement before production

<h2><a class="anchor" id="author-contact"></a>📝 Author & Contact</h2>

**Gaurav Patil** (Data Analyst) 
- 🔗 [LinkedIn](https://www.linkedin.com/in/gaurav-patil-in/)


