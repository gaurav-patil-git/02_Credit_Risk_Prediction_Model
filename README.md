# 💳 Credit Risk Prediction Model

### Developed a credit risk model with a custom scorecard for an Indian NBFC to classify loan applicants into risk categories, enhancing credit assessment and decision-making.

## 🔗 Project Demonstration
### Demo GIF

![Risk](https://github.com/user-attachments/assets/7a61388b-0fb9-4f4c-98d5-33ea1ae6592e)

## 🪪 Credits

This capstone project is a part of the “_Master Machine Learning for Data Science & AI: Beginner to Advanced_” course offered by **Codebasics** - All rights reserved.

- **Course Instructor**: Mr. Dhaval Patel
- **Platform**: codebasics.io – All rights reserved.

All education content and dataset used as learning resources belong to Codebasics and are protected under their respective rights and terms of use.

## 📌 Table of Contents
- [Description](#description)
- [Tech Stack](#tech-stack)
- [Data Understanding](#data-understanding)
- [Data Preparation](#data-preparation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Model Development](#model-development)
- [Model Evaluation](#model-evaluation)
- [Streamlit Application](#streamlit-application)
- [Reflections](#reflections)
- [Conclusion](#conclusion)

## 📝 Description

**AtliQ AI** a leading AI service provider, to develop a sophisticated credit risk model for a Non-Banking Financial Company (NBFC) based in India. This model will include a credit scorecard that categorizes loan applications into Poor, Average, Good, and Excellent categories, based on criteria similar to the CIBIL scoring system. 

The project will be executed in two phases:
- **Phase 1 (MVP)**: Build and deploy a predictive model integrated into a Streamlit application. This app will allow loan officers to input borrower demographics, loan details and bureau information to instantly receive the default probability and an associated credit rating. 
- **Phase 2**: After a two-month production trial, implement monitoring tools to continuously evaluate model’s performance. Based on the results, establish procedures for Straight Through Processing (STP) for high-confidence applications, minimizing the need for manual review.

_Note: This project scope is limited to Phase 1. A detailed Phase 1 walkthrough is provided in scope-of-work document._

## 🛠️ Tech Stack  
| Task                 | Tools Used                          |
|----------------------|-------------------------------------|
| Data Preprocessing   | Pandas                              |
| Data Visualization   | Matplotlib, Seaborn                 |
| Feature Engineering  | Pandas, Statsmodels, Scikit-learn   |
| Model Training       | Scikit-learn, XGBoost               |
| Model Fine Tuning    | Optuna                              |
| UI Frontend          | Streamlit                           |

## Jira (Kandan Board)
To manage the project activities and transform ML Lifecycle into a smooth kanban board workflow, we utilized Jira by Atlassian.

## Data Collection
1. Customer Data was collected from **Internal CRM Team**
2. Load Data was collected from **Loan Operations Team**
3. Bureau Data was collected from CIBIL via **Credit Underwriting Team**

## 📊 Data Understanding
### Raw Data Overview:
#### Customer Data:
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

#### Loan Data:
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

#### Bureau Data:
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

### Data Quality Assessment:

Performed the following data sanitation steps to ensure data quality for EDA and feature engineering:
1. Check Data Types - Validate and correct data types for each column
2. Handle Duplicate Records - Identify and remove duplicate rows.
3. Handle Missing Values - Detect nulls and decide whether to impute, drop, or flag based on context.
4. Fix Structural Errors - Correct typos and inconsistent formatting
5. Handle Outliers - Detect and address anomalies using statistical method



```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File names and corresponding variable names
file_info = {
    "Customers": "dataset/customers.csv",
    "Loans": "dataset/loans.csv",
    "Bureau": "dataset/bureau_data.csv"
}

# Dictionary to store DataFrames
dataframes = {}

# Loop through files, read them, and print their shapes
for name, path in file_info.items():
    df = pd.read_csv(path)
    dataframes[name] = df
    print(f"Shape of {name} DataFrame:")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}\n")

# Merge all DataFrames
# Access individual DataFrames if needed
# df_customers = dataframes["Customers"]
# df_loans = dataframes["Loans"]
# df_bureau = dataframes["Bureau"]

# df = pd.merge(pd.merge(dataframes["Customers"], dataframes["Loans"], on='cust_id'), dataframes["Bureau"], on='cust_id')

df = reduce(lambda left, right: pd.merge(left, right, on='cust_id'), dataframes.values())

df.head()

# Inspecting Imbalance in Target Variable:
df['default'] = df['default'].astype(int)

print("Value Count of 'default' column:")
print(df.default.value_counts())
```
We can see class imbalance in target variable - `default`. We will address it in following code blocks.
To avoid train-test contamination or data leakage, we will split the dataset into training and testing sets. All subsequent EDA and feature engineering will be performed exclusively on the training set. The test set is reserved for final model evaluation.
```
from sklearn.model_selection import train_test_split

X = df.drop("default", axis="columns")
y = df['default']

X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=0.25,random_state=42)

df_train = pd.concat([X_train, y_train], axis="columns")
df_test = pd.concat([X_test, y_test], axis="columns")

print(f"Number of rows: {df_train.shape[0]}")
print(f"Number of columns: {df_train.shape[1]}",end="\n\n")

print("Missing Values:")
print(df_train.isna().sum())

print(f"Duplicate rows: {df_train.duplicated().sum()}",end="\n\n")

columns_continuous = ['age', 'income', 'number_of_dependants', 'years_at_current_address', 
                      'sanction_amount', 'loan_amount', 'processing_fee', 'gst', 'net_disbursement', 
                      'loan_tenure_months','principal_outstanding', 'bank_balance_at_application',
                      'number_of_open_accounts','number_of_closed_accounts', 'total_loan_months', 'delinquent_months',
                       'total_dpd', 'enquiry_count', 'credit_utilization_ratio']

columns_categorical = ['gender', 'marital_status', 'employment_status', 'residence_type', 'city', 
                       'state', 'zipcode', 'loan_purpose', 'loan_type', 'default']

print("Check for outliers:")
num_plots = len(columns_continuous)
num_cols = 4  # Number of plots per row
num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows needed

fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))  # Adjust the figure size as needed
axes = axes.flatten()  # Flatten the axes array for easier indexing

for i, col in enumerate(columns_continuous):
    sns.boxplot(x=df_train[col], ax=axes[i])
    axes[i].set_title(col)  # Set the title to the name of the variable

# If there are any empty plots (if the number of plots isn't a perfect multiple of num_cols), hide the axes
for j in range(i + 1, num_rows * num_cols):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

for col in columns_categorical:
    print(col, "-->", df_train_1[col].unique())


```
## 🧼 Data Preparation
### Data Cleaning:
```
# Outlier Removal: Processing Fee:
df_train.processing_fee.describe()
df_train[(df_train.processing_fee/df_train.loan_amount)>0.03][["loan_amount","processing_fee"]]
df_train_1 = df_train[df_train.processing_fee/df_train.loan_amount<0.03].copy()

# Apply same step on test set
df_test = df_test[df_test.processing_fee/df_test.loan_amount<0.03].copy()
df_test.shape

# Use other business rules for data validation
print("Rule 1: GST should not be more than 20%")
print(df_train_1[(df_train_1.gst/df_train_1.loan_amount)>0.2].shape)

print("Rule 2: Net disbursement should not be higher than loan_amount")
print(df_train_1[df_train_1.net_disbursement>df_train_1.loan_amount].shape)

# Fix Errors in Loan Purpose Column:
df_train_1['loan_purpose'] = df_train_1['loan_purpose'].replace('Personaal', 'Personal')
print("Unique values in Loan Purpose column - df_train_1")
df_train_1['loan_purpose'].unique()

df_test['loan_purpose'] = df_test['loan_purpose'].replace('Personaal', 'Personal')
print("Unique values in Loan Purpose column - df_test")
df_test['loan_purpose'].unique()
```
## Exploratory Data Analysis
Instructions from **Senior Data Scientist**:

> Skip traditional univariate and bivariate analysis.
> Instead, perform the KDE-based approach to identify strong predictors.
1. Use Kernel Density Estimation (KDE) - Analyze the distribution of each continuous feature using KDE plots.
2. Group by Target Variable - Plot KDEs for each numeric feature grouped by the target variable to observe class-wise distribution.
3. Identify Strong Predictors - Highlight features where the KDE plots show clear separation between target classes. These features are considered potential strong predictors.
4. Document Insights - Summarize key features with class separation and tag them for feature selection or further modeling.


```
# Age Column:
df_train_1.groupby("default")['age'].describe()

# KDE visualization:
plt.figure(figsize=(8, 4))
sns.kdeplot(df_train_1['age'][df_train_1['default'] == 0], fill=True, label='default=0')
sns.kdeplot(df_train_1['age'][df_train_1['default'] == 1], fill=True, label='default=1')
plt.title(f"Age KDE Plot with Hue by default")
plt.legend()
plt.show()
```
**Insights**
1. Average age in the default group is little less (37.12) than the average (39.7) of the group that did not default.
2. Variability (standard deviation) is mostly similar in both the groups.
3. Both the groups have similar min and max ages
4. Orange (defaulted) group is slightly shifted to left indicating that younger folks are more likely to default on their loans.
```
# KDE for all the Columns
plt.figure(figsize=(24, 20))  # Width, height in inches

for i, col in enumerate(columns_continuous):
    plt.subplot(6, 4, i+1)  # 1 row, 4 columns, ith subplot
    sns.kdeplot(df_train_1[col][df_train_1['default']==0], fill=True, label='default=0')
    sns.kdeplot(df_train_1[col][df_train_1['default']==1], fill=True, label='default=1')
    plt.title(col)        
    plt.xlabel('')
    
plt.tight_layout()
plt.show()
```
**Insights**

1. In columns: loan_tenure_months, delinquent_months, total_dpd, credit_utilization, higher values indicate high likelyhood of becoming a default. Hence these 4 looks like strong predictors.
2. In remaining columns the distributions do not give any obvious insights.
3. Why loan_amount and income did not give any signs of being strong predictors? May be when we combine these two and get loan to income ratio (LTI), that may have influence on the target variable. We will explore more later.

## 🔧 Feature Engineering
### Kanban View:
![10 Jira](https://github.com/user-attachments/assets/6e599c80-0a41-451c-94d6-02c2688e64b5)

![Screenshot 2025-04-11 144740](https://github.com/user-attachments/assets/a789fca0-79bb-4991-9b0c-a2e063bcda5e)


### Generate Loan to Income (LTI) Ratio
```
df_train_1['loan_to_income'] = round(df_train_1['loan_amount'] / df_train_1['income'],2)
df_test['loan_to_income'] = round(df_test['loan_amount'] / df_test['income'],2)

# KDE Visualization
plt.figure(figsize=(8, 4))
sns.kdeplot(df_train_1['loan_to_income'][df_train_1['default'] == 0], fill=True, label='default=0')
sns.kdeplot(df_train_1['loan_to_income'][df_train_1['default'] == 1], fill=True, label='default=1')
plt.title(f"Loan to Income Ratio (LTI) KDE Plot with Hue by default")
plt.legend()
plt.show()
```
**Insights**
1. Blue graph has majority of its values on lower side of LTI.
2. Orange graph has many values when LTI is higher indicating that higher LTI means high risk loan.

### Generate Delinquency Ratio
```
df_train_1['delinquency_ratio'] = (df_train_1['delinquent_months']*100 / df_train_1['total_loan_months']).round(1)
df_test['delinquency_ratio'] = (df_test['delinquent_months']*100 / df_test['total_loan_months']).round(1)

# KDE Visualization
plt.figure(figsize=(8, 4))
sns.kdeplot(df_train_1['delinquency_ratio'][df_train_1['default'] == 0], fill=True, label='default=0')
sns.kdeplot(df_train_1['delinquency_ratio'][df_train_1['default'] == 1], fill=True, label='default=1')
plt.title(f"Delinquency Ratio KDE Plot with Hue by default")
plt.legend()
plt.show()
```
**Insights**
1. Blue graph has majority of its values on lower side of LTI
2. Orange graph has many values when delinquency ratio is higher indicating some correlation on default.

### Generate Avg DPD Per Delinquency
```
import numpy as np

df_train_1['avg_dpd_per_delinquency'] = np.where(
    df_train_1['delinquent_months'] != 0,
    (df_train_1['total_dpd'] / df_train_1['delinquent_months']).round(1),
    0
)

df_test['avg_dpd_per_delinquency'] = np.where(
    df_test['delinquent_months'] != 0,
    (df_test['total_dpd'] / df_test['delinquent_months']).round(1),
    0
)

# KDE Visualization
plt.figure(figsize=(8, 4))
sns.kdeplot(df_train_1['avg_dpd_per_delinquency'][df_train_1['default'] == 0], fill=True, label='default=0')
sns.kdeplot(df_train_1['avg_dpd_per_delinquency'][df_train_1['default'] == 1], fill=True, label='default=1')
plt.title(f"Avg DPD Per Delinquency Ratio KDE Plot with Hue by default")
plt.legend()
plt.show()
```
**Insights**
1. Graph clearly shows more occurances of default cases when avg_dpd_per_delinquency is high. This means this column is a strong predictor.

### Feature Removal
```
# Remove columns that are just unique ids and don't have influence on target
df_train_2 = df_train_1.drop(['cust_id', 'loan_id'],axis="columns")
df_test = df_test.drop(['cust_id', 'loan_id'],axis="columns")

# Remove columns that business contact person asked us to remove
drop_col = ['disbursal_date', 'installment_start_dt', 'loan_amount', 'income','total_loan_months', 'delinquent_months', 'total_dpd']

df_train_3 = df_train_2.drop(drop_col, axis="columns")
df_test = df_test.drop(drop_col, axis="columns")

df_train_3.columns
```
### Feature Scaling
```
from sklearn.preprocessing import MinMaxScaler

X_train = df_train_3.drop('default', axis='columns')
y_train = df_train_3['default']

cols_to_scale = X_train.select_dtypes(['int64', 'float64']).columns

scaler = MinMaxScaler()

X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_train.describe()

# Same transformation on test set
X_test = df_test.drop('default', axis='columns')
y_test = df_test['default']

X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
X_test.describe()
```
### Calculate VIF for Multicolinearity
```
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(data):
    vif_df = pd.DataFrame()
    vif_df['Column'] = data.columns
    vif_df['VIF'] = [variance_inflation_factor(data.values,i) for i in range(data.shape[1])]
    return vif_df

calculate_vif(X_train[cols_to_scale])

features_to_drop_vif = ['sanction_amount', 'processing_fee', 'gst', 'net_disbursement','principal_outstanding']

X_train_1 = X_train.drop(features_to_drop_vif, axis='columns')
numeric_columns = X_train_1.select_dtypes(['int64', 'float64']).columns
numeric_columns

vif_df = calculate_vif(X_train_1[numeric_columns])
vif_df

selected_numeric_features_vif = vif_df.Column.values
selected_numeric_features_vif

numeric_columns

plt.figure(figsize=(12,12))
cm = df_train_3[numeric_columns.append(pd.Index(['default']))].corr()
sns.heatmap(cm, annot=True, fmt='0.2f')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
```
### Feature Selection: Categorical Features
#### Calculate Weight of Evidence (WOE) and Information Value (IV)
```
def calculate_woe_iv(df, feature, target):
    grouped = df.groupby(feature)[target].agg(['count','sum'])
    grouped = grouped.rename(columns={'count': 'total', 'sum': 'good'})
    grouped['bad']=grouped['total']-grouped['good']
    
    total_good = grouped['good'].sum()
    total_bad = grouped['bad'].sum()
    
    grouped['good_pct'] = grouped['good'] / total_good
    grouped['bad_pct'] = grouped['bad'] / total_bad
    grouped['woe'] = np.log(grouped['good_pct']/ grouped['bad_pct'])
    grouped['iv'] = (grouped['good_pct'] -grouped['bad_pct'])*grouped['woe']
    
    grouped['woe'] = grouped['woe'].replace([np.inf, -np.inf], 0)
    grouped['iv'] = grouped['iv'].replace([np.inf, -np.inf], 0)
    
    total_iv = grouped['iv'].sum()
    
    return grouped, total_iv

grouped, total_iv = calculate_woe_iv(pd.concat([X_train_1, y_train],axis=1), 'loan_purpose', 'default')
grouped

X_train_1.info()

# IV Values:
iv_values = {}

for feature in X_train_1.columns:
    if X_train_1[feature].dtype == 'object':
        _, iv = calculate_woe_iv(pd.concat([X_train_1, y_train],axis=1), feature, 'default' )
    else:
        X_binned = pd.cut(X_train_1[feature], bins=10, labels=False)
        _, iv = calculate_woe_iv(pd.concat([X_binned, y_train],axis=1), feature, 'default' )
    iv_values[feature] = iv
        
iv_values

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

iv_df = pd.DataFrame(list(iv_values.items()), columns=['Feature', 'IV'])
iv_df = iv_df.sort_values(by='IV', ascending=False)
iv_df

# select features that has IV > 0.02
selected_features_iv = [feature for feature, iv in iv_values.items() if iv > 0.02]
selected_features_iv
```
### Feature Encoding
```
X_train_reduced = X_train_1[selected_features_iv]
X_test_reduced = X_test[selected_features_iv]

X_train_encoded = pd.get_dummies(X_train_reduced, drop_first=True)
X_test_encoded = pd.get_dummies(X_test_reduced, drop_first=True)
```
## Model Development
### Kanban View:
![11 Jira](https://github.com/user-attachments/assets/f7b06d11-f591-49fb-afe2-2c4b566c66f0)

![Screenshot 2025-04-11 144908](https://github.com/user-attachments/assets/3cde8b49-3cd8-4631-b0ca-fce5da87ed29)


### Attempt 1
1. Logistic Regression, RandomForest & XGB
2. No handling of class imbalance
```
# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

model = LogisticRegression()
model.fit(X_train_encoded, y_train)

y_pred = model.predict(X_test_encoded)
report = classification_report(y_test, y_pred)
print(report)

feature_importance = model.coef_[0]

# Create a DataFrame for easier handling
coef_df = pd.DataFrame(feature_importance, index=X_train_encoded.columns, columns=['Coefficients'])

# Sort the coefficients for better visualization
coef_df = coef_df.sort_values(by='Coefficients', ascending=True)

# Plotting
plt.figure(figsize=(8, 4))
plt.barh(coef_df.index, coef_df['Coefficients'], color='steelblue')
plt.xlabel('Coefficient Value')
plt.title('Feature Importance in Logistic Regression')
plt.show()

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train_encoded, y_train)

y_pred = model.predict(X_test_encoded)
report = classification_report(y_test, y_pred)
print(report)

# XGBoost Classifier
from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(X_train_encoded, y_train)

y_pred = model.predict(X_test_encoded)
report = classification_report(y_test, y_pred)
print(report)
```
Since we did not observed any significant difference between between XGB and Logistic, we will choose LogisticRegression as a candidate for our RandomizedSearchCV candidate it has a better interpretation.
#### RandomizedSearch CV for Attempt 1: Logistic Regression
```
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'C': np.logspace(-4, 4, 20),  # Logarithmically spaced values from 10^-4 to 10^4
    'solver': ['lbfgs', 'saga', 'liblinear', 'newton-cg']   # Algorithm to use in the optimization problem
}

# Create the Logistic Regression model
log_reg = LogisticRegression(max_iter=10000)  # Increased max_iter for convergence

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=log_reg,
    param_distributions=param_dist,
    n_iter=50,  # Number of parameter settings that are sampled
    scoring='f1',
    cv=3,  # 5-fold cross-validation
    verbose=2,
    random_state=42,  # Set a random state for reproducibility
    n_jobs=-1  # Use all available cores
)

# Fit the RandomizedSearchCV to the training data
random_search.fit(X_train_encoded, y_train)

# Print the best parameters and best score
print(f"Best Parameters: {random_search.best_params_}")
print(f"Best Score: {random_search.best_score_}")

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test_encoded)
print("Classification Report:")
print(classification_report(y_test, y_pred))
```
#### RandomizedSearch CV for Attempt 1: XGBoost
```
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV

# Define parameter distribution for RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 150, 200, 250, 300],
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'scale_pos_weight': [1, 2, 3, 5, 7, 10],
    'reg_alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],  # L1 regularization term
    'reg_lambda': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]  # L2 regularization term
}

xgb = XGBClassifier()

random_search = RandomizedSearchCV(estimator=xgb, param_distributions=param_dist, n_iter=100,
                                   scoring='f1', cv=3, verbose=1, n_jobs=-1, random_state=42)

random_search.fit(X_train_encoded, y_train)

# Print the best parameters and best score
print(f"Best Parameters: {random_search.best_params_}")
print(f"Best Score: {random_search.best_score_}")

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test_encoded)
print("Classification Report:")
print(classification_report(y_test, y_pred))
```
### Attempt 2
1. Logistic Regression & XGB
2. Handle Class Imbalance Using Under Sampling
```
# Handle Class Imbalance
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = rus.fit_resample(X_train_encoded, y_train)
y_train_res.value_counts()

# Logistic Regression
model = LogisticRegression()
model.fit(X_train_res, y_train_res)

y_pred = model.predict(X_test_encoded)
report = classification_report(y_test, y_pred)
print(report)

# XGBoost Classifier
model = XGBClassifier(**random_search.best_params_)
model.fit(X_train_res, y_train_res)

y_pred = model.predict(X_test_encoded)
report = classification_report(y_test, y_pred)
print(report)
```
### Attempt 3
1. Logistic Regression
2. Handle Class Imbalance Using SMOTE Tomek
3. Parameter tunning using optuna
```
from imblearn.combine import SMOTETomek

smt = SMOTETomek(random_state=42)
X_train_smt, y_train_smt = smt.fit_resample(X_train_encoded, y_train)
y_train_smt.value_counts()

# LogisticRegression
model = LogisticRegression()
model.fit(X_train_smt, y_train_smt)

y_pred = model.predict(X_test_encoded)
report = classification_report(y_test, y_pred)
print(report)

import optuna
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import cross_val_score

# Define the objective function for Optuna
def objective(trial):
    param = {
        'C': trial.suggest_float('C', 1e-4, 1e4, log=True),  # Logarithmically spaced values
        'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga', 'newton-cg']),  # Solvers
        'tol': trial.suggest_float('tol', 1e-6, 1e-1, log=True),  # Logarithmically spaced values for tolerance
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])  # Class weights
    }

    model = LogisticRegression(**param, max_iter=10000)
    
    # Calculate the cross-validated f1_score
    f1_scorer = make_scorer(f1_score, average='macro')
    scores = cross_val_score(model, X_train_smt, y_train_smt, cv=3, scoring=f1_scorer, n_jobs=-1)
    
    return np.mean(scores)

study_logistic = optuna.create_study(direction='maximize')
study_logistic.optimize(objective, n_trials=50)

print('Best trial:')
trial = study_logistic.best_trial
print('  F1-score: {}'.format(trial.value))
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))
    
best_model_logistic = LogisticRegression(**study_logistic.best_params)
best_model_logistic.fit(X_train_smt, y_train_smt)

# Evaluate on the test set
y_pred = best_model_logistic.predict(X_test_encoded)

report = classification_report(y_test, y_pred)
print(report)
```
### Attempt 4

1. XGBoost
2. Handle Class Imbalance Using SMOTE Tomek
3. Parameter tunning using optuna
```
# Define the objective function for Optuna
def objective(trial):
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'verbosity': 0,
        'booster': 'gbtree',
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'eta': trial.suggest_float('eta', 0.01, 0.3),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'max_delta_step': trial.suggest_int('max_delta_step', 0, 10)
    }

    model = XGBClassifier(**param)
    
    # Calculate the cross-validated f1_score
    f1_scorer = make_scorer(f1_score, average='macro')
    scores = cross_val_score(model, X_train_smt, y_train_smt, cv=3, scoring=f1_scorer, n_jobs=-1)
    
    return np.mean(scores)

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective, n_trials=50)

print('Best trial:')
trial = study_xgb.best_trial
print('  F1-score: {}'.format(trial.value))
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))
    
best_params = study_xgb.best_params
best_model_xgb = XGBClassifier(**best_params)
best_model_xgb.fit(X_train_smt, y_train_smt)

# Evaluate on the test set
y_pred = best_model_xgb.predict(X_test_encoded)

report = classification_report(y_test, y_pred)
print(report)
```
## Model Evaluation
### ROC/AUC
```
y_pred = best_model_logistic.predict(X_test_encoded)
report = classification_report(y_test, y_pred)
print(report)

from sklearn.metrics import roc_curve

probabilities = best_model_logistic.predict_proba(X_test_encoded)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, probabilities)

fpr[:5], tpr[:5], thresholds[:5]

from sklearn.metrics import auc

area = auc(fpr, tpr)
area

# Plotting
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % area)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
```
### Rankordering, KS statistic, Gini coeff
```
probabilities = best_model_logistic.predict_proba(X_test_encoded)[:,1]

df_eval = pd.DataFrame({
    'Default Truth': y_test,
    'Default Probability': probabilities
})
df_eval.head()

df_eval['Decile'] = pd.qcut(df_eval['Default Probability'], 10, labels=False, duplicates='drop')
df_eval.head()

df_eval[df_eval.Decile==8]['Default Probability'].describe()

df_decile = df_eval.groupby('Decile').apply(lambda x: pd.Series({
    'Minimum Probability': x['Default Probability'].min(),
    'Maximum Probability': x['Default Probability'].max(),
    'Events': x['Default Truth'].sum(),
    'Non-events': x['Default Truth'].count() - x['Default Truth'].sum(),    
}))
df_decile.reset_index(inplace=True)
df_decile

df_decile['Event Rate'] = df_decile['Events']*100 / (df_decile['Events']+df_decile['Non-events'])
df_decile['Non-event Rate'] = df_decile['Non-events']*100 / (df_decile['Events']+df_decile['Non-events'])
df_decile

df_decile = df_decile.sort_values(by='Decile', ascending=False).reset_index(drop=True)
df_decile

df_decile['Cum Events'] =  df_decile['Events'].cumsum()
df_decile['Cum Non-events'] =  df_decile['Non-events'].cumsum()
df_decile

df_decile['Cum Event Rate'] = df_decile['Cum Events'] * 100 / df_decile['Events'].sum()
df_decile['Cum Non-event Rate'] = df_decile['Cum Non-events']*100 / df_decile['Non-events'].sum()
df_decile
```
To assess whether rank ordering is followed, we should look at whether higher deciles (those with higher predicted probabilities) have higher event rates compared to lower deciles. Rank ordering means that as you move from the top decile to the bottom decile, the event rate should generally decrease.

Non-Events - termed as good (customers) who do not default. Events - termed as bad (customers) who default.

Events and Non-Events terms are interchanged based on usecase to usecase.

eg: for marketing usecase, which customer to reach out (who will take loan based on offers) will be events - here it will termed as good, and the customers who will not take loans will be non-events (bad).

**Insights from the Decile Table**

1. Top Deciles

* The first decile (Decile 9) has a high event rate of 72.00% and a non-event rate of 28.00%. This indicates that the model is highly confident in predicting events in this decile.
* The second decile (Decile 8) also shows a significant event rate of 12.72%, with a cumulative event rate reaching 98.6%.

2. Middle Deciles:

* Deciles 7 and 6 show a significant drop in event rates

3. Lower Deciles:

* Deciles 5 to 0 show zero events, with all predictions being non-events. These deciles collectively have a non-event rate of 100%.

4. KS Statistic:

* The KS statistic, which is the maximum difference between cumulative event rates and cumulative non-event rates, is highest at Decile 8 with a value of 85.98%. This suggests that the model performs best at distinguishing between events and non-events up to this decile.

* The KS value gradually decreases in the following deciles, indicating a decrease in model performance for distinguishing between events and non-events.

**KS Value**

The highest KS value is 85.98%, found at Decile 8. This indicates that the model's performance in distinguishing between events and non-events is most significant at this decile. (If KS is in top 3 decile and score above 40, it is considered a good predictive model.)
```
gini_coefficient = 2 * area - 1

print("AUC:", area)
print("Gini Coefficient:", gini_coefficient)
```
AUC of 0.98: The model is very good at distinguishing between events and non-events.

Gini coefficient of 0.96: This further confirms that the model is highly effective in its predictions, with almost perfect rank ordering capability.

The Gini coefficient ranges from -1 to 1, where a value closer to 1 signifies a perfect model, 0 indicates a model with no discriminative power, and -1 signifies a perfectly incorrect model.

#### Finalize The Model and Visualize Feature Importance
```
final_model = best_model_logistic

feature_importance = final_model.coef_[0]

# Create a DataFrame for easier handling
coef_df = pd.DataFrame(feature_importance, index=X_train_encoded.columns, columns=['Coefficients'])

# Sort the coefficients for better visualization
coef_df = coef_df.sort_values(by='Coefficients', ascending=True)

# Plotting
plt.figure(figsize=(8, 4))
plt.barh(coef_df.index, coef_df['Coefficients'], color='steelblue')
plt.xlabel('Coefficient Value')
plt.title('Feature Importance in Logistic Regression')
plt.show()
```
### Export the Model
```
from joblib import dump

model_data = {
    'model': final_model,
    'features': X_train_encoded.columns,
    'scaler': scaler,
    'cols_to_scale': cols_to_scale
}
dump(model_data, 'artifacts/model_data.joblib')
```
## Streamlit Application
### Kanban View:

![12 Jira](https://github.com/user-attachments/assets/294b8798-c523-4d05-b00f-999d01e14809)

![Screenshot 2025-04-11 145142](https://github.com/user-attachments/assets/547ab9e5-a60d-42cd-b861-349d1ead7e1c)


### Prediction Helper:
```
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Path to the saved model and its components
MODEL_PATH = 'artifacts/model_data.joblib'

# Load the model and its components
model_data = joblib.load(MODEL_PATH)
model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']
cols_to_scale = model_data['cols_to_scale']


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

    # Ensure the DataFrame contains only the features expected by the model
    df = df[features]

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
```
### Main File
```
import streamlit as st
from prediction_helper import predict  # Ensure this is correctly linked to your prediction_helper.py

# Set the page configuration and title
st.set_page_config(page_title="Lauki Finance: Credit Risk Modelling", page_icon="📊")
st.title("Lauki Finance: Credit Risk Modelling")

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
```
## Reflections
1. This project provided a hands-on experience with the ML pipeline in the finance domain-specifically focused on credit risk modeling.
2. We took deliberate steps to prevent train-test contamination and data leakage, maintaining the integrity of model evaluation from start to finish.
3. One of the most impactful aspects was the engineering of domain-specific features like the Loan to Income (LTI) Ratio, Delinquency Ratio, and Average Days Past Due (DPD) Per Delinquency.
4. We also applied Weight of Evidence (WOE) and Information Value (IV) to ensure the selected variables not only contributed to the model but also aligned with established credit risk practices.
5. SMOTE-Tomek proved valuable to address class imbalance and used Optuna for hyperparameter fine-tuning.
6. Evaluation went beyond accuracy, incorporating ROC/AUC, Rank ordering, KS statistic, and the Gini coefficient.
7. Finally, deployed the model via Streamlit and transformed it into an interactive tool for business users.


## Conclusion
In conclusion, the Credit Risk Model succeeded in its Phase 1 goal — delivering a scalable, data-driven solution that can provide predictions on default probabilities and credit ratings.

