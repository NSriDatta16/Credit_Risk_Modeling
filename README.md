# 🏦 Credit Risk Modeling using Machine Learning

<p align="center">

---

# 📌 Project Overview

Credit risk assessment is one of the most important applications of Machine Learning in the banking and financial industry.

Every time a customer applies for a loan, the bank must decide whether the applicant is likely to repay the loan or default. An incorrect decision can result in significant financial losses.

The objective of this project is to build an end-to-end Machine Learning pipeline that predicts whether a borrower is likely to default on a loan using demographic, financial, and credit-related information.

This project follows an industry-standard workflow from raw data to a production-ready machine learning model.

---

# 🎯 Business Problem

Banks receive thousands of loan applications every day.

Approving a loan for a risky customer can lead to financial losses.

Rejecting a genuine customer results in missed business opportunities.

The objective is to build a predictive model that helps financial institutions make better lending decisions.

---

# 🎯 Project Objective

Develop a Machine Learning model capable of predicting:

* **0 → No Default**
* **1 → Default**

using borrower information such as:

* Age
* Income
* Employment Length
* Home Ownership
* Loan Amount
* Loan Intent
* Loan Grade
* Interest Rate
* Credit History
* Previous Defaults

---

# 📂 Dataset Information

Dataset Size

* Rows: **32,581**
* Columns: **12**

Target Variable

```text
loan_status

0 → No Default

1 → Default
```

Features include:

* person_age
* person_income
* person_home_ownership
* person_emp_length
* loan_intent
* loan_grade
* loan_amnt
* loan_int_rate
* loan_percent_income
* cb_person_default_on_file
* cb_person_cred_hist_length

---

# 📁 Project Structure

```text
Credit-Risk-Modeling
│
├── data/
│   └── credit_risk_dataset.csv
│
├── notebooks/
│   └── Credit_Risk_Modeling_End_To_End.ipynb
│
├── src/
│
├── artifacts/
│
├── models/
│
├── reports/
│
├── images/
│
├── README.md
├── requirements.txt
├── .gitignore
└── main.py
```

---

# 🔄 Machine Learning Workflow

```
Business Understanding
        ↓
Data Understanding
        ↓
Exploratory Data Analysis
        ↓
Missing Value Treatment
        ↓
Feature Engineering
        ↓
Feature Selection
        ↓
Train/Test Split
        ↓
Preprocessing Pipeline
        ↓
Baseline Models
        ↓
Hyperparameter Tuning
        ↓
Model Evaluation
        ↓
Explainable AI (SHAP)
        ↓
Final Model
        ↓
Deployment
```

---

# 📊 Exploratory Data Analysis

Completed:

* Dataset Inspection
* Missing Value Analysis
* Univariate Analysis
* Bivariate Analysis
* Correlation Analysis

Key observations:

* Dataset contains missing values in employment length and interest rate.
* Loan percentage of income shows the strongest positive relationship with default.
* Higher interest rates are associated with higher default probability.
* Customers living in rented homes have a higher observed default rate.
* Extremely high age values indicate the presence of outliers requiring further investigation.

---

# 🛠 Technologies Used

Programming

* Python

Libraries

* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn
* XGBoost
* LightGBM
* CatBoost
* Optuna
* SHAP

Development Tools

* VS Code
* Git
* GitHub
* Jupyter Notebook

---

---

---

# 🎯 Future Improvements

* Advanced Feature Engineering
* Model Explainability using SHAP
* Hyperparameter Optimization using Optuna
* Streamlit Web Application
* Docker Support
* Azure Deployment
* CI/CD Pipeline

---
