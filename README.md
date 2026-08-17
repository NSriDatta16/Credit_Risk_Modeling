# Credit Risk Modeling using Machine Learning

End-to-end credit default prediction project covering exploratory analysis, feature engineering, model training, production inference, Flask API, Docker, and CI validation.

## Business Problem

Predict whether a borrower is likely to default on a loan. The target is `loan_status`:

- `0` → No Default
- `1` → Default

Dataset: 32,581 rows and 12 original columns.

## Production Architecture

```text
Applicant data
    ↓
Flask REST API / Web UI
    ↓
Saved scikit-learn Pipeline
    ↓
Preprocessing + model
    ↓
Default probability
    ↓
LOW / MEDIUM / HIGH risk
```

The Docker image trains the production artifact during image build so the container is self-contained and reproducible.

## Project Structure

```text
Credit_Risk_Modeling/
├── data/
│   └── credit_risk_dataset.csv
├── notebooks/
│   └── Credit_Risk_Modeling_End_To_End.ipynb
├── src/
│   ├── train.py
│   └── predictor.py
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── main.py
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── requirements-prod.txt
└── .github/workflows/docker.yml
```
