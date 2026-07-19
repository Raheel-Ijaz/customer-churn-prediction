# Customer Churn Prediction

A machine learning project analyzing and predicting customer churn using the Telco Customer Churn dataset.

## Dataset
- Source: Telco Customer Churn Dataset
- Size: 7,043 customers, 21 features
- Target: Predict customer churn (Yes/No)

## Files
- `week1_eda.ipynb` - Exploratory data analysis notebook
- `week2_ml_models.ipynb` - Machine learning models: training, evaluation, and feature engineering
- `customer_data.csv` - Dataset (not included, download separately — see Setup)

## Week 1: Exploratory Data Analysis

6 visualizations covering tenure, monthly charges, total charges, contract type, internet service, payment method, and feature correlations.

### Key Findings
- Month-to-month contract customers churn far more than yearly/two-year contract customers
- Customers with low tenure (< 6 months) are at the highest risk of churning
- Higher monthly charges correlate with higher churn
- Fiber optic internet customers churn more than DSL customers
- Electronic check payment users show the highest churn rate among payment methods

## Week 2: Machine Learning Models

Three classification models trained and compared, plus feature engineering experiments.

### Model Comparison

| Model | Accuracy |
|---|---|
| Logistic Regression | 80.41% |
| Decision Tree | 79.42% |
| **Random Forest (best)** | **80.70%** |

### Top Predictive Features
1. Tenure
2. Internet Service (Fiber optic)
3. Total Charges
4. Payment Method (Electronic check)

### Feature Engineering
Four engineered features were tested (TotalRevenue, TotalServices, TenureGroup, HighCharges). Accuracy slightly decreased (80.70% → 78.92%), suggesting the new features were largely redundant with existing ones rather than adding new predictive signal — a useful negative result that informs future feature selection.

## Next Steps
- Week 3: Hyperparameter tuning and cross-validation
- Explore additional feature combinations
- Handle class imbalance if needed
- Week 4: Deploy an interactive prediction dashboard

## Setup
```bash
pip install pandas numpy matplotlib seaborn jupyter scikit-learn
jupyter notebook week1_eda.ipynb
jupyter notebook week2_ml_models.ipynb
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and save it as `customer_data.csv` in the project folder before running either notebook.
