# Customer Churn Prediction

An end-to-end machine learning project analyzing and predicting customer churn using the Telco Customer Churn dataset — from exploratory analysis through model optimization and live deployment.

## Dataset
- Source: Telco Customer Churn Dataset
- Size: 7,043 customers, 21 features
- Target: Predict customer churn (Yes/No)

## Files
- `week1_eda.ipynb` - Exploratory data analysis notebook
- `week2_ml_models.ipynb` - Machine learning models: training, evaluation, and feature engineering
- `week3_optimization.ipynb` - Cross-validation, hyperparameter tuning, and XGBoost optimization
- `app.py` - Streamlit web app for real-time churn prediction
- `requirements.txt` - Python dependencies for the Streamlit app
- `customer_data.csv` - Dataset (not included, download separately — see Setup)
- `best_churn_model.pkl` - Final tuned model, saved for deployment
- `model_metadata.json` - Metadata (params, metrics, feature list) for the saved model

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

## Week 3: Model Optimization

Cross-validation, GridSearchCV hyperparameter tuning, and XGBoost were used to systematically optimize the Week 2 models.

### Cross-Validation (5-Fold)
- Mean accuracy: **78.59%** (± 1.23%)
- Confirms the single train/test split wasn't overly optimistic — CV mean closely matches the baseline test accuracy

### Hyperparameter Tuning Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Baseline Random Forest | 78.78% | 62.54% | 50.00% | 55.57% |
| Optimized Random Forest | 80.13% | 66.55% | 50.53% | 57.45% |
| Basic XGBoost | 79.77% | 63.95% | 54.55% | 58.87% |
| **Optimized XGBoost (best)** | **80.27%** | 66.22% | 52.41% | 58.51% |

**Best Random Forest parameters:** `max_depth=10, max_features='log2', min_samples_leaf=4, min_samples_split=2, n_estimators=300`

**Best XGBoost parameters:** `colsample_bytree=0.8, learning_rate=0.1, max_depth=3, n_estimators=100, subsample=0.8`

### Top Predictive Features (Optimized XGBoost)
1. Internet Service (Fiber optic)
2. Contract (Two year)
3. Online Security (No internet service)
4. Payment Method (Electronic check)
5. Contract (One year)

### Key Learnings
- Hyperparameter tuning improved Random Forest by 1.35 points and gave XGBoost a slight edge over RF overall
- The model's false-negative rate (47.6%) is notably high — it misses roughly half of actual churners, which matters more than raw accuracy for a retention use case
- Contract type and internet service type are stronger churn drivers than billing amount or usage volume
- Further gains would likely come from feature engineering (contract × internet-service interactions) or addressing class imbalance, rather than more tuning

### Final Model
- **Optimized XGBoost**, 80.27% test accuracy
- Saved as `best_churn_model.pkl` with parameters and metrics logged in `model_metadata.json`
- Note: this fell short of an 85% stretch target common for this dataset; see the notebook's summary section for suggested next steps to close that gap

## Week 4: Interactive Deployment (Streamlit)

A live, interactive web app that loads the tuned XGBoost model from Week 3 and predicts churn risk for any customer profile in real time.

### Live App
🔗 [Add your Streamlit Cloud URL here after deployment]

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

### How to use it
1. Fill in the customer's demographics, services, and account details using the form.
2. Click **Predict Churn**.
3. The app shows:
   - A high-risk / low-risk verdict with the churn probability
   - A gauge chart visualizing the risk level
   - Specific retention recommendations based on the customer's profile (contract type, payment method, tenure, internet service)

## Next Steps
- Explore class-imbalance handling (e.g., SMOTE, class weighting) and engineered interaction features to push accuracy higher
- Add a 1-2 minute demo video showing the app in action
- Consider Project 2: Document Intelligence System

## Setup
```bash
pip install pandas numpy matplotlib seaborn jupyter scikit-learn xgboost streamlit plotly
jupyter notebook week1_eda.ipynb
jupyter notebook week2_ml_models.ipynb
jupyter notebook week3_optimization.ipynb
streamlit run app.py
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and save it as `customer_data.csv` in the project folder before running any notebook.
