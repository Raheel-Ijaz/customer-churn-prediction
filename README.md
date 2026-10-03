# Customer Churn Prediction

An end-to-end machine learning project analyzing and predicting customer churn using the Telco Customer Churn dataset — from exploratory analysis through model optimization and live deployment.

## Dataset
- Source: Telco Customer Churn Dataset
- Size: 7,043 customers, 21 features
- Target: Predict customer churn (Yes/No)

## Files
- `week1_eda.ipynb` - Exploratory data analysis notebook
- `week2_ml_models.ipynb` - Machine learning models: training, evaluation, and feature engineering
- `week3_optimization.ipynb` - Cross-validation, tuning (LR/RF/XGBoost), K-means segmentation, PCA, final pipeline
- `app.py` - Streamlit web app for real-time churn prediction
- `requirements.txt` - Python dependencies for the Streamlit app
- `customer_data.csv` - Dataset (not included, download separately — see Setup)
- `best_churn_model.pkl` - Final tuned model, saved for deployment
- `model_metadata.json` - Metadata (params, metrics, feature list) for the saved model
- `churn_model.joblib` - Week 3 final pipeline (XGBoost, chosen by CV, tested once)
- `churn_model_metadata.json` - CV/test metrics and feature list for the Week 3 pipeline

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

## Week 3: Model Optimization and Unsupervised Learning

Honest, cross-validated evaluation, tuning of three model families, customer segmentation, and PCA. The test set is created once and untouched until the final step.

- **Split noise:** the same Logistic Regression scored between **78.0% and 82.8%** accuracy across 20 random splits (std 1.04 pts, theoretical SE 1.07 pts), so a single split cannot rank close models.
- **5-fold CV (mean ± std):**

| Model | AUC | Recall |
|---|---|---|
| Logistic Regression | 0.846 ± 0.013 | 0.545 ± 0.042 |
| Random Forest | 0.844 ± 0.011 | 0.496 ± 0.019 |
| **XGBoost (tuned)** | **0.850 ± 0.012** | 0.536 ± 0.032 |

- **Tuning:** validation curve for `C` (plateau from C ≈ 0.3), grid vs random search for Random Forest (both 120 fits, 130 s vs 145 s, same score within noise), XGBoost early stopping (247 trees) and random search (30 candidates).
- **Final model (chosen by CV, tested once):** tuned XGBoost, **test AUC 0.848**, recall 0.521, precision 0.659, accuracy 80.13%, inside CV mean ± 2 std.
- **Customer segments (K-means, k = 4):** Mid-tenure high spend (**43%** churn), New low-spend starters (32%), Loyal power users (14%), Loyal budget (5%), each with a retention action in the notebook.
- **PCA:** 15 of 30 components explain 90% of the variance; PC1 exposed duplicate "No internet service" dummy columns.
- **Key learnings:** tuning barely moved the score (0.846 → 0.850 AUC, inside one std). Class weighting (`scale_pos_weight`) raised recall from 0.54 to 0.81 at the cost of precision, which matters more for a retention use case than 0.4 points of accuracy.
- Saved as `churn_model.joblib` (full pipeline) with `churn_model_metadata.json`. The live app (Week 4) still uses `best_churn_model.pkl`.

## Week 4: Interactive Deployment (Streamlit)

A live, interactive web app that loads the tuned XGBoost model from Week 3 and predicts churn risk for any customer profile in real time.

### Live App
🔗 https://customer-churn-prediction-raheel-ijaz.streamlit.app/

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
