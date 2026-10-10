# Customer Churn Prediction — Telco Customer Analytics

An end-to-end machine learning project that predicts which telecom customers are likely to churn: exploratory analysis, model comparison, cross-validated tuning, customer segmentation, and a deployed Streamlit app (**Churn Risk Advisor**) with single-customer scoring, what-if analysis and batch scoring.

🔗 **Live app:** https://customer-churn-prediction-raheel-ijaz.streamlit.app/

## Results at a glance

| | |
|---|---|
| Final model | Tuned XGBoost, chosen by 5-fold cross-validation on the training set |
| CV AUC | 0.850 ± 0.012 |
| Test AUC | 0.847 (test set used once) |
| Contact threshold | 0.15, from a cost analysis (missed churner = PKR 6,000, wasted contact = PKR 1,000) |
| Honest caveat | XGBoost beats Logistic Regression by only 0.004 AUC, less than one standard deviation, so this is a mild preference, not a decisive win |

## Dataset
- Source: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Size: 7,043 customers, 21 columns (19 customer attributes, `customerID`, and the `Churn` target)
- Churn rate: 26.5%

## Files
- `week1_eda.ipynb` - Exploratory data analysis
- `week2_ml_models.ipynb` - Model training, evaluation and feature engineering
- `week3_optimization.ipynb` - Cross-validation, tuning (LR / RF / XGBoost), K-means segmentation, PCA, final pipeline
- `week4_packaging.ipynb` - Packaging the model, training-serving skew demo, parity test, what-if analysis, metadata
- `app.py` - Streamlit app (Churn Risk Advisor)
- `requirements.txt` - Pinned dependencies (the same scikit-learn and XGBoost versions used for training)
- `churn_model.joblib` - Final trained pipeline loaded by the app
- `model_meta.json` - Metadata the app reads: feature columns, library versions, threshold, metrics
- `model_metadata.json` - Documentation of the deployed model (generated from the same run as `model_meta.json`)
- `sample_customers.csv` - 50 customers for trying batch scoring
- `best_churn_model.pkl` - Legacy model from the first version of the Week 3 notebook; no longer loaded by the app
- `customer_data.csv` - Full dataset (not included, download separately, see Setup)

## Week 1: Exploratory Data Analysis

6 visualizations covering tenure, monthly charges, total charges, contract type, internet service, payment method, and feature correlations.

### Key findings
- Month-to-month contract customers churn far more than one-year or two-year customers
- Customers with low tenure (under 6 months) are at the highest risk
- Higher monthly charges correlate with higher churn
- Fiber optic customers churn more than DSL customers
- Electronic check users show the highest churn rate among payment methods

## Week 2: Machine Learning Models

| Model | Accuracy |
|---|---|
| Logistic Regression | 80.41% |
| Decision Tree | 79.42% |
| **Random Forest (best)** | **80.70%** |

Top predictive features: tenure, fiber optic internet, total charges, electronic check payment.

Four engineered features (TotalRevenue, TotalServices, TenureGroup, HighCharges) lowered accuracy from 80.70% to 78.92%. They were largely redundant with existing features, a useful negative result.

## Week 3: Model Optimization and Unsupervised Learning

Cross-validated evaluation, tuning of three model families, customer segmentation and PCA. The test set is locked away at the start and used exactly once at the end.

- **Split noise:** the same Logistic Regression scored between **78.0% and 82.8%** accuracy across 20 random splits (std 1.04 points, theoretical standard error 1.07 points). One split cannot rank models that are this close.
- **5-fold CV (mean ± std):**

| Model | AUC | Recall |
|---|---|---|
| Logistic Regression | 0.846 ± 0.013 | 0.545 ± 0.042 |
| Random Forest | 0.844 ± 0.011 | 0.496 ± 0.019 |
| **XGBoost (tuned)** | **0.850 ± 0.012** | 0.536 ± 0.032 |

- **Tuning:** validation curve for `C` (plateau from C ≈ 0.3); grid vs random search for Random Forest (both 120 fits, 163 s vs 155 s, same score within noise); XGBoost early stopping (220 trees) and random search over 30 candidates.
- **Final model (chosen by CV, tested once):** tuned XGBoost, test AUC **0.847**, recall 0.527, precision 0.650, accuracy 79.91% at the default 0.5 cutoff. The test AUC lies inside the CV mean ± 2 std.
- **Customer segments (K-means, k = 4):**

| Segment | Customers | Churn rate |
|---|---|---|
| Mid-tenure, high spend, few add-ons | 2,157 | **43%** |
| New, low-spend starters | 1,918 | 32% |
| Loyal power users | 1,938 | 14% |
| Loyal budget customers | 1,030 | 5% |

- **PCA:** 15 of 30 components explain 90% of the variance. PC1 exposed eight duplicate "No internet service" dummy columns.
- **Key learnings:** tuning barely moved the score (0.846 to 0.850 AUC, inside one standard deviation). Class weighting (`scale_pos_weight`) raised recall from 0.54 to 0.81 at the cost of precision, which matters more for a retention campaign than a few tenths of a point of accuracy.

## Week 4: From Notebook to Product

The Week 3 pipeline is packaged and served as **Churn Risk Advisor**, a public Streamlit app.

### A real bug, found and fixed
Training used `pd.get_dummies(drop_first=True)`. On a **single** customer row that drops the only category present, so both `Contract` columns become 0 and the model silently reads the customer as month-to-month. No error is raised, the probability is just wrong.

The fix is a separate serving encoder (`prepare_input`): no `drop_first`, then reindex to the exact training columns saved with the model. A **parity test** (`week4_packaging.ipynb`) compares the notebook's probabilities with the serving path for all 7,043 customers, and the test passes.

### What the app does
- **One customer:** sidebar inputs, churn probability, risk band (HIGH if the probability is at or above the threshold, WATCH from half the threshold, otherwise LOW) and a suggested action. `TotalCharges` is derived from tenure × monthly charges rather than typed.
- **What would change the risk?:** a what-if table (contract, payment method, tech support), labelled as associations learned from data, not guaranteed effects.
- **Batch scoring:** upload a CSV with the original Telco columns, see how many customers are above the threshold, and download the scored file. Missing columns give a clear error message.
- **About:** model name, library versions, metrics and limitations.

The contact threshold is adjustable in the sidebar and starts at the value stored in `model_meta.json` (0.15).

## Model card: Churn Risk Advisor v1.0

**Intended use:** rank existing telecom customers by churn risk so a retention team can prioritize calls. Decision support, not automation.

**Not for:** credit, pricing, or any decision that denies a service.

**Data:** IBM Telco Customer Churn, 7,043 customers, 26.5% churn.

**Model:** XGBoost (tuned), chosen by 5-fold CV. Features: 30 encoded columns.

**Performance:** CV AUC 0.8504 ± 0.0121; test AUC 0.8472 (test set used once).

**Threshold:** 0.15, from a cost analysis on out-of-fold training predictions (FN = PKR 6,000, FP = PKR 1,000). The test set was not used to choose it.

**Limitations:** one US dataset from one period; no Pakistan data; associations, not causes; performance may drift as plans change.

**Fairness check (Week 15):** compare recall across gender and seniors.

**Owner and version:** Muhammad Raheel Ijaz, v1.0, 10 October 2026.

## Known limitations
- At a threshold of 0.15 the app flags a large share of customers (28 of the 50 sample customers). A real team's calling capacity would call for a higher threshold.
- The what-if table lists every other contract option, including ones that raise a customer's risk.
- The "Why this score?" explanation is not available because the final model is XGBoost, not Logistic Regression.
- The what-if results describe patterns in historical data; offering a customer a different contract would not necessarily make them stay.

## Run locally (Windows, Command Prompt)

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. Tested with Python 3.13. `requirements.txt` pins the scikit-learn and XGBoost versions used for training, because a pickled model can misbehave under different versions.

## Setup for the notebooks

```
pip install pandas numpy matplotlib seaborn scipy jupyter scikit-learn xgboost streamlit
jupyter notebook week1_eda.ipynb
jupyter notebook week2_ml_models.ipynb
jupyter notebook week3_optimization.ipynb
jupyter notebook week4_packaging.ipynb
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and save it as `customer_data.csv` in the project folder before running any notebook. Run `week3_optimization.ipynb` before `week4_packaging.ipynb`, because Week 4 loads the pipeline Week 3 saves.

## Next steps
- Run the fairness check across gender and senior citizens
- Pick the threshold from a real contact budget instead of cost alone
- Record a 1-2 minute demo video of the app
- Project 2: Document Intelligence System
