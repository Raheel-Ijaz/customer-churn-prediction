import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title='Customer Churn Predictor',
    page_icon='📊',
    layout='wide'
)

st.title('📊 Customer Churn Prediction System')
st.markdown('Enter a customer\'s details below to estimate their risk of churning.')


@st.cache_resource
def load_model():
    with open('best_churn_model.pkl', 'rb') as file:
        return pickle.load(file)


@st.cache_resource
def load_metadata():
    with open('model_metadata.json', 'r') as file:
        return json.load(file)


model = load_model()
metadata = load_metadata()
feature_columns = metadata['features']

st.success('Model loaded successfully!')

with st.expander('Model info'):
    st.write(f"Model type: {metadata.get('model_type', 'N/A')}")
    st.write(f"Test accuracy: {metadata.get('accuracy', 0) * 100:.2f}%")
    st.write(f"Precision: {metadata.get('precision', 0) * 100:.2f}%")
    st.write(f"Recall: {metadata.get('recall', 0) * 100:.2f}%")
    st.write(f"F1-score: {metadata.get('f1_score', 0) * 100:.2f}%")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader('Demographics')
    gender = st.selectbox('Gender', ['Male', 'Female'])
    senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
    partner = st.selectbox('Partner', ['No', 'Yes'])
    dependents = st.selectbox('Dependents', ['No', 'Yes'])

with col2:
    st.subheader('Services')
    phone_service = st.selectbox('Phone Service', ['No', 'Yes'])
    multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    online_backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
    device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
    tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
    streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
    streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])

with col3:
    st.subheader('Account')
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox('Paperless Billing', ['No', 'Yes'])
    payment_method = st.selectbox(
        'Payment Method',
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
    )
    tenure = st.slider('Tenure (months)', 0, 72, 12)
    monthly_charges = st.number_input('Monthly Charges ($)', min_value=0.0, max_value=200.0, value=70.0)
    total_charges = st.number_input(
        'Total Charges ($)', min_value=0.0, max_value=10000.0,
        value=round(tenure * monthly_charges, 2)
    )

st.divider()

if st.button('🔮 Predict Churn', type='primary'):

    # Build a single-row dataframe matching the RAW column names from training
    input_data = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    input_df = pd.DataFrame([input_data])

    # Encode exactly like Week 2/3 training preprocessing
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                         'MultipleLines', 'InternetService', 'OnlineSecurity',
                         'OnlineBackup', 'DeviceProtection', 'TechSupport',
                         'StreamingTV', 'StreamingMovies', 'Contract',
                         'PaperlessBilling', 'PaymentMethod']

    input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

    # Align columns EXACTLY to what the model was trained on.
    # Any dummy column not present in this single row is filled with 0.
    # Any column the model doesn't expect is dropped.
    input_final = input_encoded.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(input_final)[0]
    probability = model.predict_proba(input_final)[0]
    churn_prob = probability[1] * 100

    result_col, gauge_col = st.columns(2)

    with result_col:
        if prediction == 1:
            st.error('⚠️ HIGH RISK: Customer likely to churn')
            st.metric('Churn Probability', f'{churn_prob:.1f}%')
        else:
            st.success('✅ LOW RISK: Customer likely to stay')
            st.metric('Retention Probability', f'{100 - churn_prob:.1f}%')

        st.subheader('Recommendations')
        recs = []
        if prediction == 1:
            if contract == 'Month-to-month':
                recs.append('Offer an incentive to switch to a 1- or 2-year contract.')
            if internet_service == 'Fiber optic':
                recs.append('Review fiber pricing and service quality — fiber customers churn more.')
            if payment_method == 'Electronic check':
                recs.append('Encourage a switch to automatic payment (bank transfer or credit card).')
            if tenure < 6:
                recs.append('New customers are highest-risk — consider early-tenure outreach.')
            if online_security == 'No':
                recs.append('Bundle in online security — customers without it churn more often.')
            if not recs:
                recs.append('Flag account for proactive retention outreach.')
        else:
            recs.append('No action needed — customer profile indicates low churn risk.')

        for r in recs:
            st.write(f'- {r}')

    with gauge_col:
        fig = go.Figure(go.Indicator(
            mode='gauge+number',
            value=churn_prob,
            number={'suffix': '%'},
            title={'text': 'Churn risk'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': 'darkred' if churn_prob >= 50 else 'darkgreen'},
                'steps': [
                    {'range': [0, 30], 'color': '#d4f7d4'},
                    {'range': [30, 60], 'color': '#fff3cd'},
                    {'range': [60, 100], 'color': '#f8d7da'}
                ],
                'threshold': {
                    'line': {'color': 'black', 'width': 3},
                    'thickness': 0.8,
                    'value': churn_prob
                }
            }
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info('Fill in the customer details above, then click "Predict Churn".')
