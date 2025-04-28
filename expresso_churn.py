import streamlit as st
import pandas as pd
import pickle


# --- Load model ---
try:
    with open('expresso_churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file 'expresso_churn_model.pkl' not found.")
    st.stop()

# --- Load feature columns ---
try:
    with open('expresso_churn_model.pkl', 'rb') as f:
        feature_names = pickle.load(f)
except FileNotFoundError:
    st.error("Feature column file 'expresso_columns.pkl' not found.")
    st.stop()

# --- Load encoders ---
try:
    with open('expresso_churn_model.pkl', 'rb') as f:
        encoders = pickle.load(f)
except FileNotFoundError:
    st.error("Encoders file 'encoders.pkl' not found.")
    st.stop()

st.title("Expresso Churn Prediction")
st.subheader("Enter customer features to predict churn probability")

# --- Input Fields ---
tenure_options = ['K > 24 month', 'J 21-24 month', 'I 18-21 month', 'H 15-18 month',
                  'G 12-15 month', 'F 9-12 month', 'E 6-9 month', 'D 3-6 month',
                  'C 1-3 month', 'B 0-1 month', 'A < 1 month']
tenure = st.selectbox('TENURE', tenure_options)

montant = st.number_input('MONTANT', value=0.0)
frequence_rech = st.number_input('FREQUENCE_RECH', value=0.0)
revenue = st.number_input('REVENUE', value=0.0)
arpu_segment = st.number_input('ARPU_SEGMENT', value=0.0)
frequence = st.number_input('FREQUENCE', value=0.0)
data_volume = st.number_input('DATA_VOLUME', value=0.0)
on_net = st.number_input('ON_NET', value=0.0)
orange = st.number_input('ORANGE', value=0.0)
tigo = st.number_input('TIGO', value=0.0)
regularity = st.number_input('REGULARITY', value=0.0)

top_pack_options = ['No_Top_Pack', 'other', 'Data C', 'All Net 500MB Day', 'Data E', 'Data D']
top_pack = st.selectbox('TOP_PACK', top_pack_options)

freq_top_pack = st.number_input('FREQ_TOP_PACK', value=0.0)

region_options = ['Dakar', 'Thiès', 'Saint-Louis', 'Kaolack', 'Ziguinchor', 'Diourbel']
region = st.selectbox('REGION', region_options)

mrg_options = ['NO', 'YES']
mrg = st.selectbox('MRG', mrg_options)

# --- Predict Button ---
if st.button('Predict Churn'):
    # Construct DataFrame
    input_data = pd.DataFrame({
        'MONTANT': [montant],
        'FREQUENCE_RECH': [frequence_rech],
        'REVENUE': [revenue],
        'ARPU_SEGMENT': [arpu_segment],
        'FREQUENCE': [frequence],
        'DATA_VOLUME': [data_volume],
        'ON_NET': [on_net],
        'ORANGE': [orange],
        'TIGO': [tigo],
        'REGULARITY': [regularity],
        'FREQ_TOP_PACK': [freq_top_pack],
        'TENURE': [tenure],
        'TOP_PACK': [top_pack],
        'REGION': [region],
        'MRG': [mrg],
        'DATA_VOLUME_MISSING': [0],  # Example placeholder if used during training
    })

    # Apply label encoding using loaded encoders
    try:
        input_data['TENURE'] = encoders['TENURE'].transform(input_data['TENURE'])
        input_data['TOP_PACK'] = encoders['TOP_PACK'].transform(input_data['TOP_PACK'])
        input_data['REGION'] = encoders['REGION'].transform(input_data['REGION'])
        input_data['MRG'] = encoders['MRG'].transform(input_data['MRG'])
    except Exception as e:
        st.error(f"Encoding error: {e}")
        st.stop()

    # Align input with training feature order
    try:
        input_data = input_data[feature_names]
    except KeyError as e:
        st.error(f"Missing expected feature: {e}")
        st.stop()

    # --- Predict ---
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1][0]

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.warning(f" This customer is likely to churn (Probability: {probability:.2f})")
    else:
        st.success(f"This customer is unlikely to churn (Probability: {probability:.2f})")