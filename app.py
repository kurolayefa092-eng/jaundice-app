import streamlit as st
import joblib
import numpy as np
import threading
import requests
import time

def keep_alive():
    while True:
        time.sleep(270)  # every 4.5 minutes
        try:
            requests.get("https://neonatal-jaundiceseveritypredictionapp.streamlit.app")
        except:
            pass

thread = threading.Thread(target=keep_alive)
thread.daemon = True
thread.start()
# Load model and scaler
model = joblib.load('jaundice_severity_model.pkl')
scaler = joblib.load('jaundice_severity_scaler.pkl')

# Page config
st.set_page_config(
    page_title="Neonatal Jaundice Severity Predictor",
    page_icon="🏥",
    layout="centered"
)

# Background colour
st.markdown('<style>.stApp {background: linear-gradient(135deg, #f5f7fa, #c3cfe2);} h1,h2,h3,p,label,.stMarkdown{color:#1a1a2e !important;} .stButton>button{background-color:#0066cc !important;color:white !important;border:none !important;border-radius:10px !important;font-size:17px !important;font-weight:bold !important;padding:14px !important;}</style>', unsafe_allow_html=True)

# Title
st.title("🏥 Neonatal Jaundice Severity Predictor (NJS-P)")
st.markdown("A clinical decision support tool for neonatal jaundice severity prediction")
st.divider()

# Patient Demographics
st.subheader("Patient Demographics")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age at Admission (days)", min_value=0, max_value=30, value=0)
    birth_weight = st.number_input("Birth Weight (kg)", min_value=0.0, max_value=6.0, value=0.0, step=0.1)
    gest_age = st.number_input("Gestational Age (weeks)", min_value=0.0, max_value=44.0, value=0.0, step=0.1)

with col2:
    haemoglobin = st.number_input("Haemoglobin (g/dL)", min_value=0.0, max_value=25.0, value=0.0, step=0.1)
    wcc = st.number_input("White Cell Count (B/liters)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
    neutrophil = st.number_input("Neutrophil Count (B/liters)", min_value=0.0, max_value=20.0, value=0.0, step=0.1)

# Birth Weight Category
st.subheader("Birth Weight Category")
birth_weight_cat = st.selectbox(
    "Select Birth Weight Category",
    options=[0, 1, 2],
    format_func=lambda x: {
        0: "Normal (≥ 2.5 kg)",
        1: "Low Birth Weight (< 2.5 kg)",
        2: "Macrosomia (> 4 kg)"
    }[x]
)

# Haematological Markers
st.subheader("Haematological Markers")
col3, col4 = st.columns(2)

with col3:
    lymphocyte = st.number_input("Lymphocyte Count (B/liters)", min_value=0.0, max_value=20.0, value=0.0, step=0.1)

with col4:
    platelet = st.number_input("Platelet Count (B/liters)", min_value=0.0, max_value=600.0, value=0.0, step=1.0)

# Bilirubin Measurements
st.subheader("Bilirubin Measurements")
col5, col6 = st.columns(2)

with col5:
    direct_bil = st.number_input("Direct Bilirubin (mg/dL)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)

with col6:
    indirect_bil = st.number_input("Indirect Bilirubin (mg/dL)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)

st.divider()

# Predict button
if st.button("Predict Severity", use_container_width=True):

    # Input validation
    errors = []

    if age <= 0:
        errors.append("Age at Admission must be greater than 0")
    if birth_weight <= 0:
        errors.append("Birth Weight must be greater than 0")
    if gest_age <= 0:
        errors.append("Gestational Age must be greater than 0")
    if haemoglobin <= 0:
        errors.append("Haemoglobin must be greater than 0")
    if wcc <= 0:
        errors.append("White Cell Count must be greater than 0")
    if neutrophil <= 0:
        errors.append("Neutrophil Count must be greater than 0")
    if lymphocyte <= 0:
        errors.append("Lymphocyte Count must be greater than 0")
    if platelet <= 0:
        errors.append("Platelet Count must be greater than 0")
    if direct_bil <= 0 and indirect_bil <= 0:
        errors.append("At least one Bilirubin value must be greater than 0")

    if errors:
        st.warning("Please correct the following before predicting:")
        for error in errors:
            st.markdown("- " + error)

    else:
        # Calculate features automatically at backend

        # Age 24hr category — less than 24hrs = 0, greater than 24hrs = 1
        age_24hr_encoded = 0 if age <= 1 else 1

        # Gestational age in days
        gest_age_days = round(gest_age * 7, 1)

        # IBP in percentage
        if (direct_bil + indirect_bil) > 0:
            IBP = round((indirect_bil / (direct_bil + indirect_bil)) * 100, 0)
        else:
            IBP = 0

        # Bilirubin per day
        if age > 0:
            bilirubin_per_day = round((direct_bil + indirect_bil) / age, 2)
        else:
            bilirubin_per_day = 0

        # Show calculated indicators
        st.subheader("Calculated Clinical Indicators")
        col7, col8 = st.columns(2)

        with col7:
            st.metric(
                label="Indirect Bilirubin Proportion (IBP)",
                value=str(IBP) + "%",
                help="Percentage of total bilirubin that is unconjugated"
            )

        with col8:
            st.metric(
                label="Bilirubin Accumulation Rate",
                value=str(bilirubin_per_day) + " mg/dL/day",
                help="Average daily bilirubin accumulation since birth"
            )

        st.divider()

        # Prepare input in correct feature order — 15 features
        input_data = np.array([[
            age,                # Age_At_Admission_In_Days
            age_24hr_encoded,   # Age_24Hr_Category_encode
            birth_weight_cat,   # Birth_weight_Category_encode
            birth_weight,       # Birth_Weight_In_kg
            gest_age_days,      # Gestational_Age_In_Days
            gest_age,           # Gestational_Age_In_Weeks
            haemoglobin,        # Haemoglobin_In_g/dl
            wcc,                # White_Cell_Count_In_B/liters
            neutrophil,         # Neutrophil_Count_In_B/liters
            lymphocyte,         # Lymphocyte_Count_In_B/liters
            platelet,           # Platelet_Count_In_B/liters
            indirect_bil,       # indirect_bil_mgdl
            direct_bil,         # direct_bil_mgdl
            IBP,                # IBP_in_percentage
            bilirubin_per_day   # Bilirubin_Per_Day
        ]])

        # Scale input — required for Logistic Regression
        input_data_scaled = scaler.transform(input_data)

        # Get prediction and probability
        prediction = model.predict(input_data_scaled)[0]
        probability = model.predict_proba(input_data_scaled)[0]

        # Show prediction result
        st.subheader("Prediction Result")

        if prediction == 'Severe':
            st.error("SEVERE JAUNDICE DETECTED")
            st.markdown("**Immediate clinical intervention is required.**")
            st.markdown("**Recommended Actions:**")
            st.markdown("- Initiate phototherapy immediately")
            st.markdown("- Monitor bilirubin levels closely")
            st.markdown("- Evaluate for exchange transfusion if no improvement")
            st.markdown("- Escalate to specialist if available")
            st.metric("Confidence", str(round(probability[1] * 100, 1)) + "%")

        else:
            st.success("NON-SEVERE JAUNDICE")
            st.markdown("**Continue monitoring and reassess if condition changes.**")
            st.markdown("**Recommended Actions:**")
            st.markdown("- Continue routine monitoring")
            st.markdown("- Reassess bilirubin levels regularly")
            st.markdown("- Ensure adequate feeding")
            st.markdown("- Return immediately if symptoms worsen")
            st.metric("Confidence", str(round(probability[0] * 100, 1)) + "%")

st.divider()
st.caption("This tool is intended for research and decision support purposes only. It does not replace clinical judgement. All assessments should be validated by a qualified healthcare professional.")
