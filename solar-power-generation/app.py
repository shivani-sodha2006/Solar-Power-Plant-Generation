import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# --- Custom Page Config ---
st.set_page_config(page_title="Solar Power Forecast", page_icon="🔆", layout="centered")

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
        body {
            background-color: #0f1117;
            color: white;
        }
        .main {
            background-color: #1e1e2f;
            padding: 20px;
            border-radius: 12px;
        }
        h1, h2, h3, label {
            color: #00c0ff;
        }
        .stButton > button {
            background-color: #00c0ff;
            color: white;
            border-radius: 10px;
            padding: 0.6em 1.2em;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton > button:hover {
            background-color: #0090c0;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load Model and Scaler ---
model = joblib.load("solar_Power_Generation_Forecasting_model.pkl")
scaler = joblib.load("scaler.pkl")

# --- App Title ---
st.markdown("<h1 style='text-align: center;'>🔆 Solar DC Power Forecasting</h1>", unsafe_allow_html=True)
st.write("Provide environmental conditions and date/time to **predict the solar DC power output**.")

# --- Input Section ---
with st.container():
    st.subheader("📅 Date & Time Input")
    col1, col2 = st.columns(2)
    with col1:
        selected_date = st.date_input("Select Date", value=datetime(2020, 6, 15).date())
    with col2:
        selected_time = st.time_input("Select Time", value=datetime(2020, 6, 15, 14, 0).time())
    input_time = datetime.combine(selected_date, selected_time)

st.subheader("🌤 Environmental Conditions")
col3, col4, col5 = st.columns(3)
with col3:
    IRRADIATION = st.number_input("☀️ Irradiation (W/m²)", min_value=0.0, step=0.1, format="%.2f")
with col4:
    MODULE_TEMPERATURE = st.number_input("🌡 Module Temp (°C)", min_value=0.0, step=0.1, format="%.2f")
with col5:
    AMBIENT_TEMPERATURE = st.number_input("🌫 Ambient Temp (°C)", min_value=0.0, step=0.1, format="%.2f")

# --- Prediction ---
if st.button("⚡ Predict DC Power"):
    try:
        # Extract time features
        hour = input_time.hour
        day = input_time.day
        month = input_time.month
        day_of_week = input_time.weekday()

        # Prepare DataFrame for model
        input_df = pd.DataFrame([[IRRADIATION, MODULE_TEMPERATURE, AMBIENT_TEMPERATURE, hour, day, month, day_of_week]],
                                columns=['IRRADIATION', 'MODULE_TEMPERATURE', 'AMBIENT_TEMPERATURE', 'HOUR', 'DAY', 'MONTH', 'DAY_OF_WEEK'])

        # Scale and predict
        input_scaled = scaler.transform(input_df)
        predicted_dc_power = model.predict(input_scaled)[0]

        st.success(f"### ⚡ Predicted DC Power: `{predicted_dc_power:.2f} kW`")
    except Exception as e:
        st.error(f"🚫 An error occurred: {e}")
