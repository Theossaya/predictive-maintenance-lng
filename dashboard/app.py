import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# --------------------------
# Load Models
# --------------------------
cmapss_model = joblib.load("models/cmapss_rf_baseline.pkl")
pronostia_model = joblib.load("models/pronostia_rf_baseline.pkl")

st.title("🛠 Predictive Maintenance Dashboard")
st.markdown("Upload sensor data to predict **RUL (compressors)** or **failure risk (pumps/bearings)**.")

# --------------------------
# File Upload
# --------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.write(df.head())

    # --------------------------
    # Compressor Mode (CMAPSS-like)
    # --------------------------
    if "RUL" in df.columns or "sensor_2" in df.columns:
        st.subheader("Compressor RUL Prediction")
        X = df.drop(columns=["unit_number","time_in_cycles","RUL"], errors="ignore")
        y_pred = cmapss_model.predict(X)
        st.write(f"Predicted RUL (first 5 samples): {y_pred[:5]}")

        # Visualization
        plt.figure(figsize=(8,4))
        plt.plot(y_pred[:200])
        plt.title("Predicted RUL Trend (first 200 cycles)")
        st.pyplot(plt)

    # --------------------------
    # Pump Mode (PRONOSTIA-like)
    # --------------------------
    if "RMS_Horiz_accel" in df.columns or "RMS_Vert_accel" in df.columns:
        st.subheader("Bearing Failure Classification")
        X = df.drop(columns=["file","label"], errors="ignore")
        y_pred = pronostia_model.predict(X)
        df["Predicted_Label"] = y_pred
        st.write(df[["Predicted_Label"]].head())

        failure_rate = df["Predicted_Label"].mean()
        st.metric("Predicted Failure Probability", f"{failure_rate*100:.1f}%")

        # Visualization
        plt.figure(figsize=(8,4))
        plt.plot(df["RMS_Horiz_accel"], label="Horiz RMS")
        plt.plot(df["RMS_Vert_accel"], label="Vert RMS")
        plt.legend()
        plt.title("RMS Vibration Trend")
        st.pyplot(plt)
