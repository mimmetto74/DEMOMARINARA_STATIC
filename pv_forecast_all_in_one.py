import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# ==================
# Config login
# ==================
USER = "FVMANAGER"
PASS = "MIMMOFABIO"

# ==================
# Dataset storico
# ==================
DATA_PATH = "Dataset_Daily_EnergiaSeparata_2020_2025.csv"

# ==================
# Meteomatics API
# ==================
MET_USER = "teseospa-eiffageenergiesystemesitaly_daniello_fabio"
MET_PASS = "6S8KTHPbrUlp6523T9Xd"

# ==================
# Funzioni
# ==================
def check_login():
    st.title("🔐 Accesso richiesto")
    user = st.text_input("Username")
    pw = st.text_input("Password", type="password")
    if st.button("Login"):
        if user == USER and pw == PASS:
            st.session_state["auth"] = True
        else:
            st.error("❌ Credenziali non valide")
            st.stop()

def train_model():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.dropna(subset=["E_INT_Daily_kWh", "G_M0_Wm2"])
    X = df[["G_M0_Wm2"]]
    y = df["E_INT_Daily_kWh"]
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, "pv_model.joblib")
    return model, df

def get_meteomatics_forecast(lat, lon):
    base_url = "https://api.meteomatics.com"
    start = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
    end   = (datetime.utcnow() + timedelta(days=2)).strftime("%Y-%m-%dT23:59:00Z")
    params = "global_rad:W,m cloud_cover:octas"
    url = f"{base_url}/{start}--{end}:PT24H/{params}/{lat},{lon}/json"
    r = requests.get(url, auth=(MET_USER, MET_PASS))
    r.raise_for_status()
    data = r.json()
    rad = [float(v["value"]) for v in data["data"][0]["coordinates"][0]["dates"]]
    cloud = [float(v["value"]) for v in data["data"][1]["coordinates"][0]["dates"]]
    dates = [v["date"] for v in data["data"][0]["coordinates"][0]["dates"]]
    return pd.DataFrame({"Date": pd.to_datetime(dates), "G_M0_Wm2": rad, "CloudCover": cloud})

def predict_production(model, df_forecast):
    X_future = df_forecast[["G_M0_Wm2"]]
    y_pred = model.predict(X_future)
    df_forecast["E_INT_Pred_kWh"] = y_pred
    return df_forecast

# ==================
# UI principale
# ==================
if "auth" not in st.session_state:
    st.session_state["auth"] = False

if not st.session_state["auth"]:
    check_login()
    st.stop()

st.title("☀️ Solar Forecast - ROBOTRONIX for IMEPOWER")

# Training
st.header("📊 Analisi Storica & Training")
model, df_hist = train_model()
st.success("✅ Modello addestrato con dati storici")

st.line_chart(df_hist.set_index("Date")[["E_INT_Daily_kWh", "G_M0_Wm2"]])

# Forecast
st.header("🔮 Previsioni Meteomatics (Domani + Dopodomani)")
lat = st.number_input("Latitudine", value=40.643278)
lon = st.number_input("Longitudine", value=16.986083)

if st.button("Genera Previsione"):
    df_forecast = get_meteomatics_forecast(lat, lon)
    df_pred = predict_production(model, df_forecast)
    st.write(df_pred[["Date", "E_INT_Pred_kWh", "CloudCover"]])
    st.line_chart(df_pred.set_index("Date")[["E_INT_Pred_kWh", "G_M0_Wm2"]])
