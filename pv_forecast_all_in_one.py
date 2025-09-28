import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# ============================
# Credenziali Meteomatics (trial)
# ============================
MM_USER = "teseospa-eiffageenergiesystemesitaly_daniello_fabio"
MM_PASS = "6S8KTHPbrUlp6523T9Xd"

# ============================
# Funzione per chiamata Meteomatics
# ============================
def get_meteomatics_forecast(lat, lon, start, end):
    url = (
        f"https://api.meteomatics.com/{start}--{end}:PT1H/"
        f"global_rad:W,total_cloud_cover:p/{lat},{lon}/json"
    )
    resp = requests.get(url, auth=(MM_USER, MM_PASS), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame()
    for param in data["data"]:
        name = param["parameter"]
        df[name] = [float(e["value"]) for e in param["coordinates"][0]["dates"]]
    df["time"] = [e["date"] for e in data["data"][0]["coordinates"][0]["dates"]]
    df["time"] = pd.to_datetime(df["time"])
    return df

# ============================
# Interfaccia Streamlit
# ============================
st.title("☀️ Solar Forecast - ROBOTRONIX for IMEPOWER")

lat = st.number_input("Latitudine", value=40.643278, format="%.6f")
lon = st.number_input("Longitudine", value=16.986083, format="%.6f")

if st.button("Genera Previsione"):
    today = datetime.utcnow().date()
    start = f"{today}T00:00:00Z"
    end = f"{today + timedelta(days=1)}T00:00:00Z"

    try:
        df = get_meteomatics_forecast(lat, lon, start, end)
        st.success("✅ Dati Meteomatics ricevuti")
        st.line_chart(df.set_index("time")["global_rad:W"])
    except Exception as e:
        st.error(f"❌ Meteomatics non disponibile: {e}")
