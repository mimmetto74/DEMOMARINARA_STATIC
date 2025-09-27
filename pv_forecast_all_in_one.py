
import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression

APP_TITLE = "☀️ Solar Forecast - ROBOTRONIX for IMEPOWER"
LOGIN_USER = "FVMANAGER"
LOGIN_PASS = "MIMMOFABIO"

DATA_PATH = "Dataset_Daily_EnergiaSeparata_2020_2025.csv"
MODEL_PATH = "pv_model.joblib"

DEFAULT_LAT = 40.6432780
DEFAULT_LON = 16.9860830

# ---------------- Login -----------------
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    if st.session_state.password_correct:
        return True
    st.title("🔒 Accesso richiesto")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username.strip().upper() == LOGIN_USER and password.strip() == LOGIN_PASS:
            st.session_state.password_correct = True
            st.rerun()
        else:
            st.error("❌ Credenziali non valide")
    return False

if not check_password():
    st.stop()

# ---------------- Dataset & Model -----------------
def load_dataset():
    if not os.path.exists(DATA_PATH):
        st.error("Dataset mancante")
        return None
    return pd.read_csv(DATA_PATH, parse_dates=["Date"])

def train_model(df):
    df = df.dropna(subset=["E_INT_Daily_kWh","G_M0_Wm2"])
    X, y = df[["G_M0_Wm2"]], df["E_INT_Daily_kWh"]
    if len(X) < 2:
        st.error("⚠️ Dataset troppo piccolo per l'addestramento (servono almeno 2 righe).")
        return None
    model = LinearRegression()
    model.fit(X,y)
    joblib.dump(model, MODEL_PATH)
    return model

def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None

# ---------------- Open-Meteo provider -----------------
def get_openmeteo(lat, lon, target_date):
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&hourly=shortwave_radiation,cloudcover&timezone=auto"
           f"&start_date={target_date}&end_date={target_date}")
    r = requests.get(url, timeout=25); r.raise_for_status(); data=r.json()
    rad = data["hourly"]["shortwave_radiation"]
    cc  = data["hourly"]["cloudcover"]
    hrs = pd.date_range(start=str(target_date)+" 00:00", periods=len(rad), freq="H")
    df = pd.DataFrame({"time":hrs,"shortwave_radiation":rad,"cloudcover":cc}).set_index("time")
    # upsample to 15-min
    df = df.resample("15T").interpolate()
    # effective irradiance
    df["G_eff"] = df["shortwave_radiation"].clip(lower=0) * (1 - df["cloudcover"]/100.0)
    return df

# ---------------- Prediction -----------------
def predict_from_curve(model, df_curve):
    if model is None:
        return 0.0, pd.Series(0.0, index=df_curve.index) if df_curve is not None else (0.0, pd.Series(dtype=float))
    if df_curve["G_eff"].sum() <= 0:
        return 0.0, pd.Series(0.0, index=df_curve.index)
    rad_mean = df_curve["G_eff"].mean()
    daily_kwh = float(model.predict(pd.DataFrame({"G_M0_Wm2":[rad_mean]}))[0])
    kwh_curve = (df_curve["G_eff"]/df_curve["G_eff"].sum()) * daily_kwh
    return daily_kwh, kwh_curve

# ---------------- UI -----------------
st.set_page_config(page_title="PV Forecast Dashboard", layout="wide")
st.markdown(f"<h1 style='text-align:center;color:orange'>{APP_TITLE}</h1>", unsafe_allow_html=True)
st.write("---")

df = load_dataset()
if df is None: st.stop()

model = train_model(df)
if model is None: st.stop()
st.success("✅ Modello addestrato sul dataset storico (CSV)")

st.header("📊 Storico")
st.line_chart(df.set_index("Date")[["E_INT_Daily_kWh","G_M0_Wm2"]])

st.header("🔮 Previsioni Ieri + Oggi + Domani + Dopodomani (Open-Meteo)")
lat = st.number_input("Latitudine", value=DEFAULT_LAT, format="%.6f")
lon = st.number_input("Longitudine", value=DEFAULT_LON, format="%.6f")

if st.button("Calcola previsioni"):
    days = {"Ieri": -1, "Oggi": 0, "Domani": 1, "Dopodomani": 2}
    results = {}

    for label, offset in days.items():
        d = date.today()+timedelta(days=offset)
        df15 = get_openmeteo(lat,lon,d)
        kwh, curve = predict_from_curve(model, df15)
        results[label] = (d, kwh, curve)

    # Metriche
    for label in results:
        st.metric(f"Produzione {label} ({results[label][0]})", f"{results[label][1]:.1f} kWh")
        st.subheader(f"📈 Curva 15-min prevista per {label} ({results[label][0]})")
        st.line_chart(results[label][2])

    # Download unico CSV con tutte le curve
    out = pd.DataFrame({"Datetime":[], "kWh_15min":[], "Day":[]})
    for label in results:
        curve = results[label][2]
        out = pd.concat([out, pd.DataFrame({"Datetime":curve.index, "kWh_15min":curve.values, "Day":[str(results[label][0])]*len(curve)})])
    st.download_button("⬇️ Scarica tutte le curve (CSV)", out.to_csv(index=False).encode("utf-8"), file_name="forecast_openmeteo_4days.csv", mime="text/csv")
