
import streamlit as st
import pandas as pd
import requests
import joblib
import os
import csv
from datetime import date, timedelta, datetime
from sklearn.linear_model import LinearRegression
from requests.auth import HTTPBasicAuth

APP_TITLE = "☀️ Solar Forecast - ROBOTRONIX for IMEPOWER"
LOGIN_USER = "FVMANAGER"
LOGIN_PASS = "MIMMOFABIO"

DATA_PATH = "Dataset_Daily_EnergiaSeparata_2020_2025.csv"
MODEL_PATH = "pv_model.joblib"
LOG_FILE = "forecast_log.csv"

DEFAULT_LAT = 40.6432780
DEFAULT_LON = 16.9860830

METEO_USER = "teseospa-eiffageenergiesystemesitaly_daniello_fabio"
METEO_PASS = "6S8KTHPbrUlp6523T9Xd"

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

def load_dataset():
    if not os.path.exists(DATA_PATH):
        st.error("Dataset mancante")
        return None
    return pd.read_csv(DATA_PATH, parse_dates=["Date"])

def train_model(df):
    df = df.dropna(subset=["E_INT_Daily_kWh","G_M0_Wm2"])
    X, y = df[["G_M0_Wm2"]], df["E_INT_Daily_kWh"]
    model = LinearRegression()
    model.fit(X,y)
    joblib.dump(model, MODEL_PATH)
    return model

def get_meteomatics(lat, lon, target_date):
    base_url = "https://api.meteomatics.com"
    start = f"{target_date.isoformat()}T00:00:00Z"
    end   = f"{target_date.isoformat()}T23:45:00Z"
    params = "direct_rad:W,total_cloud_cover:p"
    interval = "PT15M"
    url = f"{base_url}/{start}--{end}:{interval}/{params}/{lat},{lon}/json"
    r = requests.get(url, auth=HTTPBasicAuth(METEO_USER,METEO_PASS), timeout=20)
    r.raise_for_status()
    data = r.json()
    times = [t["validdate"] for t in data["data"][0]["coordinates"][0]["dates"]]
    rad   = [t["value"] for t in data["data"][0]["coordinates"][0]["dates"]]
    clouds= [t["value"] for t in data["data"][1]["coordinates"][0]["dates"]]
    df = pd.DataFrame({"time":pd.to_datetime(times),"direct_rad":rad,"cloud_cover":clouds}).set_index("time")
    df["G_eff"] = df["direct_rad"].clip(lower=0) * (1 - df["cloud_cover"]/100.0)
    return df

def get_openmeteo(lat, lon, target_date):
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&hourly=shortwave_radiation,cloudcover&timezone=auto"
           f"&start_date={target_date}&end_date={target_date}")
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    data = r.json()
    rad = data["hourly"]["shortwave_radiation"]
    cc  = data["hourly"]["cloudcover"]
    hrs = pd.date_range(start=str(target_date)+" 00:00", periods=len(rad), freq="H")
    df = pd.DataFrame({"time":hrs,"direct_rad":rad,"cloud_cover":cc}).set_index("time")
    df = df.resample("15T").interpolate()
    df["G_eff"] = df["direct_rad"].clip(lower=0) * (1 - df["cloud_cover"]/100.0)
    return df

def predict_from_curve(model, df_curve):
    if df_curve["G_eff"].sum() <= 0:
        return 0.0, pd.Series(0.0, index=df_curve.index)
    rad_mean = df_curve["G_eff"].mean()
    daily_kwh = float(model.predict(pd.DataFrame({"G_M0_Wm2":[rad_mean]}))[0])
    kwh_curve = (df_curve["G_eff"]/df_curve["G_eff"].sum()) * daily_kwh
    return daily_kwh, kwh_curve

def write_log(day_label, day_date, provider, status, lat, lon, error_msg=""):
    row = {
        "timestamp": datetime.now().isoformat(),
        "day_label": day_label,
        "day_date": str(day_date),
        "provider": provider,
        "status": status,
        "lat": lat,
        "lon": lon,
        "error": error_msg
    }
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

st.set_page_config(page_title="PV Forecast Dashboard", layout="wide")
st.markdown(f"<h1 style='text-align:center;color:orange'>{APP_TITLE}</h1>", unsafe_allow_html=True)
st.write("---")

df = load_dataset()
if df is None: st.stop()

model = train_model(df)
st.success("✅ Modello addestrato sul dataset storico (CSV)")

st.header("📊 Storico")
st.line_chart(df.set_index("Date")[["E_INT_Daily_kWh","G_M0_Wm2"]])

st.header("🔮 Previsioni Ieri + Oggi + Domani + Dopodomani (Meteomatics con fallback Open-Meteo)")
lat = st.number_input("Latitudine", value=DEFAULT_LAT, format="%.6f")
lon = st.number_input("Longitudine", value=DEFAULT_LON, format="%.6f")

if st.button("Calcola previsioni"):
    days = {"Ieri": -1, "Oggi": 0, "Domani": 1, "Dopodomani": 2}
    results = {}
    for label, offset in days.items():
        d = date.today()+timedelta(days=offset)
        try:
            df15 = get_meteomatics(lat,lon,d)
            st.success(f"✅ Meteomatics usato per {label} ({d})")
            write_log(label, d, "Meteomatics", "OK", lat, lon)
        except Exception as e:
            st.warning(f"⚠️ Meteomatics non disponibile per {label} ({d}), uso Open-Meteo. Errore: {e}")
            write_log(label, d, "Meteomatics", "ERROR", lat, lon, str(e))
            df15 = get_openmeteo(lat,lon,d)
            write_log(label, d, "Open-Meteo", "OK", lat, lon)
        kwh, curve = predict_from_curve(model, df15)
        results[label] = (d, kwh, curve)
    for label in results:
        st.metric(f"Produzione {label} ({results[label][0]})", f"{results[label][1]:.1f} kWh")
        st.subheader(f"📈 Curva 15-min prevista per {label} ({results[label][0]})")
        st.line_chart(results[label][2])
