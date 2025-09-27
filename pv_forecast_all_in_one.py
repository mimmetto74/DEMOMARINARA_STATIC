import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os
from datetime import date, timedelta, datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from requests.auth import HTTPBasicAuth

# ===============================
# Login semplice
# ===============================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    if st.session_state.password_correct:
        return True
    st.title("🔒 Accesso richiesto")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "FVMANAGER" and password == "MIMMOFABIO":
            st.session_state.password_correct = True
            st.rerun()
        else:
            st.error("❌ Credenziali non valide")
    return False

if not check_password():
    st.stop()

# ===============================
# Config dataset e modello
# ===============================
DATA_PATH = "Dataset_Daily_EnergiaSeparata_2020_2025.csv"
MODEL_PATH = "pv_model.joblib"

DEFAULT_LAT = 40.6432780
DEFAULT_LON = 16.9860830

# Credenziali Meteomatics hardcoded
METEO_USER = "teseospa-eiffageenergiesystemesitaly_daniello_fabio"
METEO_PASS = "6S8KTHPbrUlp6523T9Xd"

# ===============================
# Funzioni utili
# ===============================
def load_dataset():
    if not os.path.exists(DATA_PATH):
        st.error(f"⚠️ Dataset non trovato: {DATA_PATH}")
        return None
    return pd.read_csv(DATA_PATH, parse_dates=["Date"])

def train_model(df):
    df = df.dropna(subset=["E_INT_Daily_kWh", "G_M0_Wm2"])
    train = df[df["Date"] < "2025-01-01"]
    test = df[df["Date"] >= "2025-01-01"]
    X_train, y_train = train[["G_M0_Wm2"]], train["E_INT_Daily_kWh"]
    X_test, y_test = test[["G_M0_Wm2"]], test["E_INT_Daily_kWh"]
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    joblib.dump(model, MODEL_PATH)
    return mae, r2

def load_model():
    return joblib.load(MODEL_PATH)

def forecast_day_ahead(irr):
    model = load_model()
    return float(model.predict(pd.DataFrame({"G_M0_Wm2":[irr]}))[0])

def get_meteomatics_forecast(lat, lon, days_ahead=1):
    base_url = "https://api.meteomatics.com"
    start = (datetime.utcnow() + timedelta(days=days_ahead)).strftime("%Y-%m-%dT00:00:00Z")
    end   = (datetime.utcnow() + timedelta(days=days_ahead)).strftime("%Y-%m-%dT23:45:00Z")
    url = f"{base_url}/{start}--{end}:PT15M/solar_rad:W,cloud_cover:pc/{lat},{lon}/json"
    r = requests.get(url, auth=HTTPBasicAuth(METEO_USER, METEO_PASS))
    r.raise_for_status()
    data = r.json()
    times=[t["validdate"] for t in data["data"][0]["coordinates"][0]["dates"]]
    rad=[t["value"] for t in data["data"][0]["coordinates"][0]["dates"]]
    cld=[t["value"] for t in data["data"][1]["coordinates"][0]["dates"]]
    return pd.DataFrame({"time":pd.to_datetime(times),"solar_rad":rad,"cloud_cover":cld}).set_index("time")

def get_forecast_irradiance(lat, lon, days_ahead=1):
    target_date = (date.today() + timedelta(days=days_ahead)).isoformat()
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=shortwave_radiation&timezone=auto&start_date={target_date}&end_date={target_date}"
    r = requests.get(url); r.raise_for_status(); data=r.json()
    irr = data["hourly"]["shortwave_radiation"]
    hrs = pd.date_range(start=target_date+" 00:00",periods=len(irr),freq="H")
    df=pd.DataFrame({"Ora":hrs,"Irraggiamento":irr}).set_index("Ora").resample("15T").interpolate()
    return float(df["Irraggiamento"].mean()),df

def estimate_power_curve(irr_series, daily_prod_forecast):
    irr=irr_series.values; tot=irr.sum()
    if tot==0: return pd.Series(np.zeros_like(irr),index=irr_series.index)
    return pd.Series((irr/tot)*daily_prod_forecast,index=irr_series.index)

# ===============================
# UI
# ===============================
st.set_page_config(page_title="PV Forecast Dashboard", layout="wide")
st.title("☀️ Solar Forecast - ROBOTRONIX for IMEPOWER")

st.header("📊 Analisi Storica")
df=load_dataset()
if df is not None:
    df=df.rename(columns={"E_INT_kWh":"E_INT_Daily_kWh","E_Z_EVU_kWh":"E_Z_EVU_Daily_kWh"})
    start_date,end_date=st.date_input("Intervallo date",[df["Date"].min().date(),df["Date"].max().date()])
    mask=(df["Date"]>=pd.to_datetime(start_date))&(df["Date"]<=pd.to_datetime(end_date))
    st.line_chart(df.loc[mask].set_index("Date")[["E_INT_Daily_kWh","G_M0_Wm2"]])

if st.button("Addestra modello con dati storici") and df is not None:
    mae,r2=train_model(df)
    st.success(f"MAE={mae:.1f} R²={r2:.2f}")

st.header("🔮 Previsione FV")
lat=st.number_input("Latitudine",value=DEFAULT_LAT)
lon=st.number_input("Longitudine",value=DEFAULT_LON)
giorno=st.selectbox("Giorno",["Domani","Dopodomani"])
days_ahead=1 if giorno=="Domani" else 2

if st.button("Calcola previsione"):
    try:
        dfm=get_meteomatics_forecast(lat,lon,days_ahead)
        st.success("✅ Previsione basata su Meteomatics (solar_rad + cloud_cover)")
        irr=dfm["solar_rad"].mean()
        prod=forecast_day_ahead(irr)
        prod*=1-0.5*(dfm["cloud_cover"].mean()/100)
        df_plot=pd.DataFrame({
            "Irraggiamento":dfm["solar_rad"],
            "Nuvolosità %":dfm["cloud_cover"],
            "Produzione stimata":(dfm["solar_rad"]/dfm["solar_rad"].sum())*prod
        })
    except Exception as e:
        st.warning("⚠️ Meteomatics non disponibile, uso Open-Meteo")
        irr,dfi=get_forecast_irradiance(lat,lon,days_ahead)
        prod=forecast_day_ahead(irr)
        df_plot=pd.DataFrame({
            "Irraggiamento":dfi["Irraggiamento"],
            "Produzione stimata":estimate_power_curve(dfi["Irraggiamento"],prod)
        })
    st.metric("Produzione stimata",f"{prod:.1f} kWh")
    st.line_chart(df_plot)
