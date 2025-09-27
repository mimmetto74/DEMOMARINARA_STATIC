
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
# Branding & Login
# ===============================
APP_TITLE = "☀️ Solar Forecast - ROBOTRONIX for IMEPOWER"
LOGIN_USER = "FVMANAGER"
LOGIN_PASS = "MIMMOFABIO"

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

# ===============================
# File paths
# ===============================
DATA_PATH = "Dataset_Daily_EnergiaSeparata_2020_2025.csv"
MODEL_PATH = "pv_model.joblib"

# Default coords Marinara (Taranto)
DEFAULT_LAT = 40.6432780
DEFAULT_LON = 16.9860830

# Meteomatics credentials (hardcoded for demo)
METEO_USER = "teseospa-eiffageenergiesystemesitaly_daniello_fabio"
METEO_PASS = "6S8KTHPbrUlp6523T9Xd"

# ===============================
# Data & Model
# ===============================
def load_full_dataset():
    if not os.path.exists(DATA_PATH):
        st.error(f"⚠️ Dataset non trovato: {DATA_PATH}")
        return None
    try:
        df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
        return df
    except Exception as e:
        st.error(f"Errore lettura dataset: {e}")
        return None

def train_full_model(df):
    # Usa SEMPRE tutto il CSV storico
    df = df.dropna(subset=["E_INT_Daily_kWh", "G_M0_Wm2"]).copy()
    if len(df) < 2:
        st.error("⚠️ Dataset troppo piccolo per addestrare il modello.")
        return None, None, None
    # Split cronologico per valutazione (opzionale)
    train = df[df["Date"] < "2025-01-01"]
    test  = df[df["Date"] >= "2025-01-01"]
    if len(train) < 2:
        train = df.copy()
        test = df.iloc[-min(30, len(df)):]

    X_train, y_train = train[["G_M0_Wm2"]], train["E_INT_Daily_kWh"]
    X_test,  y_test  = test[["G_M0_Wm2"]],  test["E_INT_Daily_kWh"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    try:
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred) if len(y_test)>0 else float("nan")
        r2  = r2_score(y_test, y_pred) if len(y_test)>1 else float("nan")
    except Exception:
        mae, r2 = float("nan"), float("nan")

    joblib.dump(model, MODEL_PATH)
    return model, mae, r2

def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None

# ===============================
# Forecast providers
# ===============================
def get_meteomatics_15min(lat, lon, target_date: date):
    """
    Ritorna DataFrame 15-min con colonne: solar_rad (W/m2), cloud_cover (%)
    """
    base_url = "https://api.meteomatics.com"
    day_iso = target_date.strftime("%Y-%m-%d")
    start = f"{day_iso}T00:00:00Z"
    end   = f"{day_iso}T23:45:00Z"
    params = "solar_rad:W,cloud_cover:pc"
    interval = "PT15M"
    url = f"{base_url}/{start}--{end}:{interval}/{params}/{lat},{lon}/json"
    r = requests.get(url, auth=HTTPBasicAuth(METEO_USER, METEO_PASS), timeout=25)
    r.raise_for_status()
    data = r.json()
    # Parse
    times = [t["validdate"] for t in data["data"][0]["coordinates"][0]["dates"]]
    rad   = [t["value"] for t in data["data"][0]["coordinates"][0]["dates"]]
    clouds= [t["value"] for t in data["data"][1]["coordinates"][0]["dates"]]
    df = pd.DataFrame({
        "time": pd.to_datetime(times),
        "solar_rad": rad,
        "cloud_cover": clouds
    }).set_index("time")
    return df

def get_openmeteo_15min(lat, lon, target_date: date):
    # Fallback: genera serie 15-min interpolando da orario shortwave_radiation
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=shortwave_radiation&timezone=auto"
        f"&start_date={target_date.isoformat()}&end_date={target_date.isoformat()}"
    )
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    data = r.json()
    irr_values = data["hourly"]["shortwave_radiation"]
    hours = pd.date_range(start=str(target_date) + " 00:00", periods=len(irr_values), freq="H")
    df_irr = pd.DataFrame({"Ora": hours, "Irraggiamento": irr_values}).set_index("Ora")
    df_irr_15 = df_irr.resample("15T").interpolate()
    # Cloud cover non disponibile → assumiamo 0 per evitare correzione
    df_irr_15 = df_irr_15.rename(columns={"Irraggiamento":"solar_rad"})
    df_irr_15["cloud_cover"] = 0.0
    return df_irr_15

# ===============================
# Prediction logic
# ===============================
def predict_daily_energy_from_curve(model, df_curve_15min, alpha_cloud=0.5):
    """
    - Usa la radiazione 15-min per costruire una curva di produzione proporzionale
    - Corregge il totale con la media di cloud_cover (0..100) via fattore (1 - alpha*cc/100)
    - Restituisce: daily_kWh, series_kWh_15min
    """
    rad = df_curve_15min["solar_rad"].clip(lower=0)
    total_rad = rad.sum()
    if total_rad <= 0:
        return 0.0, pd.Series(0.0, index=rad.index)

    # Stima produzione giornaliera dal modello usando la media (come proxy del feature daily)
    rad_mean = rad.mean()
    daily_kwh = float(model.predict(pd.DataFrame({"G_M0_Wm2":[rad_mean]}))[0])

    # Correzione per nuvolosità media
    cc_mean = float(df_curve_15min["cloud_cover"].mean()) if "cloud_cover" in df_curve_15min else 0.0
    cloud_factor = max(0.0, 1.0 - alpha_cloud * (cc_mean/100.0))
    daily_kwh_adj = daily_kwh * cloud_factor

    # Distribuzione 15-min proporzionale alla radiazione
    kwh_curve = (rad / total_rad) * daily_kwh_adj
    return daily_kwh_adj, kwh_curve

# ===============================
# UI
# ===============================
st.set_page_config(page_title="PV Forecast Dashboard", layout="wide")
st.markdown(f"<h1 style='text-align:center;color:orange'>{APP_TITLE}</h1>", unsafe_allow_html=True)
st.write("---")

# Load + train once per session
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

df_hist = load_full_dataset()
if df_hist is None:
    st.stop()

# Train on full CSV
model, mae, r2 = train_full_model(df_hist)
st.session_state.model_trained = model is not None

# Show training metrics
if model is not None:
    c1, c2 = st.columns(2)
    c1.success(f"✅ Modello addestrato sul CSV • MAE={mae if mae==mae else 'n/a':.2f} kWh")
    c2.info(f"R²={r2 if r2==r2 else 'n/a'}")

# Plot storico base
st.header("📊 Storico (CSV)")
df_hist = df_hist.rename(columns={"E_INT_kWh":"E_INT_Daily_kWh","E_Z_EVU_kWh":"E_Z_EVU_Daily_kWh"})
st.line_chart(df_hist.set_index("Date")[["E_INT_Daily_kWh","G_M0_Wm2"]])

# Forecast domani + dopodomani
st.header("🔮 Previsioni Giorno+1 e Giorno+2")
lat = st.number_input("Latitudine", value=DEFAULT_LAT, format="%.6f")
lon = st.number_input("Longitudine", value=DEFAULT_LON, format="%.6f")

if st.button("Calcola previsioni (Domani + Dopodomani)"):
    if not st.session_state.model_trained:
        st.error("Devi addestrare il modello prima (errore interno).")
        st.stop()
    d1 = date.today() + timedelta(days=1)
    d2 = date.today() + timedelta(days=2)

    used_provider = "meteomatics"
    try:
        df15_d1 = get_meteomatics_15min(lat, lon, d1)
        df15_d2 = get_meteomatics_15min(lat, lon, d2)
        st.success("✅ Previsione basata su Meteomatics (solar_rad + cloud_cover)")
    except Exception as e:
        used_provider = "openmeteo"
        st.warning("⚠️ Meteomatics non disponibile, uso Open-Meteo")
        df15_d1 = get_openmeteo_15min(lat, lon, d1)
        df15_d2 = get_openmeteo_15min(lat, lon, d2)

    # Predict both days
    d1_kwh, d1_curve = predict_daily_energy_from_curve(model, df15_d1, alpha_cloud=0.5)
    d2_kwh, d2_curve = predict_daily_energy_from_curve(model, df15_d2, alpha_cloud=0.5)

    # Metrics
    m1, m2 = st.columns(2)
    m1.metric(f"Produzione prevista {d1.isoformat()}", f"{d1_kwh:.1f} kWh")
    m2.metric(f"Produzione prevista {d2.isoformat()}", f"{d2_kwh:.1f} kWh")

    # Plot curves
    st.subheader("📈 Curva 15-min prevista")
    df_plot = pd.DataFrame({
        f"{d1.isoformat()} kWh/15min": d1_curve,
        f"{d2.isoformat()} kWh/15min": d2_curve
    })
    st.line_chart(df_plot)

    # Download risultati
    out = pd.DataFrame({
        "Datetime": list(d1_curve.index) + list(d2_curve.index),
        "kWh_15min": list(d1_curve.values) + list(d2_curve.values),
        "Day": [d1.isoformat()]*len(d1_curve) + [d2.isoformat()]*len(d2_curve)
    })
    st.download_button("⬇️ Scarica curva 15-min (CSV)",
                       out.to_csv(index=False).encode("utf-8"),
                       file_name="forecast_15min.csv",
                       mime="text/csv")
