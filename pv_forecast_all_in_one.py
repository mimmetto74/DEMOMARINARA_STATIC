
import os
import io
import json
import requests
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------
# Config & constants
# -----------------------
DATA_PATH = "Dataset_Daily_EnergiaSeparata_2020_2025.csv"
MODEL_PATH = "pv_model.joblib"
LOG_PATH = "logs_forecast.csv"

DEFAULT_LAT = 40.643278
DEFAULT_LON = 16.986083

MM_USER = os.getenv("MM_USER", "teseospa-eiffageenergiesystemesitaly_daniello_fabio")
MM_PASS = os.getenv("MM_PASS", "6S8KTHPbrUlp6523T9Xd")

# -----------------------
# Helpers
# -----------------------
def ensure_log_file():
    if not os.path.exists(LOG_PATH):
        pd.DataFrame(columns=[
            "timestamp","day_label","provider","status","url","lat","lon",
            "param_direct","param_cloud","daily_rad_corr","pred_kwh","note"
        ]).to_csv(LOG_PATH, index=False)

def write_log(row: dict):
    ensure_log_file()
    df = pd.read_csv(LOG_PATH)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)

@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["Date"])

def train_model():
    df = load_data()
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
    return float(mae), float(r2)

def load_model():
    if not os.path.exists(MODEL_PATH):
        train_model()
    return joblib.load(MODEL_PATH)

# -----------------------
# Providers
# -----------------------
def fetch_meteomatics(lat, lon, start_iso, end_iso):
    url = f"https://api.meteomatics.com/{start_iso}--{end_iso}:PT1H/direct_rad:W,total_cloud_cover:p/{lat},{lon}/json"
    r = requests.get(url, auth=(MM_USER, MM_PASS), timeout=20)
    r.raise_for_status()
    j = r.json()
    records = []
    for param_block in j["data"]:
        param = param_block["parameter"]
        for d in param_block["coordinates"][0]["dates"]:
            records.append({"time": d["date"], param: d["value"]})
    df = pd.DataFrame(records).drop_duplicates(subset=["time"]).sort_values("time")
    # ensure both columns exist
    if "direct_rad:W" not in df.columns:
        df["direct_rad:W"] = np.nan
    if "total_cloud_cover:p" not in df.columns:
        df["total_cloud_cover:p"] = np.nan
    df["fonte"] = "Meteomatics"
    return url, df

def fetch_openmeteo(lat, lon, start_date, end_date):
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&hourly=direct_radiation,cloudcover&start_date={start_date}&end_date={end_date}"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    j = r.json()
    hours = j.get("hourly", {})
    times = hours.get("time", [])
    rad = hours.get("direct_radiation", [np.nan]*len(times))
    cloud = hours.get("cloudcover", [np.nan]*len(times))
    df = pd.DataFrame({"time": times, "direct_rad:W": rad, "total_cloud_cover:p": cloud})
    df["fonte"] = "Open-Meteo"
    return url, df

def compute_prediction(model, df):
    if df is None or df.empty:
        return None, None
    # Normalize time format
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")
    # Cloud correction
    df["rad_corr"] = df["direct_rad:W"].fillna(0) * (1 - df["total_cloud_cover:p"].fillna(0) / 100.0)
    daily_rad = df["rad_corr"].sum()
    pred_kwh = float(model.predict([[daily_rad]])[0])
    return df, pred_kwh

def forecast_day(model, lat, lon, offset_days, label):
    # date window
    day = (datetime.now(timezone.utc).date() + timedelta(days=offset_days))
    start_iso = f"{day}T00:00:00Z"
    end_iso   = f"{day + timedelta(days=1)}T00:00:00Z"
    # try meteomatics
    try:
        url, df = fetch_meteomatics(lat, lon, start_iso, end_iso)
        provider = "Meteomatics"
        status = "OK"
    except Exception as e:
        provider = "Meteomatics"
        status = f"ERROR: {e}"
        write_log({
            "timestamp": datetime.utcnow().isoformat(),
            "day_label": label, "provider": provider, "status": status,
            "url": f"{start_iso}->{end_iso}", "lat": lat, "lon": lon,
            "param_direct": "direct_rad:W", "param_cloud": "total_cloud_cover:p",
            "daily_rad_corr": "", "pred_kwh": "", "note": "fallback to Open-Meteo"
        })
        # fallback open-meteo
        try:
            url, df = fetch_openmeteo(lat, lon, str(day), str(day + timedelta(days=1)))
            provider = "Open-Meteo"
            status = "OK"
        except Exception as e2:
            provider = "Open-Meteo"
            status = f"ERROR: {e2}"
            write_log({
                "timestamp": datetime.utcnow().isoformat(),
                "day_label": label, "provider": provider, "status": status,
                "url": "", "lat": lat, "lon": lon,
                "param_direct": "direct_radiation", "param_cloud": "cloudcover",
                "daily_rad_corr": "", "pred_kwh": "", "note": "both providers failed"
            })
            return None, None, provider, status, ""

    df, pred = compute_prediction(model, df)
    # log success
    write_log({
        "timestamp": datetime.utcnow().isoformat(),
        "day_label": label, "provider": provider, "status": status,
        "url": url, "lat": lat, "lon": lon,
        "param_direct": "direct_rad:W" if provider=="Meteomatics" else "direct_radiation",
        "param_cloud": "total_cloud_cover:p" if provider=="Meteomatics" else "cloudcover",
        "daily_rad_corr": float(df["rad_corr"].sum()) if df is not None else "",
        "pred_kwh": pred if pred is not None else "",
        "note": ""
    })
    return df, pred, provider, status, url

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="Solar Forecast – ROBOTRONIX", layout="wide")
st.title("☀️ Solar Forecast – ROBOTRONIX for IMEPOWER")

# Sidebar: log filters and download
st.sidebar.header("📥 Log previsioni")
log_filter = st.sidebar.selectbox("Tipo di log", ["Tutti","Solo Meteomatics","Solo Open‑Meteo","Solo Errori"])
ensure_log_file()
log_df = pd.read_csv(LOG_PATH)
filt = log_df.copy()
if log_filter=="Solo Meteomatics":
    filt = filt[filt["provider"]=="Meteomatics"]
elif log_filter=="Solo Open‑Meteo":
    filt = filt[filt["provider"]=="Open-Meteo"]
elif log_filter=="Solo Errori":
    filt = filt[filt["status"].str.startswith("ERROR", na=False)]
st.sidebar.write(f"Righe filtrate: {len(filt)}")
csv_buf = io.StringIO()
filt.to_csv(csv_buf, index=False)
st.sidebar.download_button("⬇️ Scarica CSV filtrato", data=csv_buf.getvalue(), file_name="logs_filtered.csv", mime="text/csv")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Analisi Storica","🛠️ Addestramento","🔮 Previsioni"])

with tab1:
    df = load_data()
    st.subheader("Storico produzione e irraggiamento")
    st.line_chart(df.set_index("Date")[["E_INT_Daily_kWh","G_M0_Wm2"]])

with tab2:
    if st.button("Addestra/Riaddestra modello"):
        mae, r2 = train_model()
        st.success(f"Modello addestrato ✅  MAE: {mae:.2f}  |  R²: {r2:.3f}")
    if os.path.exists(MODEL_PATH):
        st.info("Modello presente su disco: **OK**")

with tab3:
    st.subheader("Previsioni Ieri + Oggi + Domani + Dopodomani (Meteomatics con fallback Open‑Meteo)")
    lat = st.number_input("Latitudine", value=DEFAULT_LAT, format="%.6f")
    lon = st.number_input("Longitudine", value=DEFAULT_LON, format="%.6f")
    go = st.button("Calcola previsioni")
    model = load_model()

    placeholders = {
        "Ieri": st.container(),
        "Oggi": st.container(),
        "Domani": st.container(),
        "Dopodomani": st.container(),
    }

    if go:
        for label, offset in [("Ieri",-1),("Oggi",0),("Domani",1),("Dopodomani",2)]:
            with placeholders[label]:
                st.markdown(f"### {label}")
                df_pred, pred_kwh, provider, status, url = forecast_day(model, lat, lon, offset, label)
                st.caption(f"Provider: **{provider}** | Stato: **{status}**")
                if url:
                    st.code(url, language="text")
                if df_pred is not None and pred_kwh is not None:
                    st.metric(f"Produzione prevista {label}", f"{pred_kwh:.1f} kWh")
                    # time index and chart
                    chart_df = df_pred.set_index("time")[["rad_corr"]].rename(columns={"rad_corr":"Radiazione corretta (W)"})
                    st.line_chart(chart_df)
                else:
                    st.warning("Nessun dato disponibile per questo giorno.")
