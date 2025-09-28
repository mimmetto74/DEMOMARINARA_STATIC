import os
import io
import requests
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# =========================
# Login semplice
# =========================
st.set_page_config(page_title="Solar Forecast – ROBOTRONIX", layout="wide")

if "auth" not in st.session_state:
    st.session_state["auth"] = False

if not st.session_state["auth"]:
    st.title("🔐 Accesso richiesto")
    user = st.text_input("Username")
    pw = st.text_input("Password", type="password")
    if st.button("Login"):
        if user == "FVMANAGER" and pw == "MIMMOFABIO":
            st.session_state["auth"] = True
            st.experimental_rerun()
        else:
            st.error("Credenziali non valide.")
    st.stop()

# =========================
# Config & Paths
# =========================
DATA_PATH = "Dataset_Daily_EnergiaSeparata_2020_2025.csv"
MODEL_PATH = "pv_model.joblib"
LOG_PATH = "forecast_log.csv"

DEFAULT_LAT = 40.643278
DEFAULT_LON = 16.986083

# Meteomatics creds
MM_USER = "teseospa-eiffageenergiesystemesitaly_daniello_fabio"
MM_PASS = "6S8KTHPbrUlp6523T9Xd"

# =========================
# Utils for log
# =========================
def ensure_log_file():
    if not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0:
        pd.DataFrame(columns=[
            "timestamp","day_label","provider","status","url",
            "lat","lon","sum_rad_corr","pred_kwh","note"
        ]).to_csv(LOG_PATH, index=False)

def write_log(**row):
    ensure_log_file()
    try:
        df = pd.read_csv(LOG_PATH)
    except Exception:
        df = pd.DataFrame(columns=[
            "timestamp","day_label","provider","status","url",
            "lat","lon","sum_rad_corr","pred_kwh","note"
        ])
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)

# =========================
# Data & Model
# =========================
@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["Date"])

def train_model():
    df = load_data().dropna(subset=["E_INT_Daily_kWh","G_M0_Wm2"])
    if df.empty:
        return float("nan"), float("nan")
    train = df[df["Date"] < "2025-01-01"]
    test  = df[df["Date"] >= "2025-01-01"]
    X_train, y_train = train[["G_M0_Wm2"]], train["E_INT_Daily_kWh"]
    model = LinearRegression().fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    if len(test) > 0:
        y_pred = model.predict(test[["G_M0_Wm2"]])
        mae = float(mean_absolute_error(test["E_INT_Daily_kWh"], y_pred))
        r2  = float(r2_score(test["E_INT_Daily_kWh"], y_pred))
    else:
        mae = float("nan"); r2 = float("nan")
    return mae, r2

def load_model():
    if not os.path.exists(MODEL_PATH):
        train_model()
    return joblib.load(MODEL_PATH)

# =========================
# Providers
# =========================
@st.cache_data(ttl=3600)
def fetch_meteomatics(lat, lon, start_iso, end_iso):
    url = f"https://api.meteomatics.com/{start_iso}--{end_iso}:PT1H/direct_rad:W,total_cloud_cover:p/{lat},{lon}/json"
    r = requests.get(url, auth=(MM_USER, MM_PASS), timeout=25)
    r.raise_for_status()
    j = r.json()
    frames = []
    for blk in j.get("data", []):
        prm = blk.get("parameter")
        if prm == "direct_rad:W": prm = "DirectRad_W"
        if prm == "total_cloud_cover:p": prm = "CloudCover_P"
        for d in blk["coordinates"][0]["dates"]:
            frames.append({"time": d["date"], prm: d["value"]})
    df = pd.DataFrame(frames)
    if df.empty:
        return url, df
    df = df.groupby("time", as_index=False).max().sort_values("time")
    if "DirectRad_W" not in df.columns: df["DirectRad_W"] = np.nan
    if "CloudCover_P" not in df.columns: df["CloudCover_P"] = np.nan
    df["fonte"] = "Meteomatics"
    return url, df

def fetch_openmeteo(lat, lon, start_date, end_date):
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&hourly=direct_radiation,cloudcover&start_date={start_date}&end_date={end_date}")
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    j = r.json()
    hh = j.get("hourly", {})
    times = hh.get("time", [])
    rad   = hh.get("direct_radiation", [np.nan]*len(times))
    cld   = hh.get("cloudcover", [np.nan]*len(times))
    df = pd.DataFrame({"time": times, "DirectRad_W": rad, "CloudCover_P": cld})
    df["fonte"] = "Open-Meteo"
    return url, df

# =========================
# Forecast logic
# =========================
def compute_curve_and_daily(df, model):
    if df is None or df.empty:
        return None, 0.0
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")
    df["rad_corr"] = df["DirectRad_W"].fillna(0) * (1 - df["CloudCover_P"].fillna(0)/100.0)
    sum_rad = df["rad_corr"].sum()
    pred_kwh = float(model.predict([[sum_rad]])[0]) if sum_rad > 0 else 0.0
    if sum_rad > 0:
        df["kWh_curve"] = pred_kwh * (df["rad_corr"]/sum_rad)
    else:
        df["kWh_curve"] = 0.0
    return df, pred_kwh

def forecast_for_day(lat, lon, offset_days, label, model):
    day = (datetime.now(timezone.utc).date() + timedelta(days=offset_days))
    start_iso = f"{day}T00:00:00Z"; end_iso = f"{day + timedelta(days=1)}T00:00:00Z"
    provider = "Meteomatics"; status = "OK"; url = ""
    try:
        url, df = fetch_meteomatics(lat, lon, start_iso, end_iso)
    except Exception as e:
        provider = "Meteomatics"; status = f"ERROR: {e}"; url = ""
        write_log(timestamp=datetime.utcnow().isoformat(), day_label=label,
                  provider=provider, status=status, url=url, lat=lat, lon=lon,
                  sum_rad_corr="", pred_kwh="", note="fallback Open‑Meteo")
        try:
            url, df = fetch_openmeteo(lat, lon, str(day), str(day + timedelta(days=1)))
            provider = "Open-Meteo"; status = "OK"
        except Exception as e2:
            provider = "Open-Meteo"; status = f"ERROR: {e2}"; df = None
    if df is None or df.empty:
        write_log(timestamp=datetime.utcnow().isoformat(), day_label=label,
                  provider=provider, status=status, url=url, lat=lat, lon=lon,
                  sum_rad_corr="", pred_kwh="", note="no data")
        return None, 0.0, provider, status, url
    df2, pred = compute_curve_and_daily(df, model)
    write_log(timestamp=datetime.utcnow().isoformat(), day_label=label,
              provider=provider, status=status, url=url, lat=lat, lon=lon,
              sum_rad_corr=float(df2["rad_corr"].sum()), pred_kwh=float(pred), note="")
    return df2, pred, provider, status, url

# =========================
# UI
# =========================
st.title("☀️ Solar Forecast - ROBOTRONIX for IMEPOWER")

# Sidebar: Log download/filter
ensure_log_file()
st.sidebar.header("📥 Log Previsioni")
log_df = pd.read_csv(LOG_PATH)
flt = st.sidebar.selectbox("Filtro log", ["Tutti","Solo Meteomatics","Solo Open‑Meteo","Solo Errori"])
ldf = log_df.copy()
if flt=="Solo Meteomatics":
    ldf = ldf[ldf["provider"]=="Meteomatics"]
elif flt=="Solo Open‑Meteo":
    ldf = ldf[ldf["provider"]=="Open-Meteo"]
elif flt=="Solo Errori":
    ldf = ldf[ldf["status"].astype(str).str.startswith("ERROR", na=False)]
st.sidebar.write(f"Righe: {len(ldf)}")
csv_io = io.StringIO(); ldf.to_csv(csv_io, index=False)
st.sidebar.download_button("⬇️ Scarica log filtrato", csv_io.getvalue(), "forecast_log_filtered.csv", "text/csv")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Storico","🛠️ Modello","🔮 Previsioni 4 giorni"])

with tab1:
    try:
        df = load_data()
        st.subheader("Storico produzione (kWh) e irradianza (W/m²)")
        st.line_chart(df.set_index("Date")[["E_INT_Daily_kWh","G_M0_Wm2"]])
    except Exception as e:
        st.error(f"Impossibile caricare il dataset: {e}")

with tab2:
    colA, colB = st.columns(2)
    if colA.button("Addestra / Riaddestra modello"):
        mae, r2 = train_model()
        st.success(f"Modello addestrato ✅  MAE: {mae:.2f} | R²: {r2:.3f}")
    if os.path.exists(MODEL_PATH):
        colB.info("Modello disponibile su disco ✅")

with tab3:
    st.subheader("Previsioni (Meteomatics con fallback Open‑Meteo)")
    c1, c2 = st.columns(2)
    lat = c1.number_input("Latitudine", value=DEFAULT_LAT, format="%.6f")
    lon = c2.number_input("Longitudine", value=DEFAULT_LON, format="%.6f")
    go = st.button("Calcola previsioni (Ieri + Oggi + Domani + Dopodomani)")
    model = load_model()

    results = {}
    containers = {
        "Ieri": st.container(),
        "Oggi": st.container(),
        "Domani": st.container(),
        "Dopodomani": st.container(),
    }

    if go:
        for label, off in [("Ieri",-1),("Oggi",0),("Domani",1),("Dopodomani",2)]:
            with containers[label]:
                st.markdown(f"### {label}")
                dfp, pred, provider, status, url = forecast_for_day(lat, lon, off, label, model)
                results[label] = dfp
                st.caption(f"Provider: **{provider}** | Stato: **{status}**")
                if url:
                    st.code(url, language="text")
                if dfp is None or dfp.empty:
                    st.warning("Nessun dato disponibile.")
                else:
                    st.metric(f"Produzione prevista {label}", f"{pred:.1f} kWh")
                    chart_df = dfp.set_index("time")[["kWh_curve"]].rename(columns={"kWh_curve":"Produzione stimata (kWh/h)"})
                    st.line_chart(chart_df)

        # Grafico comparativo
        st.subheader("📊 Confronto curve previste (4 giorni)")
        comp = pd.DataFrame()
        for lbl, dfp in results.items():
            if dfp is not None and not dfp.empty:
                tmp = dfp[["time","kWh_curve"]].copy()
                tmp = tmp.rename(columns={"kWh_curve": lbl})
                if comp.empty:
                    comp = tmp
                else:
                    comp = pd.merge(comp, tmp, on="time", how="outer")
        if not comp.empty:
            comp = comp.set_index("time")
            st.line_chart(comp)
