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

st.set_page_config(page_title="Solar Forecast – ROBOTRONIX", layout="wide")

# ---------------- Auth ----------------
if "auth" not in st.session_state:
    st.session_state["auth"] = False
if not st.session_state["auth"]:
    st.title("🔐 Accesso richiesto")
    user = st.text_input("Username")
    pw = st.text_input("Password", type="password")
    if st.button("Login"):
        if user.strip().upper() == "FVMANAGER" and pw == "MIMMOFABIO":
            st.session_state["auth"] = True
            st.experimental_rerun()
        else:
            st.error("Credenziali non valide.")
    st.stop()

# ---------------- Config ----------------
DATA_PATH = "Dataset_Daily_EnergiaSeparata_2020_2025.csv"
MODEL_PATH = "pv_model.joblib"
LOG_PATH = "forecast_log.csv"
DEFAULT_LAT = 40.643278
DEFAULT_LON = 16.986083

MM_USER = "teseospa-eiffageenergiesystemesitaly_daniello_fabio"
MM_PASS = "6S8KTHPbrUlp6523T9Xd"

def ensure_log_file():
    if not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0:
        pd.DataFrame(columns=[
            "timestamp","day_label","provider","status","url",
            "lat","lon","tilt","orient","sum_rad_corr","pred_kwh","note"
        ]).to_csv(LOG_PATH, index=False)

def write_log(**row):
    ensure_log_file()
    try:
        df = pd.read_csv(LOG_PATH)
    except Exception:
        df = pd.DataFrame(columns=[
            "timestamp","day_label","provider","status","url",
            "lat","lon","tilt","orient","sum_rad_corr","pred_kwh","note"
        ])
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)

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

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_meteomatics_pt15m(lat, lon, start_iso, end_iso, tilt=None, orient=None):
    if tilt is not None and orient is not None and tilt > 0:
        rad_param = f"global_rad_tilt_{int(round(tilt))}_orientation_{int(round(orient))}:W"
    else:
        rad_param = "global_rad:W"
    url = (f"https://api.meteomatics.com/"
           f"{start_iso}--{end_iso}:PT15M/"
           f"{rad_param},total_cloud_cover:p/"
           f"{lat},{lon}/json")
    r = requests.get(url, auth=(MM_USER, MM_PASS), timeout=25)
    r.raise_for_status()
    j = r.json()
    rows = []
    for blk in j.get("data", []):
        prm = blk.get("parameter")
        if prm.endswith(":W"):
            prm = "GlobalRad_W"
        if prm == "total_cloud_cover:p":
            prm = "CloudCover_P"
        for d in blk["coordinates"][0]["dates"]:
            rows.append({"time": d["date"], prm: d["value"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return url, df
    df = df.groupby("time", as_index=False).max().sort_values("time")
    if "GlobalRad_W" not in df.columns: df["GlobalRad_W"] = np.nan
    if "CloudCover_P" not in df.columns: df["CloudCover_P"] = np.nan
    df["time"] = pd.to_datetime(df["time"])
    df["fonte"] = "Meteomatics"
    return url, df

def fetch_openmeteo_hourly(lat, lon, start_date, end_date):
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&hourly=direct_radiation,cloudcover&start_date={start_date}&end_date={end_date}")
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    j = r.json()
    hh = j.get("hourly", {})
    times = hh.get("time", [])
    rad   = hh.get("direct_radiation", [np.nan]*len(times))
    cld   = hh.get("cloudcover", [np.nan]*len(times))
    df = pd.DataFrame({"time": times, "GlobalRad_W": rad, "CloudCover_P": cld})
    df["time"] = pd.to_datetime(df["time"])
    df["fonte"] = "Open-Meteo"
    df = df.set_index("time").resample("15min").interpolate(method="time").reset_index()
    return url, df

def compute_curve_and_daily(df, model):
    if df is None or df.empty:
        return None, 0.0
    df = df.copy().sort_values("time")
    df["rad_corr"] = df["GlobalRad_W"].fillna(0) * (1 - df["CloudCover_P"].fillna(0)/100.0)
    sum_rad = df["rad_corr"].sum()
    pred_kwh = float(model.predict([[sum_rad]])[0]) if sum_rad > 0 else 0.0
    if sum_rad > 0:
        df["kWh_curve"] = pred_kwh * (df["rad_corr"]/sum_rad)
    else:
        df["kWh_curve"] = 0.0
    return df, pred_kwh

def forecast_for_day(lat, lon, offset_days, label, model, tilt, orient):
    day = (datetime.now(timezone.utc).date() + timedelta(days=offset_days))
    start_iso = f"{day}T00:00:00Z"; end_iso = f"{day + timedelta(days=1)}T00:00:00Z"
    provider = "Meteomatics"; status = "OK"; url = ""
    try:
        url, df = fetch_meteomatics_pt15m(lat, lon, start_iso, end_iso, tilt=tilt, orient=orient)
    except Exception as e:
        provider = "Meteomatics"; status = f"ERROR: {e}"; url = ""
        write_log(timestamp=datetime.utcnow().isoformat(), day_label=label,
                  provider=provider, status=status, url=url, lat=lat, lon=lon,
                  tilt=tilt, orient=orient, sum_rad_corr="", pred_kwh="", note="fallback Open‑Meteo")
        try:
            url, df = fetch_openmeteo_hourly(lat, lon, str(day), str(day + timedelta(days=1)))
            provider = "Open-Meteo"; status = "OK"
        except Exception as e2:
            provider = "Open-Meteo"; status = f"ERROR: {e2}"; df = None
    if df is None or df.empty:
        write_log(timestamp=datetime.utcnow().isoformat(), day_label=label,
                  provider=provider, status=status, url=url, lat=lat, lon=lon,
                  tilt=tilt, orient=orient, sum_rad_corr="", pred_kwh="", note="no data")
        return None, 0.0, provider, status, url
    df2, pred = compute_curve_and_daily(df, model)
    write_log(timestamp=datetime.utcnow().isoformat(), day_label=label,
              provider=provider, status=status, url=url, lat=lat, lon=lon,
              tilt=tilt, orient=orient, sum_rad_corr=float(df2["rad_corr"].sum()), pred_kwh=float(pred), note="")
    return df2, pred, provider, status, url

# ---------------- UI ----------------
st.title("☀️ Solar Forecast - ROBOTRONIX for IMEPOWER")

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

tab1, tab2, tab3 = st.tabs(["📊 Storico","🛠️ Modello","🔮 Previsioni 4 giorni (15 min)"])

with tab1:
    try:
        df = load_data()
        st.subheader("Storico produzione (kWh) e irradianza (W/m²)")
        if "E_INT_Daily_KWh" in df.columns:
            st.line_chart(df.set_index("Date")[["E_INT_Daily_KWh","G_M0_Wm2"]])
        else:
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
    st.subheader("Previsioni (Meteomatics PT15M con fallback Open‑Meteo)")
    c1, c2, c3, c4 = st.columns(4)
    lat = c1.number_input("Latitudine", value=DEFAULT_LAT, format="%.6f")
    lon = c2.number_input("Longitudine", value=DEFAULT_LON, format="%.6f")
    tilt = c3.slider("Tilt (°)", min_value=0, max_value=90, value=0, step=1)
    orient = c4.slider("Orientation (°, 0=N, 90=E, 180=S, 270=W)", min_value=0, max_value=360, value=180, step=5)
    st.caption("Se Tilt > 0 uso `global_rad_tilt_<tilt>_orientation_<orient>:W`, altrimenti `global_rad:W`.")
    go = st.button("Calcola previsioni (Ieri + Oggi + Domani + Dopodomani)")
    model = load_model()

    results = {}
    containers = {"Ieri": st.container(), "Oggi": st.container(), "Domani": st.container(), "Dopodomani": st.container()}

    if go:
        for label, off in [("Ieri",-1),("Oggi",0),("Domani",1),("Dopodomani",2)]:
            with containers[label]:
                st.markdown(f"### {label}")
                dfp, pred, provider, status, url = forecast_for_day(lat, lon, off, label, model, tilt, orient)
                results[label] = dfp
                st.caption(f"Provider: **{provider}** | Stato: **{status}**")
                if url:
                    st.code(url, language="text")
                if dfp is None or dfp.empty:
                    st.warning("Nessun dato disponibile.")
                else:
                    st.metric(f"Produzione prevista {label}", f"{pred:.1f} kWh")
                    chart_df = dfp.set_index("time")[["kWh_curve"]].rename(columns={"kWh_curve":"Produzione stimata (kWh/15min)"})
                    st.line_chart(chart_df)

        # Grafico comparativo
        st.subheader("📊 Confronto curve previste (4 giorni, 15 min)")
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
        else:
            st.info("Nessuna curva disponibile per il confronto.")
