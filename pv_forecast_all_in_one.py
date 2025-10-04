# -*- coding: utf-8 -*-
import os, base64, math, json, requests
import numpy as np, pandas as pd
import altair as alt, folium, streamlit as st
from streamlit_folium import st_folium
from datetime import datetime, timedelta, timezone
from sklearn.linear_model import LinearRegression

APP_TITLE = "Solar Forecast - ROBOTRONIX for IMEPOWER (V7)"
DATASET_DEFAULT_PATH = "Dataset_Daily_EnergiaSeparata_2020_2025.csv"
DATASET_FALLBACK_ABS = "/mnt/data/Dataset_Daily_EnergiaSeparata_2020_2025.csv"
MODEL_FEATS = ["G_M0_Wm2"]
TARGET_COL = "E_INT_Daily_kWh"
DEFAULT_LAT, DEFAULT_LON = 40.643278, 16.986083
VALID_USER, VALID_PASS = "FVMANAGER", "MIMMOFABIO"

st.set_page_config(page_title=APP_TITLE, page_icon="☀️", layout="wide")

def bytes_download_link(data: bytes, filename: str, label: str):
    import base64
    b64 = base64.b64encode(data).decode()
    st.markdown(f'<a href="data:file/octet-stream;base64,{b64}" download="{filename}">{label}</a>', unsafe_allow_html=True)

def load_historical_csv():
    for p in [DATASET_DEFAULT_PATH, DATASET_FALLBACK_ABS]:
        if os.path.exists(p):
            return pd.read_csv(p, parse_dates=["Date"]).sort_values("Date")
    st.error("Dataset storico non trovato.")
    return None

def train_linear_model(df):
    df2 = df.dropna(subset=MODEL_FEATS + [TARGET_COL]).copy()
    if df2.empty: return None, None
    X = df2[MODEL_FEATS].values.reshape(-1, len(MODEL_FEATS))
    y = df2[TARGET_COL].values
    m = LinearRegression().fit(X, y)
    yhat = m.predict(X)
    return m, {"mae": float(np.mean(np.abs(y-yhat))), "r2": float(m.score(X,y)), "n": int(len(df2))}

def open_meteo_fetch(lat, lon, start_dt, days=4):
    start = start_dt.strftime("%Y-%m-%d")
    end = (start_dt + timedelta(days=days)).strftime("%Y-%m-%d")
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=shortwave_radiation&timezone=UTC&start_date={start}&end_date={end}"
    r = requests.get(url, timeout=30); r.raise_for_status()
    js = r.json(); times = pd.to_datetime(js["hourly"]["time"])
    vals = js["hourly"].get("shortwave_radiation", [0]*len(times))
    df = pd.DataFrame({"timestamp": times, "global_rad_Wm2": vals}).set_index("timestamp")
    return df.resample("15min").interpolate()

def meteomatics_fetch(lat, lon, start_dt, days=4, step="PT15M"):
    user, pwd = os.environ.get("METEO_USER"), os.environ.get("METEO_PASS")
    if not user or not pwd: raise RuntimeError("METEO_USER / METEO_PASS mancanti")
    start = start_dt.strftime("%Y-%m-%dT00:00:00Z")
    end = (start_dt + timedelta(days=days)).strftime("%Y-%m-%dT00:00:00Z")
    url = f"https://api.meteomatics.com/{start}--{end}:{step}/global_rad:W/{lat},{lon}/json"
    r = requests.get(url, auth=(user,pwd), timeout=30); r.raise_for_status()
    js = r.json(); series=[]
    vals = js["data"][0]["coordinates"][0]["dates"]
    for it in vals: series.append((pd.to_datetime(it["date"]), float(it["value"])))
    return pd.DataFrame(series, columns=["timestamp","global_rad_Wm2"]).set_index("timestamp")

def split_days(df15, base_dt):
    out={}
    labels=[("Ieri",-1),("Oggi",0),("Domani",1),("Dopodomani",2)]
    for lab,off in labels:
        day = (base_dt + timedelta(days=off)).date()
        d = df15[df15.index.date==day].copy()
        out[f"{lab} ({day})"]=d
    return out

def curve_energy(df15, minutes=15):
    if df15.empty: return df15.assign(energy_kWh_m2=0.0), 0.0
    kwh = (df15["global_rad_Wm2"]*(minutes/60.0))/1000.0
    out = df15.copy(); out["energy_kWh_m2"]=kwh
    return out, float(kwh.sum())

def make_map(lat,lon):
    m = folium.Map(location=[lat,lon], tiles="Esri.WorldImagery", zoom_start=17, control_scale=True)
    html = f"""<div style='background:#fff;padding:10px;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,.25);'>
    <h4 style='margin:0 0 6px 0;'>Impianto FV – Marinara</h4>
    <div style='font-size:12px;'>Lat: {lat:.6f}<br/>Lon: {lon:.6f}<br/>Satellite: Esri World Imagery</div>
    </div>"""
    folium.Marker([lat,lon], tooltip="Impianto FV", popup=folium.Popup(html, max_width=260)).add_to(m)
    return m

# ---------------- UI ----------------
if "auth_ok" not in st.session_state: st.session_state["auth_ok"]=False
with st.sidebar:
    st.subheader("🔐 Login")
    u=st.text_input("Username"); p=st.text_input("Password", type="password")
    if st.button("Accedi"):
        if u.strip().upper()=="FVMANAGER" and p=="MIMMOFABIO":
            st.session_state["auth_ok"]=True; st.success("Accesso effettuato.")
        else: st.error("Credenziali non valide.")

if not st.session_state["auth_ok"]:
    st.title(APP_TITLE); st.stop()

with st.sidebar:
    st.markdown("## ⚙️ Menu delle impostazioni")
    lat = st.number_input("Latitudine", value=float(DEFAULT_LAT), step=0.0001)
    lon = st.number_input("Longitudine", value=float(DEFAULT_LON), step=0.0001)
    src = st.radio("Sorgente previsioni", ["Meteomatics (preferita)", "Open‑Meteo (fallback)"], index=1)
    st.caption("Imposta METEO_USER/METEO_PASS per usare Meteomatics.")
    st.markdown("---")

st.title(APP_TITLE)

# Storico + Modello
st.header("📊 Storico e Modello")
df = load_historical_csv()
if df is not None:
    st.dataframe(df.tail(200), use_container_width=True)
    if st.button("Addestra modello con dati storici"):
        m, met = train_linear_model(df)
        if m is not None:
            st.session_state["model"]=m
            st.success(f"Modello addestrato ✅ MAE={met['mae']:.1f} kWh • R²={met['r2']:.3f} • N={met['n']}")
        else:
            st.error("Addestramento fallito.")

# Previsioni 15m
st.header("🔮 Previsioni 4 giorni (15 min)")
base = datetime.utcnow().replace(hour=0,minute=0,second=0,microsecond=0)
df15=None
try:
    if src.startswith("Meteomatics"):
        try:
            df15 = meteomatics_fetch(lat,lon, base-timedelta(days=1), days=4, step="PT15M")
            st.success("Dati Meteomatics ricevuti.")
        except Exception as e:
            st.warning(f"Meteomatics non disponibile: {e}. Fallback Open‑Meteo.")
            df15 = open_meteo_fetch(lat,lon, base-timedelta(days=1), days=4)
    else:
        df15 = open_meteo_fetch(lat,lon, base-timedelta(days=1), days=4)
except Exception as e:
    st.error(f"Errore fetch previsioni: {e}")

if df15 is not None and not df15.empty:
    days = split_days(df15, base)
    energy = {}
    for lab, d in days.items():
        colA,colB = st.columns([2,1])
        with colA:
            ch = alt.Chart(d.reset_index()).mark_line().encode(
                x=alt.X("timestamp:T", title="Ora"),
                y=alt.Y("global_rad_Wm2:Q", title="Radiazione (W/m²)"),
                tooltip=["timestamp:T","global_rad_Wm2:Q"]
            ).properties(height=260, title=f"Radiazione – {lab}")
            st.altair_chart(ch, use_container_width=True)
        with colB:
            enr, tot = curve_energy(d, minutes=15); energy[lab]=tot
            st.metric("Energia [kWh/m²]", f"{tot:.2f}")
            if not d.empty:
                st.metric("Picco irradianza [%]", f"{(float(d['global_rad_Wm2'].max())/1000*100):.0f}%")
        if not d.empty:
            csv = d.to_csv().encode("utf-8")
            bytes_download_link(csv, f"curva_15m_{lab.replace(' ','_')}.csv", "⬇️ Scarica CSV 15m")

    dtab = pd.DataFrame([{"Giorno":k, "Energia_kWh_m2":v} for k,v in energy.items()])
    st.dataframe(dtab, use_container_width=True)
    bytes_download_link(dtab.to_csv(index=False).encode("utf-8"), "aggregato_giornaliero.csv", "⬇️ Scarica aggregato")

# Mappa
st.header("🗺️ Mappa impianto (satellitare)")
mp = make_map(lat,lon); st_folium(mp, height=520)
