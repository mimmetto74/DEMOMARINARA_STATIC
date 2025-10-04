# -*- coding: utf-8 -*-
import os, io, base64, requests
import numpy as np, pandas as pd, altair as alt
import streamlit as st
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta, timezone
from sklearn.linear_model import LinearRegression

APP_TITLE = "Solar Forecast - PROVA (base V4 con Modello)"
DATASET_PATHS = ["Dataset_Daily_EnergiaSeparata_2020_2025.csv", "/mnt/data/Dataset_Daily_EnergiaSeparata_2020_2025.csv"]
TARGET_COL = "E_INT_Daily_kWh"
FEAT_COL  = "G_M0_Wm2"

DEFAULT_LAT, DEFAULT_LON = 40.643278, 16.986083

st.set_page_config(page_title=APP_TITLE, page_icon="☀️", layout="wide")

# -------------------- Auth --------------------
if "auth" not in st.session_state:
    st.session_state.auth = False

def login_box():
    st.title(APP_TITLE)
    st.subheader("🔐 Accesso richiesto")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Entra"):
        if u.strip().upper() == "FVMANAGER" and p == "MIMMOFABIO":
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Credenziali non valide")

if not st.session_state.auth:
    login_box()
    st.stop()

# -------------------- Sidebar --------------------
st.sidebar.title("📋 Menu delle Impostazioni")
provider = st.sidebar.selectbox("Provider meteo", ["Auto", "Meteomatics", "Open‑Meteo"])
lat = st.sidebar.number_input("Latitudine", value=DEFAULT_LAT, format="%.6f")
lon = st.sidebar.number_input("Longitudine", value=DEFAULT_LON, format="%.6f")
tilt = st.sidebar.slider("Tilt (°)", 0, 90, 0)
orient = st.sidebar.slider("Orientamento (°, 0=N, 90=E, 180=S, 270=W)", 0, 360, 180, 5)
plant_kw = st.sidebar.number_input("Potenza di targa (kW)", value=1000.0, step=50.0, min_value=0.0)
autosave = st.sidebar.toggle("Salva automaticamente CSV (curva + aggregato)", value=True)

# -------------------- Utils --------------------
def load_dataset():
    for p in DATASET_PATHS:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, parse_dates=["Date"])
                # compat: alcuni file hanno 'E_INT_Daily_KWh'
                if "E_INT_Daily_KWh" in df.columns and TARGET_COL not in df.columns:
                    df = df.rename(columns={"E_INT_Daily_KWh": TARGET_COL})
                return df.sort_values("Date").copy()
            except Exception as e:
                st.warning(f"Errore leggendo {p}: {e}")
    st.error("Dataset non trovato. Inserisci 'Dataset_Daily_EnergiaSeparata_2020_2025.csv' nella root.")
    return None

def train_model(df: pd.DataFrame):
    d = df.dropna(subset=[FEAT_COL, TARGET_COL]).copy()
    if d.empty:
        return None, {"mae": np.nan, "r2": np.nan, "n": 0}
    X = d[[FEAT_COL]].values
    y = d[TARGET_COL].values
    m = LinearRegression().fit(X, y)
    y_hat = m.predict(X)
    mae = float(np.mean(np.abs(y - y_hat)))
    r2  = float(m.score(X, y))
    return m, {"mae": mae, "r2": r2, "n": int(len(d))}

def make_map(lat, lon):
    m = folium.Map(location=[lat, lon], tiles="Esri.WorldImagery", zoom_start=16)
    popup_html = f"""
    <div style='background:#ffffffcc;padding:8px;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.25);font-size:13px'>
      <b>Impianto FV</b><br/>
      Lat: {lat:.6f} — Lon: {lon:.6f}<br/>
      Tilt: {tilt}° — Orient: {orient}°
    </div>
    """
    folium.Marker([lat,lon], tooltip="Impianto FV", popup=folium.Popup(popup_html, max_width=260)).add_to(m)
    return m

def meteomatics_fetch(lat, lon, start_iso, end_iso, step="PT15M"):
    user = os.environ.get("METEO_USER")
    pwd  = os.environ.get("METEO_PASS")
    if not user or not pwd:
        raise RuntimeError("Meteomatics: mancano METEO_USER/METEO_PASS")
    # Nota: se vuoi usare tilt/orient specifici di meteomatics, cambia in global_rad_tilt_X_orientation_Y
    url = f"https://api.meteomatics.com/{start_iso}--{end_iso}:{step}/global_rad:W/{lat},{lon}/json"
    r = requests.get(url, auth=(user, pwd), timeout=30)
    r.raise_for_status()
    js = r.json()
    dates = js["data"][0]["coordinates"][0]["dates"]
    rows = [(pd.to_datetime(d["date"]), float(d["value"])) for d in dates]
    df = pd.DataFrame(rows, columns=["time", "GlobalRad_W"]).sort_values("time")
    return url, df

def openmeteo_fetch(lat, lon, start_date, end_date):
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&hourly=shortwave_radiation&start_date={start_date}&end_date={end_date}&timezone=UTC")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    j = r.json()
    t = pd.to_datetime(j["hourly"]["time"])
    v = j["hourly"].get("shortwave_radiation", [np.nan]*len(t))
    dfh = pd.DataFrame({"time": t, "GlobalRad_W": v}).sort_values("time")
    df15 = dfh.set_index("time").resample("15min").interpolate().reset_index()
    return url, df15

def compute_curve(df, model, plant_kw):
    if df is None or df.empty:
        return None, 0.0, 0.0, 0.0
    df = df.copy()
    df["GlobalRad_W"] = df["GlobalRad_W"].clip(lower=0)
    # Semplice normalizzazione: distribuzione proporzionale della produzione sul profilo radiazione
    total_rad = df["GlobalRad_W"].sum()
    if total_rad <= 0:
        df["kWh_curve"] = 0.0
        df["kW_inst"] = 0.0
        return df, 0.0, 0.0, 0.0
    # Stima energia giornaliera con modello (se presente), altrimenti usa coeff proporzionale
    if "model" in st.session_state and st.session_state["model"] is not None:
        # trasforma la somma radiazione "a scala" del training: usiamo l'area (somma) come feature
        est_kwh = float(st.session_state["model"].predict([[total_rad]])[0])
    else:
        # fallback grezzo: converte W a Wh per step (15 min) e scala al kWp (molto approssimato)
        est_kwh = float(np.sum(df["GlobalRad_W"] * 0.25) / 1000.0)
    # distribuisci l'energia stimata sulla curva 15-min
    df["kWh_curve"] = est_kwh * (df["GlobalRad_W"] / total_rad)
    df["kW_inst"] = df["kWh_curve"] * 4.0
    peak_kW = float(df["kW_inst"].max())
    peak_pct = float((peak_kW / plant_kw) * 100.0) if plant_kw > 0 else 0.0
    return df, est_kwh, peak_kW, peak_pct

def download_button_bytes(label, data: bytes, filename: str, mime="text/csv"):
    st.download_button(label, data, file_name=filename, mime=mime)

# -------------------- Tabs --------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Storico", "🧠 Modello", "🔮 Previsioni 4 giorni (15m)", "🗺️ Mappa"])

with tab1:
    df = load_dataset()
    if df is not None:
        st.subheader("Storico – anteprima")
        st.dataframe(df.tail(200), use_container_width=True)
        st.subheader("Produzione storica (kWh)")
        st.line_chart(df.set_index("Date")[TARGET_COL])

with tab2:
    st.subheader("Addestra modello su dati storici")
    df = load_dataset()
    if df is not None:
        if st.button("Addestra / Riaddestra"):
            model, metrics = train_model(df)
            if model is not None:
                st.session_state["model"] = model
                st.success(f"Modello OK – MAE: {metrics['mae']:.1f} kWh | R²: {metrics['r2']:.3f} | N={metrics['n']}")
            else:
                st.error("Addestramento fallito.")
        if "model" in st.session_state and st.session_state["model"] is not None:
            st.info("Modello in memoria e pronto per le previsioni.")

with tab3:
    st.subheader("Previsioni a 15 minuti per 4 giorni (Ieri, Oggi, Domani, Dopodomani)")
    model = st.session_state.get("model", None)
    start_base = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    results = {}
    for label, offset in [("Ieri", -1), ("Oggi", 0), ("Domani", 1), ("Dopodomani", 2)]:
        day = start_base + timedelta(days=offset)
        start_iso = (day).strftime("%Y-%m-%dT00:00:00Z")
        end_iso   = (day + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
        st.markdown(f"### {label} – {day.date()}")
        try:
            if provider == "Meteomatics" or (provider == "Auto"):
                try:
                    url, dfp = meteomatics_fetch(lat, lon, start_iso, end_iso, step="PT15M")
                    provider_used = "Meteomatics"
                except Exception as e:
                    if provider == "Meteomatics":
                        raise
                    url, dfp = openmeteo_fetch(lat, lon, str(day.date()), str((day + timedelta(days=1)).date()))
                    provider_used = "Open‑Meteo"
                    st.warning(f"Meteomatics non disponibile: {e}. Uso Open‑Meteo.")
            else:
                url, dfp = openmeteo_fetch(lat, lon, str(day.date()), str((day + timedelta(days=1)).date()))
                provider_used = "Open‑Meteo"
            st.caption(f"Provider: **{provider_used}**")
            st.code(url, language="text")
        except Exception as e:
            st.error(f"Errore fetch previsioni: {e}")
            continue

        dfp, energy_kWh, peak_kW, peak_pct = compute_curve(dfp, model, plant_kw)

        if dfp is None or dfp.empty:
            st.warning("Nessun dato disponibile.")
            continue

        # Grafico
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Energia stimata", f"{energy_kWh:.1f} kWh")
        c2.metric("Picco stimato", f"{peak_kW:.1f} kW")
        c3.metric("% della targa", f"{peak_pct:.1f}%")
        c4.metric("Punti curva", f"{len(dfp)}")

        st.line_chart(dfp.set_index("time")[["kWh_curve"]].rename(columns={"kWh_curve":"kWh/15min"}))

        # Download CSV
        # curva 15-min
        buf = io.StringIO()
        dfp[["time","GlobalRad_W","kWh_curve","kW_inst"]].to_csv(buf, index=False)
        download_button_bytes(f"⬇️ Scarica curva 15-min ({label})", buf.getvalue().encode("utf-8"),
                              f"curva_{label.lower()}_15min.csv")
        # aggregato giornaliero
        buf2 = io.StringIO()
        pd.DataFrame([{"date": str(day.date()), "energy_kWh": energy_kWh, "peak_kW": peak_kW}]).to_csv(buf2, index=False)
        download_button_bytes(f"⬇️ Scarica aggregato ({label})", buf2.getvalue().encode("utf-8"),
                              f"daily_{label.lower()}.csv")

with tab4:
    st.subheader("Mappa impianto (satellitare)")
    st.caption("Popup migliorato con box chiaro; le info si leggono anche su imagery.")
    m = make_map(lat, lon)
    st_folium(m, width=None, height=550)
