
import os
import folium
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, timezone
from sklearn.linear_model import LinearRegression

APP_TITLE = "☀️ Solar Forecast - ROBOTRONIX for IMEPOWER"
DATASET_PATH = "Dataset_Daily_EnergiaSeparata_2020_2025.csv"

APP_USER = "FVMANAGER"
APP_PASS = "MIMMOFABIO"

METEO_USER = "teseospa-eiffageenergiesystemesitaly_daniello_fabio"
METEO_PASS = "6S8KTHPbrUlp6523T9Xd"

DEFAULT_LAT = 40.643278
DEFAULT_LON = 16.986083
PLANT_NOMINAL_KW = 100.0

def utcnow_iso():
    return datetime.now(timezone.utc)

def download_button(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv, file_name=filename, mime="text/csv")

def login_gate():
    if "logged" not in st.session_state:
        st.session_state.logged = False
    if st.session_state.logged:
        return True
    st.title("🔒 Accesso richiesto")
    with st.form("login"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        ok = st.form_submit_button("Login")
    if ok:
        if u.strip().upper() == APP_USER and p == APP_PASS:
            st.session_state.logged = True
            st.success("Accesso effettuato.")
            st.rerun()
        else:
            st.error("Credenziali non valide")
    st.stop()

@st.cache_data(show_spinner=False)
def load_dataset(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    return df

@st.cache_data(show_spinner=False)
def fit_daily_model(df: pd.DataFrame):
    ddf = df.dropna(subset=["E_INT_Daily_kWh", "G_M0_Wm2"]).copy()
    if len(ddf) < 3:
        return None, None
    X = ddf[["G_M0_Wm2"]].values.reshape(-1,1)
    y = ddf["E_INT_Daily_kWh"].values
    model = LinearRegression().fit(X, y)
    mae = float(np.mean(np.abs(model.predict(X)-y)))
    return model, mae

def compute_alpha_from_history(df: pd.DataFrame):
    ddf = df.dropna(subset=["E_INT_Daily_kWh", "G_M0_Wm2"]).copy()
    if len(ddf)==0:
        return 0.001
    Pavg = (ddf["E_INT_Daily_kWh"] / 24.0).values
    k_list = Pavg / np.maximum(ddf["G_M0_Wm2"].values, 1e-6)
    k = float(np.median(k_list))
    if not np.isfinite(k) or k<=0:
        k = 0.001
    return k

def meteomatics_timeseries(lat, lon, start_dt, end_dt, step="PT15M"):
    start = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    end = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    params = "global_rad:W,total_cloud_cover:p"
    url = f"https://api.meteomatics.com/{start}--{end}:{step}/{params}/{lat},{lon}/json"
    r = requests.get(url, auth=(METEO_USER, METEO_PASS), timeout=20)
    r.raise_for_status()
    js = r.json()
    out = []
    for series in js.get("data", []):
        name = series["parameter"]
        for ts in series["coordinates"][0]["dates"]:
            out.append((ts["date"], name, ts["value"]))
    df = pd.DataFrame(out, columns=["time","param","value"])
    df["time"] = pd.to_datetime(df["time"])
    piv = df.pivot_table(index="time", columns="param", values="value", aggfunc="first").reset_index()
    return piv.sort_values("time")

def openmeteo_timeseries(lat, lon, start_dt, end_dt):
    start_date = start_dt.date().isoformat()
    end_date   = end_dt.date().isoformat()
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=shortwave_radiation,cloudcover"
        f"&start_date={start_date}&end_date={end_date}"
        "&timezone=UTC"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    js = r.json()
    times = pd.to_datetime(js["hourly"]["time"])
    rad   = np.array(js["hourly"].get("shortwave_radiation",[0]*len(times)))
    cld   = np.array(js["hourly"].get("cloudcover",[0]*len(times)))
    df = pd.DataFrame({"time": times, "global_rad:W": rad, "total_cloud_cover:p": cld})
    df = df.set_index("time").resample("15min").interpolate().reset_index()
    df = df[(df["time"]>=start_dt) & (df["time"]<=end_dt)]
    return df

def get_forecast_df(lat, lon, start_dt, end_dt, source="auto"):
    err=None
    if source in ("auto","meteo"):
        try:
            p = meteomatics_timeseries(lat,lon,start_dt,end_dt,"PT15M")
            p["source"]="meteomatics"; return p,None
        except Exception as e:
            err = f"Meteomatics error: {e}"
    if source in ("auto","openmeteo","fallback"):
        try:
            p = openmeteo_timeseries(lat,lon,start_dt,end_dt)
            p["source"]="open-meteo"; return p,err
        except Exception as e:
            err = (err + " | " + f"Open-Meteo error: {e}") if err else f"Open-Meteo error: {e}"
    return pd.DataFrame(), err

def power_curve_from_rad(df_times: pd.DataFrame, alpha_k=0.001):
    d = df_times.copy()
    if "global_rad:W" not in d.columns and "shortwave_radiation" in d.columns:
        d.rename(columns={"shortwave_radiation":"global_rad:W"}, inplace=True)
    if "global_rad:W" not in d.columns:
        d["global_rad:W"] = 0.0
    d["P_kW"] = np.maximum(0, d["global_rad:W"] * alpha_k)
    cols = ["time","P_kW","global_rad:W"]
    if "total_cloud_cover:p" in d.columns:
        cols.append("total_cloud_cover:p")
    return d[cols].copy()

def daily_kwh_from_curve(curve: pd.DataFrame):
    return float((curve.get("P_kW", pd.Series(0)) * 0.25).sum())

def page_historical_and_model(df_hist, model, mae, alpha_k):
    st.subheader("📊 Analisi Storica")
    with st.expander("Mostra dataset storico", expanded=False):
        st.dataframe(df_hist.tail(200), use_container_width=True)
        download_button(df_hist, "storico.csv", "Scarica CSV storico")
    st.subheader("🧠 Modello giornaliero (E = a*Irr + b)")
    if model is None:
        st.warning("Modello non addestrato (pochi dati).")
    else:
        st.write(f"MAE su training: **{mae:.2f} kWh**")
        st.code(f"E_kWh ≈ {float(model.coef_[0]):.6f} * G_M0_Wm2 + {float(model.intercept_):.2f}")
    st.info(f"Fattore α (kW per W/m²) ottenuto da storico: **{alpha_k:.6f}**")

def day_block(title, curve, nominal_kw):
    st.markdown(f"### {title}")
    if curve.empty:
        st.error("Nessun dato disponibile"); return
    day_kwh = daily_kwh_from_curve(curve)
    peak_kw = float(curve["P_kW"].max())
    peak_pct = (peak_kw / nominal_kw * 100.0) if nominal_kw>0 else 0.0
    c1,c2,c3 = st.columns(3)
    c1.metric("Energia stimata (kWh)", f"{day_kwh:.1f}")
    c2.metric("Picco (kW)", f"{peak_kw:.1f}")
    c3.metric("Picco su targa (%)", f"{peak_pct:.0f}%")
    st.line_chart(curve.set_index("time")[["P_kW"]])
    dstr = curve["time"].dt.date.iloc[0].isoformat()
    download_button(curve[["time","P_kW","global_rad:W"]], f"curva_15min_{dstr}.csv", "⬇️ Scarica curva 15-min")
    download_button(pd.DataFrame([{"date": dstr, "kWh": day_kwh}]), f"daily_{dstr}.csv", "⬇️ Scarica aggregato giornaliero")

def page_forecast(df_hist, alpha_k):
    st.subheader("⚡ Previsioni a 15 minuti per 4 giorni (Ieri, Oggi, Domani, Dopodomani)")
    lat = st.number_input("Latitudine", value=float(DEFAULT_LAT), format="%.6f")
    lon = st.number_input("Longitudine", value=float(DEFAULT_LON), format="%.6f")
    st.sidebar.header("Menu delle impostazioni")
    source = st.sidebar.radio("Fonte dati meteo", ["Auto (Meteomatics→Open‑Meteo)","Solo Meteomatics","Solo Open‑Meteo"], index=0)
    forced = "auto"
    if source == "Solo Meteomatics": forced = "meteo"
    if source == "Solo Open‑Meteo": forced = "openmeteo"
    nominal_kw = st.sidebar.number_input("Potenza nominale impianto [kW]", value=float(PLANT_NOMINAL_KW), min_value=1.0, step=1.0)
    now = utcnow_iso().replace(minute=0, second=0, microsecond=0)
    days = [("Ieri", now - timedelta(days=1)), ("Oggi", now), ("Domani", now + timedelta(days=1)), ("Dopodomani", now + timedelta(days=2))]
    for title, d0 in days:
        st.divider()
        start, end = d0, d0 + timedelta(days=1)
        df_times, err = get_forecast_df(lat, lon, start, end, source=forced)
        if err: st.warning(err)
        if df_times.empty:
            st.error("Nessun dato meteo disponibile"); continue
        curve = power_curve_from_rad(df_times, alpha_k=alpha_k)
        day_block(f"{title} – {d0.date().isoformat()}", curve, nominal_kw)

def page_map():
    st.subheader("🗺️ Mappa impianto (satellitare)")
    lat = st.number_input("Latitudine", value=float(DEFAULT_LAT), format="%.6f", key="map_lat")
    lon = st.number_input("Longitudine", value=float(DEFAULT_LON), format="%.6f", key="map_lon")
    m = folium.Map(location=[lat, lon], zoom_start=17, tiles="Esri.WorldImagery")
    popup_html = f'''<div style="font-size:13px;"><b>Impianto:</b> Marinara (Taranto)<br/><b>Lat/Lon:</b> {lat:.6f}, {lon:.6f}<br/><b>Pnom:</b> {PLANT_NOMINAL_KW:.0f} kW</div>'''
    folium.Marker([lat, lon], tooltip="Impianto",
                  popup=folium.Popup(folium.IFrame(popup_html, width=220, height=90), max_width=260)).add_to(m)
    legend_html = '''
    <div style="position: fixed; bottom: 30px; left: 30px; width: 250px;
                z-index:9999; font-size:12px; background: rgba(255,255,255,0.92);
                border:1px solid #888; border-radius:8px; padding:10px;">
        <b>Legenda</b><br>
        • Sfondo: Esri World Imagery<br>
        • Marker: posizione impianto<br>
        • Popup: info sintetiche
    </div>'''
    m.get_root().html.add_child(folium.Element(legend_html))
    st.components.v1.html(m._repr_html_(), height=600, scrolling=True)

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="☀️", layout="wide")
    login_gate()
    st.sidebar.markdown("### Menu delle impostazioni")
    st.title(APP_TITLE)
    tabs = st.tabs(["📈 Storico", "🧪 Modello", "⚡ Previsioni 4 giorni (15m)", "🗺️ Mappa"])
    with tabs[0]:
        df_hist = load_dataset(DATASET_PATH)
        st.session_state.df_hist = df_hist
        st.dataframe(df_hist.tail(200), use_container_width=True)
        download_button(df_hist, "storico.csv", "⬇️ Scarica CSV storico")
    with tabs[1]:
        df_hist = st.session_state.get("df_hist")
        model, mae = fit_daily_model(df_hist)
        alpha_k = compute_alpha_from_history(df_hist)
        st.session_state.model = model
        st.session_state.alpha_k = alpha_k
        st.subheader("🧠 Modello giornaliero (E = a*Irr + b)")
        if model is None:
            st.warning("Modello non addestrato (pochi dati).")
        else:
            st.write(f"MAE su training: **{mae:.2f} kWh**")
            st.code(f"E_kWh ≈ {float(model.coef_[0]):.6f} * G_M0_Wm2 + {float(model.intercept_):.2f}")
        st.info(f"Fattore α (kW per W/m²) ottenuto da storico: **{alpha_k:.6f}**")
    with tabs[2]:
        df_hist = st.session_state.get("df_hist")
        alpha_k = st.session_state.get("alpha_k", 0.001)
        page_forecast(df_hist, alpha_k)
    with tabs[3]:
        page_map()

if __name__ == "__main__":
    main()
