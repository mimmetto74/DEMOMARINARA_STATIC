import os
import io
import json
import math
import time
import base64
import zipfile
import folium
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from dateutil import tz

import streamlit as st
from streamlit_folium import st_folium

# -----------------------------
# Config base app
# -----------------------------
st.set_page_config(
    page_title="Solar Forecast - ROBOTRONIX for IMEPOWER",
    layout="wide",
    page_icon="☀️"
)

# -----------------------------
# Utility
# -----------------------------
LOCAL_TZ = tz.gettz("Europe/Rome")

def to_local(dt_utc: datetime) -> datetime:
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    return dt_utc.astimezone(LOCAL_TZ)

def day_span(day: datetime, freq_minutes: int = 15):
    """Ritorna start/end ISO per un giorno in UTC e la frequenza PT15M o PT1H"""
    start_local = day.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(LOCAL_TZ)
    end_local   = start_local + timedelta(days=1)
    start_utc = start_local.astimezone(timezone.utc)
    end_utc   = end_local.astimezone(timezone.utc)
    iso_s = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    iso_e = end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    step  = f"PT{freq_minutes}M"
    return iso_s, iso_e, step

def ensure_pt15m(df, col_power_wm2, tzinfo=LOCAL_TZ):
    """Riporta a PT15M (se serve) e imposta timezone locale."""
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(tzinfo)
    df = df.set_index("time").sort_index()
    # se già a 15', mantieni; se oraria, upsample + interp
    infreq = pd.infer_freq(df.index)
    if infreq in ("15T","15min"):
        return df
    # upsample a 15'
    df = df.resample("15min").interpolate(method="time")
    return df

def irradiance_to_power_kw(g_wm2, p_peak_kw, efficiency):
    """Modello semplice: P[kW] = P_peak[kW]·(G/1000)·Eff."""
    return p_peak_kw * (g_wm2 / 1000.0) * efficiency

def export_csv_curva(df_15m, giorno_label, provider, tilt, orient, p_peak_kw, eff):
    """Ritorna bytes CSV curva 15' + righe di header con meta/aggregati."""
    df = df_15m.copy()
    # aggregati
    energy_kwh_15m = df["P_kW"] * 0.25
    energy_day_kwh = energy_kwh_15m.sum()
    peak_kw = df["P_kW"].max()
    peak_pct = 0 if p_peak_kw <= 0 else (peak_kw / p_peak_kw) * 100.0
    mean_cc = df.get("cloudcover", pd.Series([np.nan])).mean()

    meta = {
        "provider": provider,
        "giorno": giorno_label,
        "tilt_deg": tilt,
        "orient_deg": orient,
        "p_peak_kw": p_peak_kw,
        "efficiency": eff,
        "energy_day_kwh": round(energy_day_kwh, 2),
        "peak_kw": round(peak_kw, 2),
        "peak_pct": round(peak_pct, 1),
        "cloudcover_mean_pct": round(float(mean_cc), 1) if not np.isnan(mean_cc) else None
    }

    buf = io.StringIO()
    buf.write("# META\n")
    buf.write(json.dumps(meta, ensure_ascii=False) + "\n")
    buf.write("# CURVA_15MIN\n")
    curva = df[["G_Wm2","P_kW"]].copy()
    if "cloudcover" in df.columns:
        curva["cloudcover"] = df["cloudcover"]
    curva.to_csv(buf)
    return buf.getvalue().encode("utf-8")

# -----------------------------
# Provider: Meteomatics
# -----------------------------
def get_meteomatics_pt15m(lat, lon, day_dt, tilt_deg, orient_deg, user, passwd):
    """global_rad_tilt_<tilt>_orientation_<orient>:W, total_cloud_cover:p a PT15M"""
    start_iso, end_iso, step = day_span(day_dt, 15)
    param = f"global_rad_tilt_{int(round(tilt_deg))}_orientation_{int(round(orient_deg))}:W,total_cloud_cover:p"
    url = f"https://api.meteomatics.com/{start_iso}--{end_iso}:{step}/{param}/{lat},{lon}/json"
    r = requests.get(url, auth=(user, passwd), timeout=30)
    r.raise_for_status()
    js = r.json()

    # parsing
    data = js["data"]
    # global_rad
    g_data = [d for d in data if d["parameter"].startswith("global_rad_tilt_")]
    cc_data = [d for d in data if d["parameter"] == "total_cloud_cover:p"]

    rows = []
    # uso la prima serie di global_rad
    if g_data:
        for v in g_data[0]["coordinates"][0]["dates"]:
            rows.append({"time": v["date"], "G_Wm2": v["value"]})
    df = pd.DataFrame(rows)

    # merge cloudcover
    if cc_data:
        cc_rows = []
        for v in cc_data[0]["coordinates"][0]["dates"]:
            cc_rows.append({"time": v["date"], "cloudcover": v["value"]})
        dcc = pd.DataFrame(cc_rows)
        df = df.merge(dcc, on="time", how="left")
    df = ensure_pt15m(df, "G_Wm2", LOCAL_TZ)

    debug_url = url
    return df, debug_url

# -----------------------------
# Provider: Open-Meteo (fallback)
# -----------------------------
def get_openmeteo_pt15m(lat, lon, day_dt):
    """Usa shortwave_radiation (Wh/m²) oraria e cloudcover, normalizza a 15'."""
    # 1 giorno locale
    d0 = day_dt.astimezone(LOCAL_TZ).strftime("%Y-%m-%d")
    d1 = (day_dt + timedelta(days=1)).astimezone(LOCAL_TZ).strftime("%Y-%m-%d")
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=shortwave_radiation,cloudcover"
        f"&start_date={d0}&end_date={d1}"
        "&timezone=Europe%2FRome"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    js = r.json()
    hourly = js.get("hourly", {})

    times = hourly.get("time", [])
    swr = hourly.get("shortwave_radiation", [])  # Wh/m² per ora
    cc  = hourly.get("cloudcover", [])

    df = pd.DataFrame({"time": times, "swr_Whm2": swr})
    if cc:
        df["cloudcover"] = cc

    # convert to datetime localized
    df["time"] = pd.to_datetime(df["time"], utc=False).dt.tz_localize(LOCAL_TZ)

    # convert Wh/m² per ora → W/m² media ora ≈ Wh/h
    df["G_Wm2"] = df["swr_Whm2"]  # ≈ potenza media
    df = df.drop(columns=["swr_Whm2"])

    # upsample a 15'
    df = df.set_index("time").sort_index()
    df = df.resample("15min").interpolate("time").reset_index()
    return df, url

# -----------------------------
# UI: Sidebar (Impostazioni)
# -----------------------------
with st.sidebar:
    st.header("Impostazioni")

    provider = st.selectbox("Fonte meteo", ["Meteomatics","Open-Meteo"], index=0)
    save_csv = st.checkbox("Salvataggio automatico CSV (curva + aggregato)", value=True)

    st.subheader("Potenza & Rendimento")
    p_peak_kw = st.number_input("Potenza di targa impianto (kW)", value=947.32, step=1.0, format="%.2f")
    efficiency = st.slider("Efficienza modello FV (PR complessiva)", min_value=0.6, max_value=1.0, value=0.88, step=0.01)

    st.subheader("Posizione & Piano")
    lat = st.number_input("Latitudine", value=40.643278, format="%.6f")
    lon = st.number_input("Longitudine", value=16.986083, format="%.6f")

    tilt = st.slider("Tilt (°)", 0, 90, 20)
    orientation = st.slider("Orientazione (°; 0=N, 90=E, 180=S, 270=W)", 0, 359, 180)

    st.subheader("Credenziali Meteomatics")
    default_user = os.getenv("METEO_USER","")
    default_pass = os.getenv("METEO_PASS","")
    meteo_user = st.text_input("Username", value=default_user, type="default", placeholder="utente@azienda")
    meteo_pass = st.text_input("Password", value=default_pass, type="password", placeholder="••••••••")

# -----------------------------
# HEADER / MENU
# -----------------------------
tabs = st.tabs(["📚 Storico", "📈 Previsioni 4 giorni (15m)", "🗺️ Mappa"])

# -----------------------------
#  TAB: Storico (opzionale – dataset utente)
# -----------------------------
with tabs[0]:
    st.subheader("Analisi Storica")
    path = "Dataset_Daily_EnergiaSeparata_2020_2025.csv"
    if os.path.exists(path):
        try:
            dfh = pd.read_csv(path, parse_dates=["Date"])
            st.success(f"Dataset caricato: {path}")
            cols = [c for c in dfh.columns if c.lower() not in ("date","time")]
            metric = st.selectbox("Seleziona metrica", cols, index=0)
            d1, d2 = st.date_input("Intervallo", (dfh["Date"].min().date(), dfh["Date"].max().date()))
            m = (dfh["Date"].dt.date >= d1) & (dfh["Date"].dt.date <= d2)
            st.line_chart(dfh.loc[m].set_index("Date")[metric])
        except Exception as e:
            st.error(f"Errore lettura dataset: {e}")
    else:
        st.info("Carica il file 'Dataset_Daily_EnergiaSeparata_2020_2025.csv' nella root per attivare l’analisi storica.")

# -----------------------------
# Funzione comune per una giornata
# -----------------------------
def run_day(day_dt, tag):
    """Esegue la previsione per una data (ieri/oggi/domani/dopodomani). Ritorna df15m + summary + url + provider effettivo."""
    try_provider = provider

    # 1) prova Meteomatics se scelto e credenziali ok
    url_dbg = ""
    used = ""
    if try_provider == "Meteomatics" and meteo_user and meteo_pass:
        try:
            df15, url_dbg = get_meteomatics_pt15m(lat, lon, day_dt, tilt, orientation, meteo_user, meteo_pass)
            used = "Meteomatics"
        except Exception as e:
            st.warning(f"[{tag}] Meteomatics non disponibile, fallback a Open-Meteo. Errore: {e}")
            try_provider = "Open-Meteo"

    # 2) fallback Open-Meteo
    if try_provider == "Open-Meteo":
        df15, url_dbg = get_openmeteo_pt15m(lat, lon, day_dt)
        used = "Open-Meteo"

    # 3) calcolo potenze/energia
    df15["P_kW"] = irradiance_to_power_kw(df15["G_Wm2"], p_peak_kw, efficiency)
    energy_day_kwh = (df15["P_kW"] * 0.25).sum()
    peak_kw = df15["P_kW"].max()
    peak_pct = 0 if p_peak_kw <= 0 else (peak_kw / p_peak_kw) * 100.0
    cloud_avg = float(df15.get("cloudcover", pd.Series([np.nan])).mean())

    # 4) salva CSV
    if save_csv:
        label = to_local(day_dt).strftime("%Y-%m-%d")
        csv_bytes = export_csv_curva(df15.set_index(df15.index), label, used, tilt, orientation, p_peak_kw, efficiency)
        st.download_button(
            f"⬇️ Scarica curva 15-min ({tag})",
            data=csv_bytes,
            file_name=f"curve15min_{tag}_{label}.csv",
            mime="text/csv"
        )

    # 5) UI
    c1,c2,c3,c4 = st.columns([1,1,1,1])
    c1.metric("Energia stimata", f"{energy_day_kwh:,.1f} kWh")
    c2.metric("Picco stimato", f"{peak_kw:,.1f} kW")
    c3.metric("% della targa", f"{peak_pct:,.1f}%")
    if not math.isnan(cloud_avg):
        c4.metric("Nuvolosità media", f"{cloud_avg:,.0f}%")
    else:
        c4.metric("Nuvolosità media", "n/a")

    st.caption(f"Provider: **{used}** | URL: `{url_dbg}`")
    st.line_chart(df15[["P_kW"]])

    return df15, used, url_dbg

# -----------------------------
#  TAB: Previsioni 4 giorni (15m)
# -----------------------------
with tabs[1]:
    st.subheader("Previsioni (PT15M, tilt/orient, provider toggle)")

    if st.button("Calcola previsioni (Ieri/Oggi/Domani/Dopodomani)", type="primary"):
        today_local = datetime.now(LOCAL_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
        days = [
            ("Ieri",        today_local - timedelta(days=1)),
            ("Oggi",        today_local),
            ("Domani",      today_local + timedelta(days=1)),
            ("Dopodomani",  today_local + timedelta(days=2)),
        ]
        for tag, d in days:
            st.markdown(f"### {tag}")
            run_day(d.astimezone(timezone.utc), tag)

# -----------------------------
#  TAB: Mappa
# -----------------------------
with tabs[2]:
    st.subheader("Mappa impianto (satellitare)")
    m = folium.Map(location=[lat, lon], zoom_start=16, tiles="Esri.WorldImagery")
    folium.Marker(
        [lat, lon],
        tooltip="Impianto FV",
        icon=folium.Icon(color="green", icon="leaf")
    ).add_to(m)

    # piccolo riquadro laterale (legend-like) con info
    html_box = f"""
    <div style="position: fixed; top: 80px; right: 40px; z-index: 9999;
                background: rgba(20,20,20,0.85); color:#fff; padding: 12px 14px;
                border-radius: 10px; font-size: 13px; border: 1px solid rgba(255,255,255,0.15);">
        <b>Impianto</b><br>
        Lat/Lon: {lat:.6f}, {lon:.6f}<br>
        Tilt: {tilt}° – Orient: {orientation}°<br>
        P. targa: {p_peak_kw:.1f} kW<br>
        Efficienza: {efficiency:.2f}
    </div>
    """
    folium.Marker([lat, lon], popup="Impianto FV").add_to(m)
    m.get_root().html.add_child(folium.Element(html_box))
    st_folium(m, width=None, height=620)
