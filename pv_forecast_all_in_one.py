# -*- coding: utf-8 -*-
import os
import io
import math
import json
import time
import pytz
import base64
import altair as alt
import pandas as pd
import numpy as np
import requests
import streamlit as st
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# =========================
# CONFIGURAZIONE GENERALE
# =========================
APP_TZ = ZoneInfo("Europe/Rome")   # timezone locale impianto
UTC = ZoneInfo("UTC")

# Credenziali Meteomatics (da ENV, così non appaiono in UI)
METEO_USER = os.getenv("METEO_USER", "")
METEO_PASS = os.getenv("METEO_PASS", "")

# Mappa orientamento: 0=N, 90=E, 180=S, 270=W (Meteomatics accetta il grado numerico)
# Il parametro finale sarà: global_rad_tilt_{tilt}_orientation_{azimuth}:W
# NB: :W indica unità (W/m²). Usiamo sempre i Watt per il calcolo poi convertiamo.

# ===== Funzioni utili =====
def _to_local_midnight(dt: datetime) -> datetime:
    """Porta un datetime naive a mezzanotte locale (Europe/Rome)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=APP_TZ)
    dt0 = dt.astimezone(APP_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    return dt0

def _local_midnight_utc(day_offset=0) -> tuple[str, str]:
    """Restituisce start/end (stringhe ISO) da mezzanotte locale a mezzanotte+1 giorno, convertiti in UTC."""
    today_local = datetime.now(APP_TZ)
    start_local = _to_local_midnight(today_local + timedelta(days=day_offset))
    end_local   = start_local + timedelta(days=1)
    start_utc = start_local.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_utc   = end_local.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return start_utc, end_utc

def _resample_to_15min(df_utc: pd.DataFrame, tz=APP_TZ) -> pd.DataFrame:
    """Resample a 15 minuti sul timezone locale, poi ritorna con index in locale."""
    if df_utc.empty:
        return df_utc
    df = df_utc.copy()
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(tz)
    # riporta a 15 min
    df = df.resample("15min").interpolate(limit_direction="both")
    return df

def _meters2_to_kwh_per_15min(w_m2: pd.Series) -> pd.Series:
    """W/m² → kWh/m² su slot di 15 minuti."""
    return w_m2.fillna(0) * (0.25 / 1000.0)  # 0.25h / 1000

def _poa_to_power_kW(irr_wm2: pd.Series, kWp: float, pr: float) -> pd.Series:
    """Potenza FV stimata (kW) da irradianza POA (W/m²) ~ (W/m² / 1000) * kWp * PR."""
    return (irr_wm2.clip(lower=0) / 1000.0) * float(kWp) * float(pr)

def _daily_kwh_from_curve_kW(p_kW: pd.Series) -> float:
    """Somma 15-min (kWh) da potenza kW."""
    # energia 15min (kWh) = p_kW * 0.25
    return float((p_kW.fillna(0) * 0.25).sum())

def _download_csv_button(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=True).encode("utf-8")
    st.download_button(label, data=csv, file_name=filename, mime="text/csv")

# =========================
#    PROVIDER: Meteomatics
# =========================
def fetch_meteomatics(lat, lon, start_iso, end_iso, tilt, azimuth) -> pd.DataFrame:
    """
    Richiama Meteomatics a PT15M con global_rad_tilt/orientation (POA) + total_cloud_cover.
    Ritorna DataFrame con index UTC e colonne ['poa_Wm2','cloud_p'].
    """
    if not METEO_USER or not METEO_PASS:
        raise RuntimeError("Credenziali Meteomatics mancanti in ENV (METEO_USER/METEO_PASS).")

    param = f"global_rad_tilt_{int(round(tilt))}_orientation_{int(round(azimuth))}:W,total_cloud_cover:p"
    url = (
        f"https://api.meteomatics.com/"
        f"{start_iso}--{end_iso}:PT15M/{param}/{lat},{lon}/json"
    )

    # Log solo in console, non in UI
    try:
        r = requests.get(url, auth=(METEO_USER, METEO_PASS), timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as ex:
        raise RuntimeError(f"Errore Meteomatics: {ex}")

    # Parse JSON Meteomatics
    # Struttura: data['data'] -> lista parametri, ognuno con 'parameter' e 'coordinates' ...
    try:
        entries = data["data"]
        poa_series = None
        cloud_series = None
        for block in entries:
            par = block["parameter"]
            coords = block["coordinates"][0]["dates"]
            ts = [pd.to_datetime(d["date"]).tz_localize("UTC") for d in block["coordinates"][0]["dates"]]
            vals = [d["value"] for d in coords]
            s = pd.Series(vals, index=ts)

            if par.startswith("global_rad_tilt_"):
                poa_series = s
            elif par == "total_cloud_cover":
                cloud_series = s

        df = pd.DataFrame(index=poa_series.index)
        df["poa_Wm2"] = poa_series
        if cloud_series is not None:
            df["cloud_p"] = cloud_series
        else:
            df["cloud_p"] = np.nan
        return df
    except Exception as ex:
        raise RuntimeError(f"Parse Meteomatics fallita: {ex}")

# ========================
#     PROVIDER: Open-Meteo
# ========================
def fetch_openmeteo(lat, lon, start_iso, end_iso, tilt, azimuth) -> pd.DataFrame:
    """
    Open-Meteo: prendo hourly direct_radiation + diffuse_radiation + shortwave_radiation.
    Interpolo a 15 min e approssimo POA ~ shortwave_radiation (W/m²).
    (Non esiste parametro "tilt/orientation" diretto su Open-Meteo.)
    """
    # costruiamo range date locale (Open-Meteo usa date locali nella query), ma usiamo start/end locali derivati:
    start_local = pd.to_datetime(start_iso).tz_convert(APP_TZ).strftime("%Y-%m-%d")
    end_local   = (pd.to_datetime(end_iso).tz_convert(APP_TZ) - pd.Timedelta(seconds=1)).strftime("%Y-%m-%d")

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=shortwave_radiation,direct_radiation,diffuse_radiation,cloudcover"
        f"&start_date={start_local}&end_date={end_local}"
        "&timezone=UTC"
    )
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        js = r.json()
    except Exception as ex:
        raise RuntimeError(f"Errore Open-Meteo: {ex}")

    try:
        times = pd.to_datetime(js["hourly"]["time"]).tz_localize("UTC")
        df = pd.DataFrame(index=times)
        # shortwave_radiation: W/m2 (global on horizontal, istantanea/average hour)
        df["poa_Wm2_raw"] = js["hourly"].get("shortwave_radiation", [np.nan]*len(times))
        df["cloud_p"] = js["hourly"].get("cloudcover", [np.nan]*len(times))
        # Interpolo a 15 min in locale poi torno alla serie locale
        df = _resample_to_15min(df, tz=APP_TZ)
        # Approssimazione: correggo lievemente per tilt con fattore coseno giornaliero (very light-touch)
        # per evitare curve troppo piatte, ma senza introdurre errori grossolani.
        # Fattore semplice: f = 0.92 + 0.08*cos((tilt-30°)/60° * pi)
        f = 0.92 + 0.08 * math.cos(math.radians((tilt - 30.0) / 60.0 * 180.0))
        df["poa_Wm2"] = (df["poa_Wm2_raw"].clip(lower=0)) * float(f)
        df.drop(columns=["poa_Wm2_raw"], inplace=True)
        return df.tz_convert(UTC)  # mantenere index in UTC internamente
    except Exception as ex:
        raise RuntimeError(f"Parse Open-Meteo fallita: {ex}")

# ========================
#      PIPELINE GIORNO
# ========================
def compute_day(provider, lat, lon, tilt, azimuth, kWp, pr, day_offset):
    """Restituisce dict con df (locale), kWh_giorno, peak_kW, peak_pct, cloud_mean, provider_used."""
    start_utc, end_utc = _local_midnight_utc(day_offset)
    # Hidden logs solo su console
    # print(f"[DEBUG] range UTC {start_utc}→{end_utc}, provider={provider}")

    if provider == "Meteomatics":
        df_utc = fetch_meteomatics(lat, lon, start_utc, end_utc, tilt, azimuth)
    else:
        df_utc = fetch_openmeteo(lat, lon, start_utc, end_utc, tilt, azimuth)

    # porta a 15min locale
    df = _resample_to_15min(df_utc, tz=APP_TZ)
    # calcola potenza stimata
    df["P_kW"] = _poa_to_power_kW(df["poa_Wm2"], kWp, pr)
    day_kWh = _daily_kwh_from_curve_kW(df["P_kW"])
    peak_kW = float(df["P_kW"].max())
    peak_pct = 0.0 if kWp <= 0 else (peak_kW / kWp) * 100.0
    cloud_mean = float(df["cloud_p"].mean()) if "cloud_p" in df.columns else np.nan

    return {
        "df": df,
        "kWh": day_kWh,
        "peak_kW": peak_kW,
        "peak_pct": peak_pct,
        "cloud_mean": cloud_mean,
        "provider": provider,
        "start": df.index.min(),
    }

# ========================
#            UI
# ========================
st.set_page_config(page_title="Solar Forecast - ROBOTRONIX", page_icon="🔆", layout="wide")

# Sidebar – in stile V4
st.sidebar.title("Impostazioni")

provider = st.sidebar.selectbox("Fonte meteo:", ["Open-Meteo", "Meteomatics"], index=0)
kWp = st.sidebar.number_input("Potenza di targa impianto (kWp)", min_value=0.1, step=0.01, value=947.32)
pr  = st.sidebar.slider("Efficienza/Performance Ratio (0.70–0.98)", 0.60, 0.99, 0.90, 0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Posizione & Piano")
lat = st.sidebar.number_input("Latitudine", value=40.643278, step=0.000001, format="%.6f")
lon = st.sidebar.number_input("Longitudine", value=16.986083, step=0.000001, format="%.6f")
tilt = st.sidebar.slider("Tilt (°)", 0, 60, 20)
azimuth = st.sidebar.slider("Orientazione (°, 0=N, 90=E, 180=S, 270=W)", 0, 359, 180)

st.sidebar.markdown("---")
save_curves = st.sidebar.checkbox("Salvataggio automatico CSV (curva + aggregato)", value=False)

st.title("🔭 Previsioni (PT15M, tilt/orient, provider toggle)")

# Pulsante calcolo
if st.button("Calcola previsioni (Ieri/Oggi/Domani/Dopodomani)", type="primary"):
    st.session_state["go"] = True

if "go" not in st.session_state:
    st.info("Premi il bottone per generare le curve a 15 minuti dei prossimi 4 giorni (Ieri/Oggi/Domani/Dopodomani).")
    st.stop()

# 4 giorni: -1, 0, +1, +2
labels = [
    ("Ieri", -1),
    ("Oggi", 0),
    ("Domani", +1),
    ("Dopodomani", +2),
]

for label, offs in labels:
    with st.container(border=True):
        try:
            res = compute_day(provider, lat, lon, tilt, azimuth, kWp, pr, offs)
            df = res["df"]
            # KPI
            c1,c2,c3,c4 = st.columns(4)
            c1.metric(f"Energia stimata {label.lower()}", f"{res['kWh']:.1f} kWh")
            c2.metric("Picco stimato", f"{res['peak_kW']:.1f} kW")
            c3.metric("% della targa", f"{res['peak_pct']:.1f}%")
            c4.metric("Nuvolosità media", f"{res['cloud_mean']:.0f}%" if not math.isnan(res["cloud_mean"]) else "—")

            # Grafico
            chart = alt.Chart(df.reset_index().rename(columns={"index":"ts"})).mark_line(opacity=0.95).encode(
                x=alt.X("ts:T", title="Ora (locale)"),
                y=alt.Y("P_kW:Q", title="Potenza [kW]"),
                tooltip=[
                    alt.Tooltip("ts:T", title="Ora"),
                    alt.Tooltip("P_kW:Q", title="kW", format=".1f"),
                    alt.Tooltip("poa_Wm2:Q", title="POA [W/m²]", format=".0f"),
                    alt.Tooltip("cloud_p:Q", title="Cloud [%]", format=".0f")
                ]
            ).properties(height=280)
            st.altair_chart(chart, use_container_width=True)

            # Download CSV
            if save_curves:
                fname_curve = f"curva_15min_{label.lower()}.csv"
                _download_csv_button(df[["poa_Wm2","P_kW","cloud_p"]], fname_curve, f"⬇️ Scarica curva 15-min ({label.lower()})")

                # Aggregato: unico numero, salvo come csv a 1 riga
                agg = pd.DataFrame({
                    "day":[df.index.min().date().isoformat()],
                    "kWh":[round(res["kWh"],1)],
                    "peak_kW":[round(res["peak_kW"],1)],
                    "peak_pct":[round(res["peak_pct"],1)],
                    "cloud_mean":[round(res["cloud_mean"],1) if not math.isnan(res["cloud_mean"]) else ""],
                    "provider":[res["provider"]],
                    "tilt":[tilt],
                    "azimuth":[azimuth],
                    "kWp":[kWp],
                    "PR":[pr]
                })
                _download_csv_button(agg, f"aggregato_{label.lower()}.csv", f"⬇️ Scarica aggregato ({label.lower()})")

        except Exception as ex:
            # Nessuna URL Meteomatics esposta: messaggio sintetico
            st.error(f"Errore fetch previsioni: {ex}")

st.caption("Provider attivo: **{0}** – Logs API non esposti in UI.".format(provider))

