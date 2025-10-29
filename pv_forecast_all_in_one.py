
# -*- coding: utf-8 -*-
"""
Solar Forecast - ROBOTRONIX (build semplificata e corretta)
- Fix timezone Europe/Rome
- Fetch Meteomatics opzionale (1H/PT15M)
- Load dataset con cache
- UI Streamlit minimale ma funzionante
"""

import os, json, numpy as np, pandas as pd
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px

# --------------------------- Config Streamlit -------------------------------
st.set_page_config(page_title='Solar Forecast - FIXED', layout='wide')

# ------------------------------- Globals -----------------------------------
DATA_PATH = os.environ.get('PV_DAILY_DATA', 'Dataset_Daily_EnergiaSeparata_2020_2025.csv')
LOG_DIR   = os.environ.get('PV_LOG_DIR', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

DEFAULT_LAT = float(os.environ.get('PV_LAT', '40.643278'))
DEFAULT_LON = float(os.environ.get('PV_LON', '16.986083'))
DEFAULT_TILT = float(os.environ.get('PV_TILT', '10'))
DEFAULT_ORIENT = float(os.environ.get('PV_ORIENT', '180'))
DEFAULT_PLANT_KW = float(os.environ.get('PV_PLANT_KW', '947.25'))

MM_USER = os.environ.get('MM_USER', '')
MM_PASS = os.environ.get('MM_PASS', '')

# ------------------------------ Utils --------------------------------------
from dateutil import tz

def fix_timezone(df: pd.DataFrame, column: str = "Date", tz_local: str = "Europe/Rome") -> pd.DataFrame:
    """Converte la colonna temporale all'ora locale Europe/Rome gestendo anche UTC."""
    if column not in df.columns:
        return df
    df[column] = pd.to_datetime(df[column], errors="coerce")
    # Se non è timezone-aware, assumo UTC e converto
    try:
        if getattr(df[column].dt, "tz", None) is None:
            df[column] = df[column].dt.tz_localize("UTC").dt.tz_convert(tz_local)
        else:
            df[column] = df[column].dt.tz_convert(tz_local)
    except Exception:
        # Fallback: localizza direttamente come Europe/Rome
        df[column] = df[column].dt.tz_localize(tz_local)
    return df

def write_log(**kwargs):
    try:
        ts = datetime.utcnow().strftime('%Y%m%d')
        with open(os.path.join(LOG_DIR, f'forecast_log_{ts}.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(kwargs, default=str) + '\n')
    except Exception:
        pass

# ------------------------------ Data ---------------------------------------
@st.cache_data(show_spinner=False, ttl=600)
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalizzo colonne note
    if 'Date' not in df.columns:
        # prova qualche variante comune
        for alt in ['date', 'DATA', 'timestamp', 'time']:
            if alt in df.columns:
                df = df.rename(columns={alt: 'Date'})
                break
    # timezone fix
    df = fix_timezone(df, column='Date')
    # rinomina eventuale colonna energia
    if 'E_INT_Daily_KWh' in df.columns and 'E_INT_Daily_kWh' not in df.columns:
        df = df.rename(columns={'E_INT_Daily_KWh': 'E_INT_Daily_kWh'})
    # colonne irraggiamento: prendo media se presenti più sensori
    g_cols = [c for c in df.columns if c.lower().startswith('g_m') or 'Wm2' in c]
    if g_cols:
        df['G_mean'] = df[g_cols].mean(axis=1)
    return df

# --------------------------- Meteomatics (opzionale) ------------------------
import requests

def _ensure_iso(dt: datetime) -> str:
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_meteomatics_hourly(lat: float, lon: float, start: datetime, end: datetime,
                             tilt: float|None=None, orient: float|None=None) -> pd.DataFrame:
    if not MM_USER or not MM_PASS:
        return pd.DataFrame()
    if tilt is not None and orient is not None and float(tilt) > 0:
        rad_param = f'global_rad_tilt_{int(round(float(tilt)))}_orientation_{int(round(float(orient)))}:W'
    else:
        rad_param = 'global_rad:W'
    params = f'{rad_param},total_cloud_cover:p,t_2m:C'
    url = f'https://api.meteomatics.com/{_ensure_iso(start)}--{_ensure_iso(end)}:PT1H/{params}/{lat},{lon}/json'
    r = requests.get(url, auth=(MM_USER, MM_PASS), timeout=30)
    if not r.ok:
        return pd.DataFrame()
    j = r.json()
    rows = []
    for blk in j.get('data', []):
        prm = blk.get('parameter')
        col = 'GlobalRad_W'
        if prm == 'total_cloud_cover:p':
            col = 'CloudCover_P'
        elif prm == 't_2m:C':
            col = 'Temp_Air'
        for d in blk.get('coordinates', [{}])[0].get('dates', []):
            rows.append({'time': d.get('date'), col: d.get('value')})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.groupby('time', as_index=False).mean().sort_values('time')
    df = fix_timezone(df, column='time')
    return df

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_meteomatics_pt15m(lat: float, lon: float, start: datetime, end: datetime,
                             tilt: float|None=None, orient: float|None=None) -> pd.DataFrame:
    if not MM_USER or not MM_PASS:
        return pd.DataFrame()
    if tilt is not None and orient is not None and float(tilt) > 0:
        rad_param = f'global_rad_tilt_{int(round(float(tilt)))}_orientation_{int(round(float(orient)))}:W'
    else:
        rad_param = 'global_rad:W'
    params = f'{rad_param},total_cloud_cover:p,t_2m:C'
    url = f'https://api.meteomatics.com/{_ensure_iso(start)}--{_ensure_iso(end)}:PT15M/{params}/{lat},{lon}/json'
    r = requests.get(url, auth=(MM_USER, MM_PASS), timeout=30)
    if not r.ok:
        return pd.DataFrame()
    j = r.json()
    rows = []
    for blk in j.get('data', []):
        prm = blk.get('parameter')
        col = 'GlobalRad_W'
        if prm == 'total_cloud_cover:p':
            col = 'CloudCover_P'
        elif prm == 't_2m:C':
            col = 'Temp_Air'
        for d in blk.get('coordinates', [{}])[0].get('dates', []):
            rows.append({'time': d.get('date'), col: d.get('value')})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.groupby('time', as_index=False).mean().sort_values('time')
    df = fix_timezone(df, column='time')
    return df

# ------------------------------- UI ----------------------------------------
st.title('⚡ PV Forecast – versione corretta (timezone fixed)')

col1, col2, col3 = st.columns(3)
with col1:
    lat = st.number_input('Lat', value=DEFAULT_LAT, format="%.6f")
with col2:
    lon = st.number_input('Lon', value=DEFAULT_LON, format="%.6f")
with col3:
    plant_kw = st.number_input('Potenza impianto [kW]', value=float(DEFAULT_PLANT_KW), step=10.0)

df = load_data(DATA_PATH)

st.success(f"Dataset caricato: {len(df)} righe – timezone: {df['Date'].dt.tz} ")

# Grafico energia giornaliera
if 'E_INT_Daily_kWh' in df.columns:
    fig_e = px.line(df, x='Date', y='E_INT_Daily_kWh', title='Energia Giornaliera [kWh] (ora locale)',
                    labels={'Date':'Ora (Europe/Rome)'})
    st.plotly_chart(fig_e, use_container_width=True)

# Grafico irraggiamento medio (se presente)
if 'G_mean' in df.columns:
    fig_g = px.line(df, x='Date', y='G_mean', title='Irraggiamento medio (sensori) [W/m²] (ora locale)',
                    labels={'Date':'Ora (Europe/Rome)'})
    st.plotly_chart(fig_g, use_container_width=True)

# Meteo (opzionale) – ultimo giorno
st.subheader('Meteo (Meteomatics opzionale)')
today = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
start = today - timedelta(hours=24)
dfm = fetch_meteomatics_hourly(lat, lon, start, today, DEFAULT_TILT, DEFAULT_ORIENT)
if not dfm.empty:
    fig_m = px.line(dfm, x='time', y=[c for c in ['GlobalRad_W','CloudCover_P','Temp_Air'] if c in dfm.columns],
                    title='Meteo orario (ora locale)')
    st.plotly_chart(fig_m, use_container_width=True)
else:
    st.info('Dati Meteomatics non disponibili (controllare credenziali MM_USER/MM_PASS nei secrets).')
