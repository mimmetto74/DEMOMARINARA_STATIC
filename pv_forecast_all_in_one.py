import os, io, requests, joblib
import pandas as pd, numpy as np
from datetime import datetime, timedelta, timezone
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import folium
from streamlit_folium import st_folium

# =============================
#  STILE E CONFIGURAZIONE UI
# =============================
st.set_page_config(page_title="ROBOTRONIX – Solar Forecast", page_icon="⚡", layout="wide")

st.markdown(
    """
    <style>
    /* Sfondo generale */
    .main {
        background-color: #f7fafc;
        color: #002b5b;
        font-family: 'Lato', sans-serif;
    }
    /* Titoli */
    h1, h2, h3 {
        color: #003366;
        font-family: 'Lato', sans-serif;
    }
    /* Pulsanti */
    .stButton > button {
        background: linear-gradient(90deg, #0077b6, #0096c7);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.6em 1.2em;
        font-size: 16px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #005b8a, #0077b6);
        color: #e0f7ff;
    }
    /* Input box */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #0077b6;
        background-color: #ffffff;
    }
    /* Header */
    .custom-header {
        text-align: center;
        padding: 15px;
        background: linear-gradient(90deg, #003366, #0077b6);
        color: white;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown(
    """
    <div class='custom-header'>
        <h1>⚡ ROBOTRONIX – Solar Forecast Dashboard</h1>
        <p>Previsioni energetiche fotovoltaiche con interfaccia moderna</p>
    </div>
    """, unsafe_allow_html=True
)

# =============================
#  AUTENTICAZIONE
# =============================
if "auth" not in st.session_state:
    st.session_state["auth"] = False

if not st.session_state["auth"]:
    st.subheader("🔐 Accesso richiesto")
    col1, col2 = st.columns([1,1])
    with col1:
        u = st.text_input("Username")
    with col2:
        p = st.text_input("Password", type="password")
    if st.button("Accedi"):
        if u.strip().upper() == "FVMANAGER" and p == "MIMMOFABIO":
            st.session_state["auth"] = True
            st.experimental_rerun()
        else:
            st.error("Credenziali non valide. Riprova.")
    st.stop()

# =============================
#  CARICAMENTO E CONFIGURAZIONE
# =============================
DATA_PATH = "Dataset_Daily_EnergiaSeparata_2020_2025.csv"
MODEL_PATH = "pv_model.joblib"
LOG_PATH = "forecast_log.csv"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

DEFAULT_LAT = 40.643278
DEFAULT_LON = 16.986083

# =============================
#  FUNZIONI (come nel codice originale)
# =============================
def normalize_real_csv(file_like):
    df = pd.read_csv(file_like, sep=';', decimal=',', engine='python')
    if df.shape[1] == 1:
        file_like.seek(0); df = pd.read_csv(file_like)
    df.columns = [str(c).strip() for c in df.columns]
    time_col = df.columns[0]
    val_col = df.columns[1] if len(df.columns) > 1 else None
    if val_col is None:
        raise ValueError("CSV non valido: servono 2 colonne (timestamp; valore).")
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    df[val_col] = pd.to_numeric(df[val_col], errors='coerce').fillna(0.0)
    s = df.set_index(time_col)[val_col].astype(float).rename('kWh_15m')
    s = s[~s.index.duplicated(keep='first')]
    idx = pd.date_range(s.index.min(), s.index.max(), freq='15T')
    s = s.reindex(idx).fillna(0.0)
    df15 = s.to_frame()
    df15['kW_inst'] = df15['kWh_15m'] * 4.0
    daily = df15['kWh_15m'].resample('D').sum().to_frame('kWh_day')
    daily['kW_peak'] = df15['kW_inst'].resample('D').max()
    return df15, daily

st.success("✅ Interfaccia caricata con stile professionale blu-marino! Pronta per l'integrazione del modello.")
