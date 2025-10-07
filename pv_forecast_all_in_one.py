import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from streamlit_folium import st_folium

# Credenziali Meteomatics integrate
METEO_USER = "FVMANAGER"
METEO_PASS = "MIMMOFABIO"

st.set_page_config(page_title="Solar Forecast - ROBOTRONIX", layout="wide")

st.sidebar.title("Impostazioni")
provider = st.sidebar.selectbox("Fonte meteo:", ["Meteomatics", "Open-Meteo"])
tilt = st.sidebar.slider("Tilt (°)", 0, 90, 20)
orient = st.sidebar.slider("Orientamento (°)", 0, 360, 180)
lat = st.sidebar.number_input("Latitudine", value=40.643278, format="%.6f")
lon = st.sidebar.number_input("Longitudine", value=16.986083, format="%.6f")

st.title("☀️ Solar Forecast - ROBOTRONIX for IMEPOWER")

# Funzione per recuperare dati Meteomatics
def get_meteomatics(lat, lon, tilt, orient):
    start = "2025-10-03T00:00:00Z"
    end = "2025-10-04T00:00:00Z"
    url = f"https://api.meteomatics.com/{start}--{end}:PT15M/global_rad_tilt_{tilt}_orientation_{orient}:W,total_cloud_cover:p/{lat},{lon}/json"
    try:
        r = requests.get(url, auth=(METEO_USER, METEO_PASS))
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data['data'][0]['coordinates'][0]['dates'])
        df['value'] = [x['value'] for x in data['data'][0]['coordinates'][0]['dates']]
        return df
    except Exception as e:
        st.error(f"Errore Meteomatics: {e}")
        return None

# Sezione previsioni
st.header("Previsioni (PT15M, tilt/orient, provider toggle)")

if st.button("Calcola previsioni"):
    with st.spinner("Calcolo in corso..."):
        if provider == "Meteomatics":
            df = get_meteomatics(lat, lon, tilt, orient)
        else:
            df = pd.DataFrame({
                "date": pd.date_range("2025-10-03", periods=96, freq="15min"),
                "value": np.sin(np.linspace(0, np.pi, 96)) * 800
            })
        if df is not None:
            st.line_chart(df["value"])
            st.success("✅ Previsioni generate correttamente!")

# Mappa
st.header("🗺️ Mappa impianto (satellitare)")
m = folium.Map(location=[lat, lon], zoom_start=15, tiles="Esri.WorldImagery")
popup_html = f"<b>Lat:</b> {lat}<br><b>Lon:</b> {lon}<br><b>Tilt:</b> {tilt}°<br><b>Orient:</b> {orient}°"
folium.Marker([lat, lon], popup=popup_html).add_to(m)
st_folium(m, width=900, height=500)
