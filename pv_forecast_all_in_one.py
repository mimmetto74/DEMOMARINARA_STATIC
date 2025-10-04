import os, io, requests, joblib
import pandas as pd, numpy as np
from datetime import datetime, timedelta, timezone
import streamlit as st
import folium
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="ROBOTRONIX – Solar Forecast", layout="wide")

# ---------------- Auth ----------------
if "auth" not in st.session_state:
    st.session_state["auth"] = False
if not st.session_state["auth"]:
    st.title("🔐 Accesso richiesto")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u.strip().upper() == "FVMANAGER" and p == "MIMMOFABIO":
            st.session_state["auth"] = True
            st.rerun()
        else:
            st.error("Credenziali non valide.")
    st.stop()

# ---------------- Config ----------------
DEFAULT_LAT, DEFAULT_LON = 40.643278, 16.986083

st.sidebar.header("📋 Menu delle Impostazioni")
plant_name = st.sidebar.text_input("Nome impianto", value="Impianto FV")
plant_kw = st.sidebar.number_input("Potenza di targa (kW)", value=1000.0, step=50.0)
lat_sidebar = st.sidebar.number_input("Latitudine", value=DEFAULT_LAT, format="%.6f")
lon_sidebar = st.sidebar.number_input("Longitudine", value=DEFAULT_LON, format="%.6f")
tilt = st.sidebar.slider("Tilt (°)", min_value=0, max_value=90, value=20, step=1)
orient = st.sidebar.slider("Orientamento (°, 0=N, 90=E, 180=S, 270=W)", min_value=0, max_value=360, value=180, step=5)
provider_used = "Meteomatics"
status_used = "OK"

tab1, tab2 = st.tabs(["🔮 Previsioni","🗺️ Mappa"])

with tab2:
    st.subheader("Mappa impianto (satellitare)")
    # Box descrittivo sopra la mappa
    st.info(f"**{plant_name}**\n\n"
            f"⚡ Potenza di targa: {plant_kw:.0f} kW\n\n"
            f"🌍 Lat: {lat_sidebar:.6f}, Lon: {lon_sidebar:.6f}\n\n"
            f"📐 Tilt: {tilt}°, Orientamento: {orient}°\n\n"
            f"☁️ Provider: {provider_used} | Stato: {status_used}")

    # Mappa Folium incorniciata
    m = folium.Map(location=[lat_sidebar, lon_sidebar], zoom_start=15, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite"
    ).add_to(m)
    folium.Marker([lat_sidebar, lon_sidebar], tooltip=plant_name).add_to(m)
    st_folium(m, width=900, height=500)
