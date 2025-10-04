# Streamlit app principale aggiornato (popup fix + box descrittivo)
import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Solar Forecast - ROBOTRONIX", layout="wide")

# --- Sidebar ---
st.sidebar.title("☀️ Menu delle impostazioni")
view_mode = st.sidebar.radio("Come visualizzare info impianto:", ["Popup su mappa", "Box laterale"])

# --- Titolo ---
st.title("☀️ Solar Forecast - ROBOTRONIX for IMEPOWER")

# --- Mappa ---
st.header("🗺️ Mappa impianto (satellitare)")

lat, lon = 40.643278, 16.986083
m = folium.Map(location=[lat, lon], zoom_start=16, tiles="Esri World Imagery")

# Popup con miglior leggibilità
popup_html = folium.Popup(
    '<div style="font-size:14px; font-weight:bold; background:white; padding:5px; border-radius:8px; color:black;">Impianto FV Robotronix<br>Lat: 40.643278<br>Lon: 16.986083</div>',
    max_width=300
)

folium.Marker(
    [lat, lon],
    tooltip="Impianto Robotronix",
    popup=popup_html,
    icon=folium.Icon(color="green", icon="info-sign")
).add_to(m)

map_out = st_folium(m, width=900, height=500)

# Box laterale descrittivo opzionale
if view_mode == "Box laterale":
    with st.sidebar.expander("📋 Dettagli impianto"):
        st.markdown("""
        **Impianto Robotronix - IMEPOWER**
        - Località: Taranto (Italia)
        - Latitudine: 40.643278
        - Longitudine: 16.986083
        - Tecnologia: FV policristallino
        """)
