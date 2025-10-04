import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# ----------------- LOGIN -----------------
def check_password():
    def password_entered():
        if st.session_state["username"] == "FVMANAGER" and st.session_state["password"] == "MIMMOFABIO":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
    if "password_correct" not in st.session_state:
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password", on_change=password_entered)
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password", on_change=password_entered)
        st.error("Credenziali non valide")
        return False
    else:
        return True

if not check_password():
    st.stop()

# ----------------- APP -----------------
st.sidebar.title("☀️ Menu delle impostazioni")

DATA_PATH = "Dataset_Daily_EnergiaSeparata_2020_2025.csv"

# Caricamento dati storici
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
st.title("☀️ Solar Forecast - ROBOTRONIX for IMEPOWER")

# Allenamento modello
X = df[["G_M0_Wm2"]]
y = df["E_INT_Daily_kWh"]
model = LinearRegression().fit(X, y)

# Input utente
lat = st.sidebar.number_input("Latitudine", value=40.643278)
lon = st.sidebar.number_input("Longitudine", value=16.986083)
tilt = st.sidebar.slider("Tilt (°)", 0, 90, 30)
orientation = st.sidebar.selectbox("Orientamento", ["Sud", "Est", "Ovest"])

# Selezione fonte dati
use_meteomatics = st.sidebar.toggle("Usa Meteomatics", value=True)

# Funzione Meteomatics
def get_meteomatics(lat, lon):
    username = "teseospa-eiffageenergiesystemesitaly_daniello_fabio"
    password = "6S8KTHPbrUlp6523T9Xd"
    start = datetime.utcnow().strftime("%Y-%m-%dT00:00:00Z")
    end = (datetime.utcnow() + timedelta(days=3)).strftime("%Y-%m-%dT23:59:00Z")
    url = f"https://api.meteomatics.com/{start}--{end}:PT15M/global_rad:W/ {lat},{lon}/json"
    r = requests.get(url, auth=(username, password))
    if r.status_code == 200:
        data = r.json()["data"][0]["coordinates"][0]["dates"]
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = df["value"].astype(float)
        return df
    else:
        return None

# Previsioni
if st.button("Genera Previsione"):
    if use_meteomatics:
        forecast = get_meteomatics(lat, lon)
        if forecast is not None:
            st.success("✅ Dati Meteomatics ricevuti")
            forecast["Pred_kWh"] = model.predict(forecast[["value"]])
            for i in range(4):
                day = (datetime.utcnow().date() + timedelta(days=i-1))
                dff = forecast[forecast["date"].dt.date == day]
                if not dff.empty:
                    st.subheader(f"📅 Previsione {day}")
                    st.line_chart(dff.set_index("date")["Pred_kWh"])
                    # CSV giornaliero
                    daily = dff.groupby(dff["date"].dt.date)["Pred_kWh"].sum()
                    daily.to_csv(f"forecast_{day}.csv")
                    # Potenza di picco
                    peak = dff["Pred_kWh"].max()
                    st.metric("Potenza di picco stimata (%)", f"{(peak/peak.max())*100:.1f}%")
        else:
            st.error("Meteomatics non disponibile, uso Open-Meteo")
    else:
        st.warning("Uso Open-Meteo non ancora implementato")

# Mappa Folium
m = folium.Map(location=[lat, lon], zoom_start=14)
folium.Marker([lat, lon], popup="📍 Impianto Fotovoltaico").add_to(m)
st.subheader("🗺️ Mappa Impianto")
st_folium(m, width=700, height=400)
