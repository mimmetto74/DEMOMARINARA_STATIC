
import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta, timezone
import io

st.set_page_config(page_title="Solar Forecast - ROBOTRONIX", page_icon="☀️", layout="wide")

# -------------------------
# Login semplice
# -------------------------
def check_password():
    def on_enter():
        ok = (st.session_state.get("username","") == "FVMANAGER" and 
              st.session_state.get("password","") == "MIMMOFABIO")
        st.session_state["auth_ok"] = bool(ok)
    if "auth_ok" not in st.session_state:
        st.text_input("Username", key="username", on_change=on_enter)
        st.text_input("Password", key="password", type="password", on_change=on_enter)
        st.stop()
    if not st.session_state["auth_ok"]:
        st.text_input("Username", key="username", on_change=on_enter)
        st.text_input("Password", key="password", type="password", on_change=on_enter)
        st.error("Credenziali non valide")
        st.stop()

check_password()

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("📋 Menu delle Impostazioni")
section = st.sidebar.radio("Sezione:", ["Storico", "Modello", "Previsioni (15m) 4 giorni", "Mappa"])

lat = st.sidebar.number_input("Latitudine", value=40.643278, format="%.6f")
lon = st.sidebar.number_input("Longitudine", value=16.986083, format="%.6f")

source = st.sidebar.selectbox("Sorgente dati meteo", ["Meteomatics (pt15m)", "Open-Meteo (fallback)"])

# -------------------------
# Helpers
# -------------------------
def utc_floor_today():
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)

def meteomatics_df(lat, lon, start_iso, end_iso, interval="PT15M"):
    user = "teseospa-eiffageenergiesystemesitaly_daniello_fabio"
    pwd  = "6S8KTHPbrUlp6523T9Xd"
    url = f"https://api.meteomatics.com/{start_iso}--{end_iso}:{interval}/global_rad:W/{lat},{lon}/json"
    r = requests.get(url, auth=(user, pwd), timeout=30)
    r.raise_for_status()
    data = r.json()
    dates = data["data"][0]["coordinates"][0]["dates"]
    times  = pd.to_datetime([d["date"] for d in dates], utc=True)
    values = [d.get("value", np.nan) for d in dates]
    return pd.DataFrame({"time": times, "global_rad_W": values}).sort_values("time")

def openmeteo_df(lat, lon, days=4):
    # Open-Meteo non ha minutely_15 ovunque per radiazione, quindi uso hourly e interpolazione a 15m
    url = ("https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}&hourly=shortwave_radiation&forecast_days={days}&timezone=UTC")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    j = r.json()
    t = pd.to_datetime(j["hourly"]["time"], utc=True)
    v = j["hourly"]["shortwave_radiation"]  # W/m2
    dfh = pd.DataFrame({"time": t, "global_rad_W": v})
    # upsample a 15m con interpolazione lineare
    dfh = dfh.set_index("time").resample("15T").interpolate().reset_index()
    return dfh

def split_by_day_4(df):
    t0 = utc_floor_today() - timedelta(days=1)  # ieri
    days = [t0 + timedelta(days=i) for i in range(5)]  # 4 intervalli
    parts = []
    labels = ["Ieri", "Oggi", "Domani", "Dopodomani"]
    for i in range(4):
        dfp = df[(df["time"] >= days[i]) & (df["time"] < days[i+1])].copy()
        dfp["local_time"] = dfp["time"].dt.tz_convert(None)
        parts.append((labels[i], dfp))
    return parts

def plot_day_curves(parts):
    import altair as alt
    charts = []
    for label, d in parts:
        if d.empty:
            ch = alt.Chart(pd.DataFrame({"x":[], "y":[]})).mark_line().properties(width="container", height=180, title=f"{label} (nessun dato)")
        else:
            ch = alt.Chart(d).mark_line().encode(
                x=alt.X("local_time:T", title="Ora"),
                y=alt.Y("global_rad_W:Q", title="Global Rad (W/m²)")
            ).properties(width="container", height=180, title=label)
        charts.append(ch)
    return charts

def daily_kwh(df15):
    # integrazione energia relativa (assumendo rad in W/m2 -> indicatore; qui integriamo in kWh/m2)
    if df15.empty:
        return pd.Series(dtype=float)
    d = df15.copy()
    d["day"] = d["time"].dt.date
    # 15 minuti = 0.25 h
    return d.groupby("day")["global_rad_W"].sum() * 0.25 / 1000.0

# -------------------------
# Sezioni
# -------------------------
if section == "Storico":
    st.title("📊 Analisi Storica (dataset di esempio)")
    try:
        df = pd.read_csv("Dataset_Daily_EnergiaSeparata_2020_2025.csv", parse_dates=["date"])
        st.dataframe(df.head(20))
        st.line_chart(df.set_index("date")["E_INT_Daily_kWh"])
    except Exception as e:
        st.info("Carica un dataset valido 'Dataset_Daily_EnergiaSeparata_2020_2025.csv' nella root del progetto.")
        st.exception(e)

elif section == "Modello":
    st.title("🧠 Modello (placeholder)")
    st.write("Qui puoi collegare l'addestramento con i tuoi dati storici e salvare il modello.")

elif section == "Previsioni (15m) 4 giorni":
    st.title("🔮 Previsioni a 15 minuti - 4 giorni")
    t0 = utc_floor_today() - timedelta(days=1)    # ieri 00Z
    t4 = utc_floor_today() + timedelta(days=3, hours=23, minutes=59)  # +3 giorni quasi 4
    start_iso = t0.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso   = t4.strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        if source.startswith("Meteomatics"):
            df = meteomatics_df(lat, lon, start_iso, end_iso, "PT15M")
            st.success("✅ Dati Meteomatics ricevuti")
        else:
            df = openmeteo_df(lat, lon, days=4)
            st.warning("⚠️ Meteomatics non usato. Fonte: Open-Meteo (hourly → 15m interpolato).")
    except Exception as e:
        st.error(f"Errore chiamando {source}: {e}")
        st.stop()

    parts = split_by_day_4(df)
    charts = plot_day_curves(parts)

    cols = st.columns(2)
    cols[0].altair_chart(charts[0], use_container_width=True)  # ieri
    cols[1].altair_chart(charts[1], use_container_width=True)  # oggi
    cols = st.columns(2)
    cols[0].altair_chart(charts[2], use_container_width=True)  # domani
    cols[1].altair_chart(charts[3], use_container_width=True)  # dopodomani

    # export CSV
    csv1 = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Scarica curva 15-min (4 giorni)", csv1, "forecast_15min.csv", "text/csv")
    agg = daily_kwh(df.set_index("time").reset_index())
    st.write("Aggregato giornaliero (kWh/m²):")
    st.dataframe(agg.rename("kWh_m2"))
    csv2 = agg.to_csv().encode("utf-8")
    st.download_button("📥 Scarica aggregato giornaliero", csv2, "forecast_daily.csv", "text/csv")

elif section == "Mappa":
    st.title("🗺️ Mappa impianto (satellitare)")
    m = folium.Map(location=[lat, lon], zoom_start=17, tiles="Esri.WorldImagery")
    popup_html = f"""
    <div style="width:220px; padding:8px; font-size:13px; line-height:1.4; background:white; border-radius:8px; box-shadow:0 1px 6px rgba(0,0,0,.25)">
    <b>Impianto FV - ROBOTRONIX</b><br>
    📍 {lat:.6f}, {lon:.6f}<br>
    ⏱️ Previsioni: 15 min · 4 giorni<br>
    🔗 Fonte meteo: {source}
    </div>
    """
    folium.Marker([lat, lon], popup=folium.Popup(popup_html, max_width=250), tooltip="Info impianto").add_to(m)
    st_folium(m, width=None, height=600)
