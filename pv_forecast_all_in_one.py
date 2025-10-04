
# -*- coding: utf-8 -*-
import os
import io
import json
import base64
import folium
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit as st
from streamlit.components.v1 import html as st_html

APP_TITLE = "Solar Forecast - ROBOTRONIX for IMEPOWER"

# -----------------------
# Login semplice
# -----------------------
LOGIN_USER = "FVMANAGER"
LOGIN_PASS = "MIMMOFABIO"

# -----------------------
# Meteomatics credenziali (richiesta dall'utente)
# -----------------------
METEO_USER = "teseospa-eiffageenergiesystemesitaly_daniello_fabio"
METEO_PASS = "6S8KTHPbrUlp6523T9Xd"

DATASET_PATH = "Dataset_Daily_EnergiaSeparata_2020_2025.csv"

st.set_page_config(page_title=APP_TITLE, page_icon="☀️", layout="wide")

# ----- Sidebar: login -----
with st.sidebar:
    st.markdown("### Menu delle impostazioni")
    if "auth" not in st.session_state:
        st.session_state.auth = False

    if not st.session_state.auth:
        u = st.text_input("Username", "")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if u.strip().upper()==LOGIN_USER and p==LOGIN_PASS:
                st.session_state.auth = True
                st.success("Accesso eseguito")
                st.experimental_rerun()
            else:
                st.error("Credenziali non valide")
    else:
        st.success("Autenticato come **{}**".format(LOGIN_USER))
        if st.button("Logout"):
            st.session_state.auth = False
            st.experimental_rerun()

if not st.session_state.auth:
    st.stop()

st.title(APP_TITLE)

# Tabs in alto: come nella v4
tab_storico, tab_modello, tab_previsioni, tab_mappa = st.tabs(["📊 Storico", "🧪 Modello", "⏲️ Previsioni 4 giorni (15m)", "🗺️ Mappa"])

# -----------------------
# Utility
# -----------------------
def load_dataset(path=DATASET_PATH):
    if not os.path.exists(path):
        st.warning("Dataset non trovato: {}".format(path))
        return None
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date")
    return df

def train_model(df):
    # modello lineare E_INT_Daily_kWh = a * G_M0_Wm2 + b
    data = df.dropna(subset=["E_INT_Daily_kWh","G_M0_Wm2"])
    X = data[["G_M0_Wm2"]].values
    y = data["E_INT_Daily_kWh"].values
    model = LinearRegression()
    model.fit(X, y)
    return model

def try_meteomatics(lat, lon, start_dt, end_dt, step="PT15M"):
    url = f"https://api.meteomatics.com/{start_dt}--{end_dt}:{step}/global_rad:W,total_cloud_cover:p/{lat},{lon}/json"
    try:
        r = requests.get(url, auth=(METEO_USER, METEO_PASS), timeout=30)
        r.raise_for_status()
        j = r.json()
        # parse json
        rows = []
        for par in j.get("data", []):
            pname = par.get("parameter")
            for loc in par.get("coordinates", []):
                for t in loc.get("dates", []):
                    ts = t.get("date")
                    val = t.get("value")
                    rows.append((ts, pname, val))
        df = pd.DataFrame(rows, columns=["time","param","value"])
        if df.empty:
            return None, "Empty Meteomatics response"
        df["time"] = pd.to_datetime(df["time"])
        # pivot
        df = df.pivot_table(index="time", columns="param", values="value").reset_index()
        df = df.sort_values("time")
        df.rename(columns={"global_rad:W":"global_rad_W","total_cloud_cover:p":"cloud_p"}, inplace=True)
        return df, None
    except Exception as e:
        return None, str(e)

def try_openmeteo(lat, lon, start_dt, end_dt):
    # Open-Meteo hourly -> resample 15m
    # we use shortwave_radiation as proxy of global radiation
    try:
        base = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "shortwave_radiation,cloud_cover",
            "timezone": "UTC"
        }
        r = requests.get(base, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        times = pd.to_datetime(j["hourly"]["time"])
        sw = j["hourly"]["shortwave_radiation"]
        cc = j["hourly"]["cloud_cover"]
        df = pd.DataFrame({"time":times, "global_rad_W":np.asarray(sw)*1000/1.0, "cloud_p":cc})  # W/m2 approx
        # Filter by range
        mask = (df["time"]>=pd.to_datetime(start_dt))&(df["time"]<=pd.to_datetime(end_dt))
        df = df.loc[mask].copy()
        if df.empty:
            return None, "Open-Meteo returned no data in range"
        df = df.set_index("time").resample("15min").interpolate().reset_index()
        return df, None
    except Exception as e:
        return None, str(e)

def forecast_curve_15m(model, df_irr_15m):
    # semplice conversione lineare: kWh_15m = (a*irr + b/96) * scale
    # dove irr = W (istantaneo). Approssimazione dimostrativa.
    a = model.coef_[0]
    b = model.intercept_
    out = df_irr_15m.copy()
    out["kWh_15m"] = np.maximum(0, a * (out["global_rad_W"]/1000.0) * 0.25 + (b/96.0)*0)  # 0.25h
    return out

def daily_from_curve(df15):
    d = df15.set_index("time")["kWh_15m"].resample("D").sum()
    return d

def download_link(df, filename):
    csv = df.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    return f'<a href="data:text/csv;base64,{b64}" download="{filename}">📥 Scarica {filename}</a>'

# -----------------------
# STORICO
# -----------------------
with tab_storico:
    st.subheader("Analisi Storica")
    df = load_dataset()
    if df is not None:
        st.dataframe(df.tail(20), use_container_width=True)
        st.line_chart(df.set_index("Date")[["E_INT_Daily_kWh","G_M0_Wm2"]])

# -----------------------
# MODELLO
# -----------------------
with tab_modello:
    st.subheader("Addestramento semplice (lineare)")
    df = load_dataset()
    if df is None:
        st.stop()
    model = train_model(df)
    y_pred = model.predict(df[["G_M0_Wm2"]])
    mae = mean_absolute_error(df["E_INT_Daily_kWh"], y_pred)
    r2 = r2_score(df["E_INT_Daily_kWh"], y_pred)
    st.write(f"**MAE**: {mae:.2f} kWh  |  **R²**: {r2:.3f}")
    st.info(f"Formula: E ≈ {model.coef_[0]:.5f} * G_M0_Wm2 + {model.intercept_:.2f}")

# -----------------------
# PREVISIONI
# -----------------------
with tab_previsioni:
    st.subheader("Previsioni a 15 minuti per 4 giorni (ieri, oggi, domani, dopodomani)")
    col = st.columns(4)
    with col[0]:
        lat = st.number_input("Latitudine", value=40.643278, format="%.6f")
    with col[1]:
        lon = st.number_input("Longitudine", value=16.986083, format="%.6f")
    with col[2]:
        kwp = st.number_input("Potenza nominale (kWp)", value=300.0, min_value=1.0, step=10.0)
    with col[3]:
        force_provider = st.selectbox("Sorgente preferita", ["Auto (Meteomatics→Open-Meteo)", "Meteomatics", "Open-Meteo"])

    df = load_dataset()
    if df is None:
        st.stop()
    model = train_model(df)

    # 4 giorni (ieri..+2)
    base_date = datetime.utcnow().date()
    days = [
        ("Ieri", base_date - timedelta(days=1)),
        ("Oggi", base_date),
        ("Domani", base_date + timedelta(days=1)),
        ("Dopodomani", base_date + timedelta(days=2)),
    ]

    for label, d0 in days:
        st.markdown(f"### {label} – {d0.isoformat()}")
        start = datetime(d0.year,d0.month,d0.day,tzinfo=timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z")
        end = (datetime(d0.year,d0.month,d0.day,tzinfo=timezone.utc)+timedelta(days=1)-timedelta(seconds=1)).isoformat(timespec="seconds").replace("+00:00","Z")

        if force_provider == "Meteomatics":
            src = "meteomatics"
        elif force_provider == "Open-Meteo":
            src = "openmeteo"
        else:
            src = "auto"

        df_irr = None
        msg = None
        if src in ("meteomatics","auto"):
            df_irr, msg = try_meteomatics(lat, lon, start, end, step="PT15M")
        if df_irr is None and src in ("openmeteo","auto"):
            df_irr, msg = try_openmeteo(lat, lon, start, end)

        if df_irr is None:
            st.error(f"Errore fetch previsioni: {msg}")
            continue

        curve = forecast_curve_15m(model, df_irr)
        daily = daily_from_curve(curve)

        # Peak power estimate
        peak_kw = curve["kWh_15m"].max() / 0.25  # kW in quel quarto d'ora
        peak_pct = 100.0 * peak_kw / kwp

        st.caption(f"Sorgente dati: {'Meteomatics' if 'cloud_p' in curve.columns else 'Open-Meteo'} | Peak ~ {peak_kw:.1f} kW ({peak_pct:.0f}% della targa)")
        st.line_chart(curve.set_index("time")[["kWh_15m"]])

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Energia giornaliera prevista [kWh]", f"{daily.sum():.1f}")
        with c2:
            st.markdown(download_link(curve[["time","kWh_15m"]], f"curva_15m_{d0.isoformat()}.csv"), unsafe_allow_html=True)

# -----------------------
# MAPPA
# -----------------------
with tab_mappa:
    st.subheader("Mappa impianto (satellitare)")
    lat = st.number_input("Latitudine mappa", value=40.643278, format="%.6f")
    lon = st.number_input("Longitudine mappa", value=16.986083, format="%.6f")
    m = folium.Map(location=[lat, lon], zoom_start=17, tiles="Esri.WorldImagery")
    popup_html = f"""
    <div style='font-size:13px;line-height:1.3'>
      <b>Impianto FV</b><br>
      Lat: {lat:.6f}, Lon: {lon:.6f}<br>
      Layer: Esri World Imagery
    </div>
    """
    folium.Marker([lat, lon], popup=popup_html, tooltip="Impianto").add_to(m)
    st_html(m._repr_html_(), height=600)
