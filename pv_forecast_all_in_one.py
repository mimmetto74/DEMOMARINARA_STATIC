
# -*- coding: utf-8 -*-
# PV Forecast - Monitor Ready (Railway-friendly)

import os
import io
import json
import math
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

import streamlit as st
import plotly.graph_objects as go

# ---------------- Safe XGBoost import with RF fallback ---------------- #
try:
    from xgboost import XGBRegressor
    XGB_OK = True
except Exception:
    from sklearn.ensemble import RandomForestRegressor as XGBRegressor
    XGB_OK = False
# --------------------------------------------------------------------- #

# ---------------- Logging ---------------- #
def write_log(msg: str):
    os.makedirs("logs", exist_ok=True)
    with open("logs/errors.log", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")

# ---------------- Model load/save ---------------- #
try:
    import joblib
except Exception:
    import pickle as joblib  # fallback

MODEL_PATHS = ["logs/model_xgb.joblib", "model_xgb.joblib", "rf_model.joblib"]
MODEL_SOURCE = None

def load_model():
    global MODEL_SOURCE
    for p in MODEL_PATHS:
        if os.path.exists(p):
            try:
                m = joblib.load(p)
                MODEL_SOURCE = p
                return m
            except Exception as e:
                write_log(f"Failed to load {p}: {e}")
    MODEL_SOURCE = "NEW"
    return XGBRegressor()

def save_model(model, path="logs/model_xgb.joblib"):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
    except Exception as e:
        write_log(f"Failed to save model: {e}")
# ------------------------------------------------- #

# ---------------- Weather (Open-Meteo) ---------------- #
def fetch_open_meteo(lat, lon, start_date, end_date, tz="Europe/Rome"):
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,cloudcover,global_tilted_irradiance"
        f"&timezone={tz.replace('/', '%2F')}"
        f"&start_date={start_date}&end_date={end_date}"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame({
        "datetime": pd.to_datetime(data["hourly"]["time"]),
        "Temp_Air": data["hourly"]["temperature_2m"],
        "CloudCover_P": data["hourly"]["cloudcover"],
        "G_M0_Wm2": data["hourly"]["global_tilted_irradiance"],
    })
    return df
# ------------------------------------------------------ #

# ---------------- Forecast computation ---------------- #
def compute_forecast_for_day(date_obj, model, lat, lon, plant_kw):
    start = date_obj.strftime("%Y-%m-%d")
    end = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        df = fetch_open_meteo(lat, lon, start, end)
    except Exception as e:
        write_log(f"Open-Meteo error: {e}")
        raise

    # Features
    feats = ["G_M0_Wm2", "CloudCover_P", "Temp_Air"]
    for c in feats:
        if c not in df.columns:
            df[c] = 0.0
    X = df[feats].values.astype("float32")

    # Predict power (kW). If model is unfitted, fit a trivial baseline.
    try:
        y = model.predict(X)
    except Exception:
        # quick baseline fit to avoid NotFittedError
        from sklearn.ensemble import RandomForestRegressor
        baseline = RandomForestRegressor(n_estimators=50, random_state=42)
        # fabricate a minimal target near irradiance scaled
        y_fake = (df["G_M0_Wm2"].values / max(df["G_M0_Wm2"].max(), 1.0)) * plant_kw
        baseline.fit(X, y_fake)
        y = baseline.predict(X)

    # scale to plant size if needed (already roughly scaled via model)
    df["Produzione_stimata_kW"] = np.clip(y, 0, None)
    return df

# ---------------- Utility: integrate absolute difference ---------------- #
def integrate_abs_difference(df_new, df_ref, col="Produzione_stimata_kW"):
    m = pd.merge(df_new[["datetime", col]].copy(), df_ref[["datetime", col]].copy(),
                 on="datetime", suffixes=("_new", "_ref"))
    m = m.sort_values("datetime")
    # compute absolute diff
    diff = (m[f"{col}_new"] - m[f"{col}_ref"]).abs().values
    # compute dt in hours from timestamps (handle 15-min)
    if len(m) >= 2:
        dt_hours = (m["datetime"].diff().dt.total_seconds().fillna(0).replace(0, np.nan).median() or 3600.0) / 3600.0
    else:
        dt_hours = 1.0
    integral = float(np.trapz(diff, dx=dt_hours))
    return integral, m

# ---------------- Streamlit App ---------------- #
st.set_page_config(page_title="PV Forecast - Monitor Ready", layout="wide")
st.title("🔆 PV Forecast — Monitor & Save (Railway-ready)")

with st.sidebar:
    st.header("⚙️ Parametri")
    lat = st.number_input("Latitudine", value=40.85, format="%.6f")
    lon = st.number_input("Longitudine", value=17.130000, format="%.6f")
    plant_kw = st.number_input("Potenza impianto (kW)", value=50.0, min_value=1.0, step=1.0)
    st.caption("Se XGBoost non è disponibile sull'hosting, userà automaticamente RandomForest.")

# Load model and show source
model = load_model()
if MODEL_SOURCE == "NEW":
    st.info("⚙️ Nessun modello trovato — inizializzato uno nuovo")
else:
    st.success(f"🧠 Modello caricato da: {MODEL_SOURCE}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📥 Dati", "🧠 Modello", "🔮 Previsioni", "📤 Export", "📊 Monitoraggio"])

with tab2:
    st.subheader("🧠 Modello")
    st.write("Qui puoi *(opzionale)* ri-addestrare il modello veloce di baseline con un CSV storico.\n"
             "In assenza di dataset, verrà usata una logica base legata all'irradianza.")
    uploaded = st.file_uploader("Carica CSV storico (opzionale)", type=["csv"])
    if uploaded is not None:
        try:
            dfh = pd.read_csv(uploaded)
            cols = ["G_M0_Wm2", "CloudCover_P", "Temp_Air", "Produzione_stimata_kW"]
            if not all(c in dfh.columns for c in cols):
                st.warning(f"Nel CSV devono esserci le colonne: {cols}")
            else:
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import mean_absolute_error, r2_score
                X = dfh[["G_M0_Wm2", "CloudCover_P", "Temp_Air"]].values
                y = dfh["Produzione_stimata_kW"].values
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
                try:
                    model.fit(Xtr, ytr)
                    save_model(model)
                    st.success("💾 Modello salvato correttamente dopo l’addestramento!")
                    pred = model.predict(Xte)
                    st.metric("MAE (kW)", f"{mean_absolute_error(yte, pred):.2f}")
                    st.metric("R²", f"{r2_score(yte, pred):.3f}")
                except Exception as e:
                    write_log(f"Train error: {e}")
                    st.error(f"Errore training: {e}")
        except Exception as e:
            write_log(f"CSV read error: {e}")
            st.error(f"Errore lettura CSV: {e}")

with tab3:
    st.subheader("🔮 Previsioni")
    tz = "Europe/Rome"
    today = datetime.now().date()
    days = {
        "Ieri": today - timedelta(days=1),
        "Oggi": today,
        "Domani": today + timedelta(days=1),
        "Dopodomani": today + timedelta(days=2),
    }

    cols = st.columns(4)
    forecasts = {}
    for i, (label, d) in enumerate(days.items()):
        with cols[i]:
            try:
                df = compute_forecast_for_day(d, model, lat, lon, plant_kw)
                forecasts[label] = df
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["datetime"], y=df["G_M0_Wm2"], name="🌞 Irradianza (W/m²)", mode="lines"))
                fig.add_trace(go.Scatter(x=df["datetime"], y=df["Produzione_stimata_kW"], name="⚡ Produzione (kW)", mode="lines"))
                fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), title=label)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                write_log(f"Forecast {label} error: {e}")
                st.error(f"Errore previsione {label}: {e}")

    # Save "Domani" automatically
    if "Domani" in forecasts:
        try:
            df_tom = forecasts["Domani"].copy()
            os.makedirs("logs/forecasts", exist_ok=True)
            fname = datetime.now().strftime("%Y-%m-%d_forecast_tomorrow.csv")
            path = os.path.join("logs/forecasts", fname)
            df_tom.to_csv(path, index=False)
            st.success(f"💾 Previsione di domani salvata in {path}")
        except Exception as e:
            write_log(f"Save tomorrow error: {e}")
            st.error(f"Errore salvataggio previsione 'Domani': {e}")

with tab5:
    st.subheader("📊 Monitoraggio variazione previsioni")
    # Optional autorefresh every 15 minutes
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=15 * 60 * 1000, key="forecast_refresh")
        st.caption("⏱️ Aggiornamento automatico attivo (15 minuti)")
    except Exception:
        st.caption("⏱️ Autorefresh non disponibile: aggiorna manualmente la pagina")

    # Load latest saved "tomorrow" curve
    import glob
    files = sorted(glob.glob(os.path.join("logs/forecasts", "*_forecast_tomorrow.csv")))
    if not files:
        st.warning("⚙️ Nessuna previsione 'Domani' salvata disponibile. Vai sul Tab Previsioni per generarla.")
    else:
        last_file = files[-1]
        try:
            df_ref = pd.read_csv(last_file, parse_dates=["datetime"])
            st.info(f"🧠 Curva di riferimento: {os.path.basename(last_file)}")
        except Exception as e:
            write_log(f"Read saved curve error: {e}")
            st.error(f"Errore lettura curva salvata: {e}")
            df_ref = None

        if df_ref is not None:
            try:
                # Compute a fresh forecast for "tomorrow" (same target day as reference file name date or relative)
                target_day = datetime.now().date() + timedelta(days=1)
                df_new = compute_forecast_for_day(target_day, model, lat, lon, plant_kw)
                # Ensure datetime type
                df_new["datetime"] = pd.to_datetime(df_new["datetime"])
                df_ref["datetime"] = pd.to_datetime(df_ref["datetime"])

                integral_diff, merged = integrate_abs_difference(df_new, df_ref, "Produzione_stimata_kW")

                st.metric(label="📈 Indice variazione curva (integrale |Δ|)", value=f"{integral_diff:.2f} (kW·h)")

                # Alert box with user slider
                threshold = st.slider("Soglia di allerta (kW·h)", 10.0, 500.0, 100.0, step=10.0)
                if integral_diff > threshold:
                    st.error(f"🚨 ATTENZIONE: variazione elevata – indice {integral_diff:.2f} > soglia {threshold:.2f}")
                elif integral_diff > threshold * 0.5:
                    st.warning(f"⚠️ Variazione moderata – indice {integral_diff:.2f}")
                else:
                    st.success(f"✅ Variazione contenuta – indice {integral_diff:.2f}")

                # Comparison chart
                chart_df = pd.DataFrame({
                    "datetime": merged["datetime"],
                    "Previsione corrente": merged["Produzione_stimata_kW_new"],
                    "Curva 'Domani' salvata": merged["Produzione_stimata_kW_ref"]
                }).set_index("datetime")
                st.line_chart(chart_df)
            except Exception as e:
                write_log(f"Monitor error: {e}")
                st.error(f"Errore monitoraggio variazioni: {e}")

with tab4:
    st.subheader("📤 Esporta")
    st.caption("Scarica l'ultima previsione 'Domani' salvata")
    try:
        import glob
        files = sorted(glob.glob(os.path.join("logs/forecasts", "*_forecast_tomorrow.csv")))
        if files:
            last = files[-1]
            df_last = pd.read_csv(last)
            st.download_button("⬇️ Scarica CSV 'Domani' più recente", data=df_last.to_csv(index=False), file_name=os.path.basename(last), mime="text/csv")
        else:
            st.info("Nessun file salvato ancora.")
    except Exception as e:
        write_log(f"Export error: {e}")
        st.error(f"Errore export: {e}")
