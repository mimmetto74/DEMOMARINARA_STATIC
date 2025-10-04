# -*- coding: utf-8 -*-
import os, io, base64, joblib, requests
import numpy as np, pandas as pd, altair as alt, streamlit as st, folium
from datetime import datetime, timedelta, timezone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from streamlit_folium import st_folium

APP_TITLE = "Solar Forecast - ROBOTRONIX PROVA (V5)"
DATASET_PATHS = ["Dataset_Daily_EnergiaSeparata_2020_2025.csv", "/mnt/data/Dataset_Daily_EnergiaSeparata_2020_2025.csv"]
MODEL_PATH = "pv_model.joblib"

TARGET = "E_INT_Daily_kWh"
FEAT   = "G_M0_Wm2"

DEFAULT_LAT, DEFAULT_LON = 40.643278, 16.986083

st.set_page_config(page_title=APP_TITLE, page_icon="☀️", layout="wide")

# -------- Auth --------
if "auth" not in st.session_state: st.session_state["auth"] = False
if not st.session_state["auth"]:
    st.title(APP_TITLE)
    st.subheader("🔐 Accesso richiesto")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Entra"):
        if u.strip().upper() == "FVMANAGER" and p == "MIMMOFABIO":
            st.session_state["auth"] = True
            st.rerun()
        else:
            st.error("Credenziali non valide")
    st.stop()

# -------- Sidebar --------
st.sidebar.title("📋 Menu delle Impostazioni")
provider_pref = st.sidebar.selectbox("Provider meteo", ["Auto", "Meteomatics", "Open‑Meteo"])
lat = st.sidebar.number_input("Latitudine", value=DEFAULT_LAT, format="%.6f")
lon = st.sidebar.number_input("Longitudine", value=DEFAULT_LON, format="%.6f")
tilt = st.sidebar.slider("Tilt (°)", 0, 90, 0)
orient = st.sidebar.slider("Orientamento (°, 0=N, 90=E, 180=S, 270=W)", 0, 360, 180, 5)
plant_kw = st.sidebar.number_input("Potenza di targa (kW)", value=1000.0, step=50.0, min_value=0.0)
pr = st.sidebar.slider("Performance Ratio (PR)", 0.5, 0.95, 0.82, 0.01)
autosave = st.sidebar.toggle("Salva automaticamente CSV (curva + aggregato)", value=True)
irr_scale = st.sidebar.number_input("Fattore calibrazione irradiance → feature", value=1.0, step=0.05)

# -------- Helpers --------
def load_dataset():
    for p in DATASET_PATHS:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, parse_dates=["Date"])
                if "E_INT_Daily_KWh" in df.columns and TARGET not in df.columns:
                    df = df.rename(columns={"E_INT_Daily_KWh": TARGET})
                return df.sort_values("Date").copy()
            except Exception as e:
                st.warning(f"Errore leggendo {p}: {e}")
    st.error("Dataset non trovato. Inserire 'Dataset_Daily_EnergiaSeparata_2020_2025.csv' nella root.")
    return None

def split_train_test(df, cutoff="2025-01-01"):
    train = df[df["Date"] < cutoff].copy()
    test  = df[df["Date"] >= cutoff].copy()
    return train, test

def train_and_eval(df):
    d = df.dropna(subset=[FEAT, TARGET]).copy()
    if d.empty: return None, {}
    train, test = split_train_test(d)
    Xtr, ytr = train[[FEAT]].values, train[TARGET].values
    model = LinearRegression().fit(Xtr, ytr)
    joblib.dump(model, MODEL_PATH)
    metrics = {}
    if len(test) > 0:
        ypred = model.predict(test[[FEAT]].values)
        metrics["MAE_test"] = float(mean_absolute_error(test[TARGET].values, ypred))
        metrics["R2_test"]  = float(r2_score(test[TARGET].values, ypred))
    metrics["coef"] = float(model.coef_[0])
    metrics["intercept"] = float(model.intercept_)
    metrics["N_train"] = int(len(train)); metrics["N_test"] = int(len(test))
    return model, metrics

def load_model():
    if os.path.exists(MODEL_PATH):
        try: return joblib.load(MODEL_PATH)
        except Exception: return None
    return None

def meteomatics_fetch(lat, lon, start_iso, end_iso, step="PT15M"):
    user = os.environ.get("METEO_USER"); pwd = os.environ.get("METEO_PASS")
    if not user or not pwd: raise RuntimeError("Meteomatics: mancano METEO_USER/METEO_PASS")
    url = f"https://api.meteomatics.com/{start_iso}--{end_iso}:{step}/global_rad:W/{lat},{lon}/json"
    r = requests.get(url, auth=(user, pwd), timeout=30); r.raise_for_status()
    js = r.json()
    dates = js["data"][0]["coordinates"][0]["dates"]
    rows = [(pd.to_datetime(d["date"]), float(d["value"])) for d in dates]
    df = pd.DataFrame(rows, columns=["time", "GlobalRad_W"]).sort_values("time")
    return url, df

def openmeteo_fetch(lat, lon, start_date, end_date):
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&hourly=shortwave_radiation&start_date={start_date}&end_date={end_date}&timezone=UTC")
    r = requests.get(url, timeout=30); r.raise_for_status()
    j = r.json()
    t = pd.to_datetime(j["hourly"]["time"])
    v = j["hourly"].get("shortwave_radiation", [np.nan]*len(t))
    dfh = pd.DataFrame({"time": t, "GlobalRad_W": v}).sort_values("time")
    df15 = dfh.set_index("time").resample("15min").interpolate().reset_index()
    return url, df15

def apply_model_to_curve(df15, model, plant_kw, pr, irr_scale=1.0):
    if df15 is None or df15.empty:
        return None, 0.0, 0.0, 0.0
    d = df15.copy()
    d["GlobalRad_W"] = d["GlobalRad_W"].clip(lower=0)
    d["Wh_step"] = d["GlobalRad_W"] * 0.25
    total_feature = d["Wh_step"].sum() * irr_scale
    if model is not None and total_feature > 0:
        day_kwh_model = float(model.predict([[total_feature]])[0])
    else:
        day_kwh_m2 = d["Wh_step"].sum() / 1000.0
        day_kwh_model = float(day_kwh_m2 * plant_kw * pr)
    if d["Wh_step"].sum() > 0:
        d["kWh_curve"] = day_kwh_model * (d["Wh_step"] / d["Wh_step"].sum())
    else:
        d["kWh_curve"] = 0.0
    d["kW_inst"] = d["kWh_curve"] * 4.0
    peak_kW = float(d["kW_inst"].max())
    peak_pct = float((peak_kW / plant_kw) * 100.0) if plant_kw > 0 else 0.0
    return d, day_kwh_model, peak_kW, peak_pct

def download_button_bytes(label, data: bytes, filename: str, mime="text/csv"):
    st.download_button(label, data, file_name=filename, mime=mime)

# -------- Tabs --------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Storico", "🧠 Modello", "🔮 Previsioni 4 giorni (15m)", "🗺️ Mappa"])

with tab1:
    df = load_dataset()
    if df is not None:
        st.subheader("Storico – anteprima")
        st.dataframe(df.tail(300), use_container_width=True)
        st.subheader("Produzione storica (kWh)")
        st.line_chart(df.set_index("Date")[TARGET])

with tab2:
    st.subheader("Addestra modello su dati storici (con split train/test)")
    df = load_dataset()
    if df is not None:
        if st.button("Addestra / Riaddestra"):
            model, metrics = train_and_eval(df)
            if model is not None:
                st.session_state["model"] = model
                st.success(
                    f"Modello OK – Coef: {metrics.get('coef', float('nan')):.6f} | Intercetta: {metrics.get('intercept', float('nan')):.2f}  \n"
                    f"MAE test: {metrics.get('MAE_test', float('nan')):.1f} kWh | R² test: {metrics.get('R2_test', float('nan')):.3f}  \n"
                    f"N train: {metrics.get('N_train',0)} | N test: {metrics.get('N_test',0)}"
                )
            else:
                st.error("Addestramento fallito.")
        if os.path.exists(MODEL_PATH):
            st.info("Modello salvato su disco (`pv_model.joblib`). Puoi scaricarlo qui sotto.")
            with open(MODEL_PATH, "rb") as f:
                download_button_bytes("⬇️ Scarica modello (.joblib)", f.read(), "pv_model.joblib")
        try:
            d = df.dropna(subset=[FEAT, TARGET]).copy()
            train, test = split_train_test(d)
            m = st.session_state.get("model", None)
            if m is not None and len(test) > 0:
                test = test.copy()
                test["Pred"] = m.predict(test[[FEAT]].values)
                ch = alt.Chart(test).mark_line().encode(
                    x=alt.X("Date:T", title="Data"),
                    y=alt.Y(f"{TARGET}:Q", title="kWh"),
                    color=alt.value("#FFA500")
                ).properties(height=240, title="Storico vs Predetto (Test)")
                ch2 = alt.Chart(test).mark_line(color="#66CCFF").encode(x="Date:T", y="Pred:Q")
                st.altair_chart(ch + ch2, use_container_width=True)
        except Exception as e:
            st.caption(f"Grafico comparativo non disponibile: {e}")

with tab3:
    st.subheader("Previsioni a 15 minuti per 4 giorni (Ieri, Oggi, Domani, Dopodomani)")
    model = st.session_state.get("model", None)
    start_base = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    for label, offset in [("Ieri", -1), ("Oggi", 0), ("Domani", 1), ("Dopodomani", 2)]:
        day = start_base + timedelta(days=offset)
        start_iso = f"{day.strftime('%Y-%m-%d')}T00:00:00Z"
        end_iso   = f"{(day + timedelta(days=1)).strftime('%Y-%m-%d')}T00:00:00Z"
        st.markdown(f"### {label} – {day.date()}")
        try:
            if provider_pref == "Meteomatics" or provider_pref == "Auto":
                try:
                    url, df15 = meteomatics_fetch(lat, lon, start_iso, end_iso, step="PT15M")
                    provider_used = "Meteomatics"
                except Exception as e:
                    if provider_pref == "Meteomatics":
                        raise
                    url, df15 = openmeteo_fetch(lat, lon, str(day.date()), str((day + timedelta(days=1)).date()))
                    provider_used = "Open‑Meteo"
                    st.warning(f"Meteomatics non disponibile: {e}. Uso Open‑Meteo.")
            else:
                url, df15 = openmeteo_fetch(lat, lon, str(day.date()), str((day + timedelta(days=1)).date()))
                provider_used = "Open‑Meteo"
            st.caption(f"Provider: **{provider_used}**"); st.code(url, language="text")
        except Exception as e:
            st.error(f"Errore fetch previsioni: {e}")
            continue

        dfp, energy_kWh, peak_kW, peak_pct = apply_model_to_curve(df15, model, plant_kw, pr, irr_scale=irr_scale)
        if dfp is None or dfp.empty:
            st.warning("Nessun dato disponibile."); continue

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Energia stimata", f"{energy_kWh:.1f} kWh")
        c2.metric("Picco stimato", f"{peak_kW:.1f} kW")
        c3.metric("% della targa", f"{peak_pct:.1f}%")
        c4.metric("Punti curva", f"{len(dfp)}")

        st.line_chart(dfp.set_index("time")[["kWh_curve"]].rename(columns={"kWh_curve":"kWh/15min"}))

        curve_csv = dfp[["time","GlobalRad_W","kWh_curve","kW_inst"]].copy().to_csv(index=False).encode("utf-8")
        download_button_bytes(f"⬇️ Scarica curva 15-min ({label})", curve_csv, f"curva_{label.lower()}_15min.csv")
        daily_csv = pd.DataFrame([{"date": str(day.date()), "energy_kWh": energy_kWh, "peak_kW": peak_kW, "PR": pr}]).to_csv(index=False).encode("utf-8")
        download_button_bytes(f"⬇️ Scarica aggregato ({label})", daily_csv, f"daily_{label.lower()}.csv")

        if autosave:
            try:
                os.makedirs("logs", exist_ok=True)
                name_curve = f"logs/curve_{label.lower()}_{day.strftime('%Y%m%d')}.csv"
                name_daily = f"logs/daily_{label.lower()}_{day.strftime('%Y%m%d')}.csv"
                dfp.to_csv(name_curve, index=False)
                pd.DataFrame([{"date": str(day.date()), "energy_kWh": energy_kWh, "peak_kW": peak_kW, "PR": pr}]).to_csv(name_daily, index=False)
                st.caption(f"Salvati: {name_curve} · {name_daily}")
            except Exception as e:
                st.caption(f"Autosave non riuscito: {e}")

with tab4:
    st.subheader("Mappa impianto (satellitare)")
    popup_html = f"""
    <div style='background:#ffffffcc;padding:8px;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.25);font-size:13px'>
      <b>Impianto FV</b><br/>
      Lat: {lat:.6f} — Lon: {lon:.6f}<br/>
      Tilt: {tilt}° — Orient: {orient}°<br/>
      PR: {pr:.2f} — Potenza: {plant_kw:.0f} kW
    </div>
    """
    m = folium.Map(location=[lat, lon], tiles="Esri.WorldImagery", zoom_start=16, control_scale=True)
    folium.Marker([lat,lon], tooltip="Impianto FV", popup=folium.Popup(popup_html, max_width=280)).add_to(m)
    st_folium(m, width=None, height=560)
