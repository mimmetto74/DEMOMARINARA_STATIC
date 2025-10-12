import os, io, requests, joblib
import pandas as pd, numpy as np
from datetime import datetime, timedelta, timezone
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import folium
from streamlit_folium import st_folium

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
DATA_PATH = "Dataset_Daily_EnergiaSeparata_2020_2025.csv"
MODEL_PATH = "pv_model.joblib"
LOG_PATH = "forecast_log.csv"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

DEFAULT_LAT = 40.643278
DEFAULT_LON = 16.986083

if "tilt" not in st.session_state: st.session_state["tilt"] = 0
if "orient" not in st.session_state: st.session_state["orient"] = 180

# Meteomatics creds (in-code per richiesta)
MM_USER = "teseospa-eiffageenergiesystemesitaly_daniello_fabio"
MM_PASS = "6S8KTHPbrUlp6523T9Xd"

def ensure_log_file():
    if not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0:
        pd.DataFrame(columns=[
            "timestamp","day_label","provider","status","url",
            "lat","lon","tilt","orient","sum_rad_corr","pred_kwh","peak_kW","cloud_mean","note"
        ]).to_csv(LOG_PATH, index=False)

def write_log(**row):
    ensure_log_file()
    try:
        df = pd.read_csv(LOG_PATH)
    except Exception:
        df = pd.DataFrame(columns=[
            "timestamp","day_label","provider","status","url",
            "lat","lon","tilt","orient","sum_rad_corr","pred_kwh","peak_kW","cloud_mean","note"
        ])
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)

@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["Date"])

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def train_model():
    df0 = load_data()

    # Corregge nomi colonne
    if "E_INT_Daily_KWh" in df0.columns and "E_INT_Daily_kWh" not in df0.columns:
        df0 = df0.rename(columns={"E_INT_Daily_KWh": "E_INT_Daily_kWh"})

    # Se mancano le nuove feature, le crea con NaN
    for col in ["CloudCover_P", "Temp_Air"]:
        if col not in df0.columns:
            df0[col] = np.nan

    # Rimuove righe incomplete
    df = df0.dropna(subset=["E_INT_Daily_kWh", "G_M0_Wm2"]).copy()

    # Feature set multivariato
    X = df[["G_M0_Wm2", "CloudCover_P", "Temp_Air"]].fillna(df[["G_M0_Wm2", "CloudCover_P", "Temp_Air"]].mean())
    y = df["E_INT_Daily_kWh"]

    # Suddivisione train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    # Modello Random Forest
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Valutazione
    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    # Salva modello
    joblib.dump(model, MODEL_PATH)

    # Importanza feature
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    feature_importances.to_csv("feature_importances.csv", index=True)

    return mae, r2, feature_importances, model


def load_model():
    if not os.path.exists(MODEL_PATH):
        train_model()
    return joblib.load(MODEL_PATH)

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_meteomatics_pt15m(lat, lon, start_iso, end_iso, tilt=None, orient=None):
    if tilt is not None and orient is not None and tilt > 0:
        rad_param = f"global_rad_tilt_{int(round(tilt))}_orientation_{int(round(orient))}:W"
    else:
        rad_param = "global_rad:W"
    url = (f"https://api.meteomatics.com/"
           f"{start_iso}--{end_iso}:PT15M/"
           f"{rad_param},total_cloud_cover:p/"
           f"{lat},{lon}/json")
    r = requests.get(url, auth=(MM_USER, MM_PASS), timeout=30)
    r.raise_for_status()
    j = r.json()
    rows = []
    for blk in j.get("data", []):
        prm = blk.get("parameter")
        if prm.endswith(":W"): prm = "GlobalRad_W"
        if prm == "total_cloud_cover:p": prm = "CloudCover_P"
        for d in blk["coordinates"][0]["dates"]:
            rows.append({"time": d["date"], prm: d["value"]})
    df = pd.DataFrame(rows)
    if df.empty: return url, df
    df = df.groupby("time", as_index=False).mean().sort_values("time")
    if "GlobalRad_W" not in df.columns: df["GlobalRad_W"] = np.nan
    if "CloudCover_P" not in df.columns: df["CloudCover_P"] = np.nan
    df["time"] = pd.to_datetime(df["time"])
    df["fonte"] = "Meteomatics"
    return url, df

def fetch_openmeteo_hourly(lat, lon, start_date, end_date):
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&hourly=direct_radiation,cloudcover&start_date={start_date}&end_date={end_date}")
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    j = r.json()
    hh = j.get("hourly", {})
    times = hh.get("time", [])
    rad   = hh.get("direct_radiation", [np.nan]*len(times))
    cld   = hh.get("cloudcover", [np.nan]*len(times))
    df = pd.DataFrame({"time": times, "GlobalRad_W": rad, "CloudCover_P": cld})
    df["time"] = pd.to_datetime(df["time"])
    df["fonte"] = "Open-Meteo"
    df = df.set_index("time").resample("15min").interpolate(method="time").reset_index()
    return url, df

def compute_curve_and_daily(df, model, plant_kw):
    if df is None or df.empty:
        return None, 0.0, 0.0, 0.0, float("nan")
    df = df.copy().sort_values("time")
    df["GlobalRad_W"] = df["GlobalRad_W"].clip(lower=0)
    df["CloudCover_P"] = df["CloudCover_P"].clip(lower=0, upper=100)
    df["rad_corr"] = df["GlobalRad_W"] * (1 - df["CloudCover_P"]/100.0)
    sum_rad = df["rad_corr"].sum()
    pred_kwh = float(model.predict([[sum_rad]])[0]) if sum_rad > 0 else 0.0
    if sum_rad > 0:
        df["kWh_curve"] = pred_kwh * (df["rad_corr"]/sum_rad)
    else:
        df["kWh_curve"] = 0.0
    df["kW_inst"] = df["kWh_curve"] * 4.0
    peak_kW = float(df["kW_inst"].max()) if len(df) else 0.0
    peak_pct = float(peak_kW/plant_kw*100.0) if plant_kw > 0 else 0.0
    cloud_mean = float(df["CloudCover_P"].mean()) if "CloudCover_P" in df.columns else float("nan")
    return df, pred_kwh, peak_kW, peak_pct, cloud_mean

def forecast_for_day(lat, lon, offset_days, label, model, tilt, orient, provider_pref, plant_kw, autosave=True):
    day = (datetime.now(timezone.utc).date() + timedelta(days=offset_days))
    start_iso = f"{day}T00:00:00Z"; end_iso = f"{day + timedelta(days=1)}T00:00:00Z"
    provider = "Meteomatics"; status = "OK"; url = ""; df = None

    def try_meteomatics():
        nonlocal url, df, provider, status
        try:
            url, df = fetch_meteomatics_pt15m(lat, lon, start_iso, end_iso, tilt=tilt, orient=orient)
            provider, status = "Meteomatics", "OK"
        except Exception as e:
            provider, status, url = "Meteomatics", f"ERROR: {e}", ""
            df = None

    def try_openmeteo():
        nonlocal url, df, provider, status
        try:
            url, df = fetch_openmeteo_hourly(lat, lon, str(day), str(day + timedelta(days=1)))
            provider, status = "Open-Meteo", "OK"
        except Exception as e:
            provider, status, url = "Open-Meteo", f"ERROR: {e}", ""
            df = None

    if provider_pref == "Meteomatics":
        try_meteomatics()
        if df is None: try_openmeteo()
    elif provider_pref == "Open-Meteo":
        try_openmeteo()
    else:  # Auto
        try_meteomatics()
        if df is None: try_openmeteo()

    if df is None or df.empty:
        write_log(timestamp=datetime.utcnow().isoformat(), day_label=label,
                  provider=provider, status=status, url=url, lat=lat, lon=lon,
                  tilt=tilt, orient=orient, sum_rad_corr="", pred_kwh="", peak_kW="", cloud_mean="", note="no data")
        return None, 0.0, 0.0, 0.0, float("nan"), provider, status, url

    df2, pred_kwh, peak_kW, peak_pct, cloud_mean = compute_curve_and_daily(df, model, plant_kw)
    write_log(timestamp=datetime.utcnow().isoformat(), day_label=label,
              provider=provider, status=status, url=url, lat=lat, lon=lon,
              tilt=tilt, orient=orient, sum_rad_corr=float(df2["rad_corr"].sum()), pred_kwh=float(pred_kwh),
              peak_kW=float(peak_kW), cloud_mean=float(cloud_mean), note="")

    if autosave:
        # salva CSV curva 15min e aggregato giorno
        curve_csv = df2[["time","GlobalRad_W","CloudCover_P","rad_corr","kWh_curve","kW_inst"]].copy()
        curve_csv.to_csv(os.path.join(LOG_DIR, f"curve_{label.lower()}_15min.csv"), index=False)
        agg_csv = pd.DataFrame([{
            "date": str(day),
            "energy_kWh": float(pred_kwh),
            "peak_kW": float(peak_kW),
            "cloud_mean": float(cloud_mean)
        }])
        agg_csv.to_csv(os.path.join(LOG_DIR, f"daily_{label.lower()}.csv"), index=False)

    return df2, pred_kwh, peak_kW, peak_pct, cloud_mean, provider, status, url

# -------- UI ---------
st.title("☀️ Solar Forecast - ROBOTRONIX for IMEPOWER")

ensure_log_file()
st.sidebar.header("Impostazioni")
provider_pref = st.sidebar.selectbox("Fonte meteo:", ["Auto","Meteomatics","Open-Meteo"])
plant_kw = st.sidebar.number_input("Potenza di targa impianto (kW)", value=1000.0, step=50.0, min_value=0.0)
autosave = st.sidebar.toggle("Salvataggio automatico CSV (curva + aggregato)", value=True)

st.sidebar.header("📍 Posizione & Piano")
lat_sidebar = st.sidebar.number_input("Latitudine", value=DEFAULT_LAT, format="%.6f")
lon_sidebar = st.sidebar.number_input("Longitudine", value=DEFAULT_LON, format="%.6f")
st.session_state["tilt"] = st.sidebar.slider("Tilt (°)", min_value=0, max_value=90, value=st.session_state["tilt"], step=1)
st.session_state["orient"] = st.sidebar.slider("Orientation (°, 0=N, 90=E, 180=S, 270=W)", min_value=0, max_value=360, value=st.session_state["orient"], step=5)

st.sidebar.header("📥 Log Previsioni")
log_df = pd.read_csv(LOG_PATH)
flt = st.sidebar.selectbox("Filtro log", ["Tutti","Solo Meteomatics","Solo Open‑Meteo","Solo Errori"])
ldf = log_df.copy()
if flt=="Solo Meteomatics":
    ldf = ldf[ldf["provider"]=="Meteomatics"]
elif flt=="Solo Open‑Meteo":
    ldf = ldf[ldf["provider"]=="Open-Meteo"]
elif flt=="Solo Errori":
    ldf = ldf[ldf["status"].astype(str).str.startswith("ERROR", na=False)]
st.sidebar.write(f"Righe: {len(ldf)}")
csv_io = io.StringIO(); ldf.to_csv(csv_io, index=False)
st.sidebar.download_button("⬇️ Scarica log filtrato", csv_io.getvalue(), "forecast_log_filtered.csv", "text/csv")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Storico","🛠️ Modello","🔮 Previsioni 4 giorni (15 min)","🗺️ Mappa"])

import plotly.graph_objects as go

with tab1:
    try:
        df = load_data()
        st.subheader("Storico produzione e irradianza separati")

        # Corregge eventuale differenza nel nome colonna
        if "E_INT_Daily_KWh" in df.columns and "E_INT_Daily_kWh" not in df.columns:
            df = df.rename(columns={"E_INT_Daily_KWh": "E_INT_Daily_kWh"})

        # --- Grafico 1: Produzione (kWh) ---
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df["Date"], y=df["E_INT_Daily_kWh"],
            mode='lines',  # solo linee
            name='Produzione (kWh)',
            line=dict(color='orange', width=2)
        ))
        fig1.update_layout(
            title="⚡ Produzione giornaliera (kWh)",
            xaxis_title="Data",
            yaxis_title="Energia (kWh)",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)

        # --- Grafico 2: Irradianza (W/m²) ---
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["Date"], y=df["G_M0_Wm2"],
            mode='lines',
            name='Irradianza (W/m²)',
            line=dict(color='deepskyblue', width=2)
        ))
        fig2.update_layout(
            title="☀️ Irradianza giornaliera (W/m²)",
            xaxis_title="Data",
            yaxis_title="W/m²",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"Impossibile caricare dataset: {e}")


import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, r2_score

with tab2:
    st.subheader("🧠 Modello di previsione – Random Forest")

    c1, c2 = st.columns([1, 3])

    # Pulsante per addestrare il modello
    if c1.button("Addestra / Riaddestra modello", use_container_width=True):
        mae, r2, _, _ = train_model()
        st.session_state["last_mae"] = mae
        st.session_state["last_r2"] = r2
        st.success(f"✅ Modello addestrato con successo!\n\n**MAE:** {mae:.2f} | **R²:** {r2:.3f}")

    # Se il modello esiste, mostriamo grafico
    if os.path.exists(MODEL_PATH):
        model = load_model()
        df = load_data()
        if "E_INT_Daily_KWh" in df.columns and "E_INT_Daily_kWh" not in df.columns:
            df = df.rename(columns={"E_INT_Daily_KWh": "E_INT_Daily_kWh"})
        df = df.dropna(subset=["E_INT_Daily_kWh", "G_M0_Wm2"])

        # Predizione
             # --- Predizione con allineamento feature ---
        # Assicura che le colonne esistano
        for col in ["CloudCover_P", "Temp_Air"]:
            if col not in df.columns:
                df[col] = np.nan

        # Prepara le feature con riempimento dei NaN
        X_pred = df[["G_M0_Wm2", "CloudCover_P", "Temp_Air"]].fillna(
            df[["G_M0_Wm2", "CloudCover_P", "Temp_Air"]].mean()
        )

        # Calcola i valori predetti
        df["Predetto"] = model.predict(X_pred)



        # Grafico: reale vs predetto
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["E_INT_Daily_kWh"], y=df["Predetto"],
            mode='markers',
            marker=dict(size=5, color='deepskyblue', opacity=0.6),
            name="Punti dati"
        ))
        fig.add_trace(go.Scatter(
            x=[df["E_INT_Daily_kWh"].min(), df["E_INT_Daily_kWh"].max()],
            y=[df["E_INT_Daily_kWh"].min(), df["E_INT_Daily_kWh"].max()],
            mode='lines',
            line=dict(color='orange', dash='dash'),
            name="Linea ideale y = x"
        ))
        fig.update_layout(
            title="📈 Confronto produzione reale vs predetta",
            xaxis_title="Produzione reale (kWh)",
            yaxis_title="Produzione predetta (kWh)",
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Mostra metriche
        mae_val = st.session_state.get("last_mae", float("nan"))
        r2_val = st.session_state.get("last_r2", float("nan"))
        st.metric("MAE (Errore medio assoluto)", f"{mae_val:.2f}")
        st.metric("R² (Coefficiente di determinazione)", f"{r2_val:.3f}")
        # --- Importanza delle feature ---
        if os.path.exists("feature_importances.csv"):
            feat = pd.read_csv("feature_importances.csv", index_col=0)
            fig_feat = go.Figure()
            fig_feat.add_trace(go.Bar(
                x=feat.index,
                y=feat["0"],
                marker_color=['orange', 'deepskyblue', 'lightgreen']
            ))
            fig_feat.update_layout(
                title="🔍 Importanza delle variabili nel modello",
                xaxis_title="Feature",
                yaxis_title="Importanza relativa",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_feat, use_container_width=True)


with tab3:
    st.subheader("Previsioni (PT15M, tilt/orient, provider toggle)")
    go = st.button("Calcola previsioni (Ieri/Oggi/Domani/Dopodomani)")
    model = load_model()
    results = {}

    sections = {"Ieri": st.container(), "Oggi": st.container(), "Domani": st.container(), "Dopodomani": st.container()}

    if go:
        for label, off in [("Ieri",-1),("Oggi",0),("Domani",1),("Dopodomani",2)]:
            with sections[label]:
                st.markdown(f"### {label}")
                dfp, energy, peak_kW, peak_pct, cloud_mean, provider, status, url = forecast_for_day(
                    lat_sidebar, lon_sidebar, off, label, model, st.session_state["tilt"], st.session_state["orient"], provider_pref, plant_kw, autosave=autosave
                )
                results[label] = dfp
                st.caption(f"Provider: **{provider}** | Stato: **{status}**")
                if url: st.code(url, language="text")
                if dfp is None or dfp.empty:
                    st.warning("Nessun dato disponibile.")
                else:
                    metr1, metr2, metr3, metr4 = st.columns(4)
                    metr1.metric(f"Energia stimata {label}", f"{energy:.1f} kWh")
                    metr2.metric("Picco stimato", f"{peak_kW:.1f} kW")
                    metr3.metric("% della targa", f"{peak_pct:.1f}%")
                    metr4.metric("Nuvolosità media", f"{cloud_mean:.0f}%")
                    chart_df = dfp.set_index("time")[["kWh_curve"]].rename(columns={"kWh_curve":"Produzione stimata (kWh/15min)"})
                    st.line_chart(chart_df)

                    # Download curva 15-min (per-giorno)
                    csv_buf = io.StringIO()
                    out_df = dfp[["time","GlobalRad_W","CloudCover_P","rad_corr","kWh_curve","kW_inst"]].copy()
                    out_df.to_csv(csv_buf, index=False)
                    st.download_button(f"⬇️ Scarica curva 15-min ({label})", csv_buf.getvalue(),
                                       file_name=f"curve_{label.replace('ò','o').lower()}_15min.csv", mime="text/csv")

                    # Scarica aggregato giornaliero (per-giorno)
                    daily_buf = io.StringIO()
                    pd.DataFrame([{"date": str((datetime.now(timezone.utc).date() + timedelta(days=off))),
                                   "energy_kWh": energy, "peak_kW": peak_kW, "cloud_mean": cloud_mean}]).to_csv(daily_buf, index=False)
                    st.download_button(f"⬇️ Scarica aggregato ({label})", daily_buf.getvalue(),
                                       file_name=f"daily_{label.replace('ò','o').lower()}.csv", mime="text/csv")

        # confronto + export unico
        st.subheader("📊 Confronto curve (4 giorni, 15 min)")
        comp = pd.DataFrame()
        for lbl, dfp in results.items():
            if dfp is not None and not dfp.empty:
                tmp = dfp[["time","kWh_curve"]].rename(columns={"kWh_curve": lbl})
                comp = tmp if comp.empty else pd.merge(comp, tmp, on="time", how="outer")
        if not comp.empty:
            comp = comp.set_index("time")
            st.line_chart(comp)

            # download unico delle 4 curve
            all_curves = pd.DataFrame()
            for lbl, dfp in results.items():
                if dfp is not None and not dfp.empty:
                    tmp = dfp[["time","GlobalRad_W","CloudCover_P","rad_corr","kWh_curve","kW_inst"]].copy()
                    tmp["giorno"] = lbl
                    all_curves = pd.concat([all_curves, tmp], ignore_index=True)
            if not all_curves.empty:
                buf_all = io.StringIO()
                all_curves.to_csv(buf_all, index=False)
                st.download_button("⬇️ Scarica TUTTE le curve (CSV unico)", buf_all.getvalue(), "all_curves_15min.csv", "text/csv")
        else:
            st.info("Nessuna curva disponibile per il confronto.")

with tab4:
    st.subheader("Mappa impianto (satellitare)")
    m = folium.Map(location=[lat_sidebar, lon_sidebar], zoom_start=15, tiles=None)
    # Satellite tiles (ESRI World Imagery)
    folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri World Imagery", name="Satellite").add_to(m)
    folium.Marker(
        location=[lat_sidebar, lon_sidebar],
        tooltip="Impianto FV",
        popup=folium.Popup(html=f"<b>Impianto FV</b><br>Lat: {lat_sidebar:.6f}<br>Lon: {lon_sidebar:.6f}<br>Tilt: {st.session_state['tilt']}°<br>Orient: {st.session_state['orient']}°", max_width=250)
    ).add_to(m)
    folium.LayerControl().add_to(m)
    st_folium(m, width=900, height=550)
