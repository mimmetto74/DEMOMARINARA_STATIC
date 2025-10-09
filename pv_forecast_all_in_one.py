import os, io, requests, joblib
import pandas as pd, numpy as np
from datetime import datetime, timedelta, timezone
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import folium
from streamlit_folium import st_folium

def normalize_real_csv(file_like):
    import pandas as pd
    try:
        df = pd.read_csv(file_like, sep=';', decimal=',', engine='python')
        if df.shape[1] == 1:
            file_like.seek(0); df = pd.read_csv(file_like)
    except Exception:
        file_like.seek(0); df = pd.read_csv(file_like)
    df.columns = [str(c).strip() for c in df.columns]
    time_col = df.columns[0]; val_col = df.columns[1] if len(df.columns)>1 else None
    if val_col is None: raise ValueError("CSV non valido: servono 2 colonne (timestamp; valore).")
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    # drop tz if any
    try:
        if hasattr(df[time_col].dt, 'tz') and df[time_col].dt.tz is not None:
            df[time_col] = df[time_col].dt.tz_localize(None)
    except Exception:
        pass
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    df[val_col] = pd.to_numeric(df[val_col], errors='coerce').fillna(0.0)
    s = df.set_index(time_col)[val_col].astype(float).rename('kWh_15m')
    s = s[~s.index.duplicated(keep='first')]
    idx = pd.date_range(s.index.min(), s.index.max(), freq='15T')
    s = s.reindex(idx).fillna(0.0)
    df15 = s.to_frame(); df15['kW_inst'] = df15['kWh_15m'] * 4.0
    daily = df15['kWh_15m'].resample('D').sum().to_frame('kWh_day'); daily['kW_peak'] = df15['kW_inst'].resample('D').max()
    return df15, daily




def evaluate_prediction_vs_real(real15, pred15, tz="Europe/Rome", apply_shift=0):
    import pandas as pd, numpy as np
    pred = pred15.copy()
    if 'kWh_15m' in pred.columns: e = pred['kWh_15m']
    elif 'kWh_curve' in pred.columns: e = pred['kWh_curve']
    elif 'kW_inst' in pred.columns: e = pred['kW_inst'] / 4.0
    else: raise ValueError("Nel dataset previsione manca 'kWh_15m' o 'kWh_curve' (o 'kW_inst').")
    if apply_shift != 0: e = e.shift(apply_shift)
    df = real15[['kWh_15m']].rename(columns={'kWh_15m':'kWh_real'}).join(e.rename('kWh_pred'), how='inner')
    err = df['kWh_pred'] - df['kWh_real']
    mae = float(err.abs().mean()); rmse = float(np.sqrt(np.mean(err**2))) if len(df) else float('nan')
    mape = float((err.abs() / df['kWh_real'].replace(0, np.nan)).dropna().mean() * 100.0) if (df['kWh_real']>0).any() else float('nan')
    r2 = float(np.corrcoef(df['kWh_real'], df['kWh_pred'])[0,1]**2) if len(df) > 1 else float('nan')
    daily = df.resample('D').sum(); daily['abs_pct_err'] = (daily['kWh_pred'] - daily['kWh_real']).abs() / daily['kWh_real'].replace(0, np.nan) * 100.0
    metrics = {'n_points_15m': int(len(df)), 'MAE_15m_kWh': mae, 'RMSE_15m_kWh': rmse, 'MAPE_15m_%': mape, 'R2': r2,
               'MAE_daily_kWh': float((daily['kWh_pred'] - daily['kWh_real']).abs().mean()) if len(daily) else float('nan'),
               'MAPE_daily_%': float(daily['abs_pct_err'].mean()) if len(daily) else float('nan'),
               'Energy_real_kWh': float(df['kWh_real'].sum()), 'Energy_pred_kWh': float(df['kWh_pred'].sum())}
    # --- TZ normalize: convert tz-aware to Europe/Rome and drop tz to avoid join errors ---
    def _to_naive_local(idx):
        try:
            if getattr(idx, 'tz', None) is not None:
                try:
                    return idx.tz_convert(tz).tz_localize(None)
                except Exception:
                    return idx.tz_localize(None)
        except Exception:
            return idx
        return idx

    try:
        if hasattr(real15, 'index'):
            real15 = real15.copy()
            real15.index = _to_naive_local(real15.index)
        if hasattr(pred, 'index'):
            pred = pred.copy()
            pred.index = _to_naive_local(pred.index)
    except Exception:
        pass
    
    return metrics, df, daily



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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def train_model():
    df0 = load_data()

    # Normalizzazione nomi colonne
    if "E_INT_Daily_KWh" in df0.columns and "E_INT_Daily_kWh" not in df0.columns:
        df0 = df0.rename(columns={"E_INT_Daily_KWh": "E_INT_Daily_kWh"})

    # Pulizia dati
    df = df0.dropna(subset=["E_INT_Daily_kWh", "G_M0_Wm2"])
    if df.empty:
        return float("nan"), float("nan"), None, None

    # ---------------- FEATURE ENGINEERING ----------------
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["dayofyear"] = df["Date"].dt.dayofyear
        df["month"] = df["Date"].dt.month
    else:
        df["dayofyear"] = np.arange(len(df)) % 365
        df["month"] = ((np.arange(len(df)) % 365) // 30) + 1

    # CloudCover inversa e radiazione corretta
    if "CloudCover_P" in df.columns:
        df["cloud_inv"] = 100 - df["CloudCover_P"]
        df["rad_eff"] = df["G_M0_Wm2"] * (df["cloud_inv"] / 100.0)
    else:
        df["cloud_inv"] = 100.0
        df["rad_eff"] = df["G_M0_Wm2"]

    # ---------------- DATASET E MODELLO ----------------
    features = ["G_M0_Wm2", "rad_eff", "cloud_inv", "dayofyear", "month"]
    target = "E_INT_Daily_kWh"

    train = df[df["Date"] < "2025-01-01"]
    test = df[df["Date"] >= "2025-01-01"]
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)

    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
    else:
        mae, r2 = float("nan"), float("nan")

    st.info(f"✅ Modello RandomForest addestrato — MAE={mae:.2f} kWh, R²={r2:.3f}")
    return mae, r2, model.named_steps["rf"].feature_importances_.tolist(), features

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

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Storico","🛠️ Modello","🔮 Previsioni 4 giorni (15 min)","🗺️ Mappa", "✅ Validazione"])


with tab1:
    st.subheader("Storico produzione (kWh)")

    df_hist = df.copy()
    slope = st.session_state.get("slope", None)
    intercept = st.session_state.get("intercept", 0.0)
    try:
        st.session_state["slope"] = float(slope)
        st.session_state["intercept"] = float(intercept)
    except Exception:
        pass

    # Fallback: prova dal modello se non in sessione
    if slope is None and "model" in st.session_state:
        try:
            slope = float(st.session_state["model"].get("slope", 1.0))
            intercept = float(st.session_state["model"].get("intercept", 0.0))
        except Exception:
            slope, intercept = 1.0, 0.0

    if slope is None:
        slope, intercept = 1.0, 0.0

    # Irradianza -> kWh equivalenti tramite retta del modello
    if "G_M0_Wm2" in df_hist.columns:
        df_hist["Irradianza (kWh eq)"] = (df_hist["G_M0_Wm2"] * float(slope) + float(intercept)).clip(lower=0)

    plot = (
        df_hist[["time","E_INT_Daily_kWh","Irradianza (kWh eq)"]]
        .rename(columns={"E_INT_Daily_kWh":"Produzione reale (kWh)"})
        .dropna(subset=["time"])
        .copy()
    )
    plot["time"] = pd.to_datetime(plot["time"])
    plot["day_str"] = plot["time"].dt.strftime("%Y-%m-%d")

    long = plot.melt(["time","day_str"], var_name="Serie", value_name="kWh")

    ch = (
        alt.Chart(long)
        .mark_line()
        .encode(
            x=alt.X("time:T", title="Giorno"),
            y=alt.Y("kWh:Q", title="kWh (giornalieri)"),
            color="Serie:N",
            tooltip=[
                alt.Tooltip("day_str:N", title="Giorno"),
                alt.Tooltip("Serie:N"),
                alt.Tooltip("kWh:Q", title="kWh", format=".0f"),
            ],
        )
        .interactive()
    )
    st.altair_chart(ch, use_container_width=True)


with tab2:
    c1, c2, c3 = st.columns(3)
    if c1.button("Addestra / Riaddestra modello"):
        mae, r2, coef, intercept = train_model()
        st.success(f"Modello addestrato ✅  MAE: {mae:.2f} | R²: {r2:.3f}")
        if coef is not None and intercept is not None:
            st.info(f"**Slope**: {coef:.6f}  |  **Intercept**: {intercept:.3f}")
    if os.path.exists(MODEL_PATH):
        model = load_model()
        try:
            coef = float(model.coef_[0]); intercept = float(model.intercept_)
            c2.metric("Slope (kWh per unità irradianza)", f"{coef:.6f}")
            c3.metric("Intercept", f"{intercept:.3f}")
        except Exception:
            pass

with tab3:
    st.subheader("Previsioni (PT15M, tilt/orient, provider toggle)")
    go = st.button("Calcola previsioni (Ieri/Oggi/Domani/Dopodomani)")
    model = load_model()
    results = {}
    st.session_state.setdefault('pred_curves', {})

    sections = {"Ieri": st.container(), "Oggi": st.container(), "Domani": st.container(), "Dopodomani": st.container()}

    if go:
        for label, off in [("Ieri",-1),("Oggi",0),("Domani",1),("Dopodomani",2)]:
            with sections[label]:
                st.markdown(f"### {label}")
                dfp, energy, peak_kW, peak_pct, cloud_mean, provider, status, url = forecast_for_day(
                    lat_sidebar, lon_sidebar, off, label, model, st.session_state["tilt"], st.session_state["orient"], provider_pref, plant_kw, autosave=autosave
                )
                results[label] = dfp
                try:
                    dfpp = dfp.copy()
                    dfpp['time'] = pd.to_datetime(dfpp['time'], utc=True, errors='coerce')
                    if hasattr(dfpp['time'].dt, 'tz'):
                        dfpp['time'] = dfpp['time'].dt.tz_convert(tz).dt.tz_localize(None)
                    st.session_state['pred_curves'][label] = dfpp
                except Exception:
                    st.session_state['pred_curves'][label] = dfp
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



with tab5:
    st.subheader("✅ Validazione previsioni vs produzione reale")
    st.caption("Carica un CSV con la produzione reale (SMA: separatore ';' e virgola decimale). Oppure usa le previsioni calcolate.")

    real_file = st.file_uploader("CSV produzione reale", type=["csv"])
    use_session_pred = st.toggle("Usa previsioni calcolate nel tab Previsioni", value=True)
    tz_local = st.toggle("Usa fuso orario Europe/Rome (altrimenti UTC)", value=True)", value=True)", value=True)", value=True)\", value=True)
    tz = \"Europe/Rome\" if tz_local else \"UTC\"
    pred_file = None if use_session_pred else st.file_uploader("CSV previsioni (opzionale)", type=["csv"], help="Colonna kWh_15m o kWh_curve (oppure kW_inst)")

    if real_file is not None:
        try:
            df_real15, df_real_daily = normalize_real_csv(real_file)
            st.success(f"Dati reali: {df_real15.index.min()} → {df_real15.index.max()}  ({len(df_real_daily)} giorni)")

            pred_source = None
            if use_session_pred and 'pred_curves' in st.session_state and len(st.session_state['pred_curves'])>0:
                labels = list(st.session_state['pred_curves'].keys())
                sel = st.selectbox("Scegli la previsione calcolata", labels, index=0)
                pred_source = st.session_state['pred_curves'][sel].copy().set_index('time')
            elif not use_session_pred and pred_file is not None:
                import pandas as pd
                pred_df = pd.read_csv(pred_file, parse_dates=['time']).set_index('time')
                pred_source = pred_df
            else:
                st.info("Seleziona una previsione o carica un CSV con la curva prevista.")

            if pred_source is not None:
                lag = st.slider("Shift previsione (multipli di 15 minuti)", -8, 8, 0)
                metrics, df_eval, df_daily = evaluate_prediction_vs_real(df_real15, pred_source, tz=tz, apply_shift=lag)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("MAE (kWh / 15m)", f"{metrics['MAE_15m_kWh']:.3f}")
                c2.metric("RMSE (kWh / 15m)", f"{metrics['RMSE_15m_kWh']:.3f}")
                c3.metric("MAPE 15m (%)", f"{metrics['MAPE_15m_%']:.1f}%")
                c4.metric("R²", f"{metrics['R2']:.3f}")
                c5, c6 = st.columns(2)
                c5.metric("Energia reale (kWh)", f"{metrics['Energy_real_kWh']:.1f}")
                c6.metric("Energia prevista (kWh)", f"{metrics['Energy_pred_kWh']:.1f}")

                plot = df_eval.reset_index().rename(columns={'index':'time'})
                plot['time_str'] = plot['time'].dt.strftime('%Y-%m-%d %H:%M')
                long = plot[['time','time_str','kWh_real','kWh_pred']].melt(['time','time_str'], var_name='Serie', value_name='kWh_15m')
                ch = alt.Chart(long).mark_line().encode(
                    x=alt.X('time:T', title='Data / Ora'),
                    y=alt.Y('kWh_15m:Q', title='kWh / 15 min'),
                    color='Serie:N',
                    tooltip=[alt.Tooltip('time_str:N', title='Data/ora'), alt.Tooltip('Serie:N'), alt.Tooltip('kWh_15m:Q', title='kWh (15m)', format='.3f')]
                ).interactive()
                st.altair_chart(ch, use_container_width=True)

                sc = alt.Chart(df_eval.reset_index()).mark_point().encode(
                    x=alt.X('kWh_real:Q', title='Reale (kWh/15m)'),
                    y=alt.Y('kWh_pred:Q', title='Prevista (kWh/15m)'),
                    tooltip=[alt.Tooltip('time_str:N', title='Data/ora'), alt.Tooltip('kWh_real:Q'), alt.Tooltip('kWh_pred:Q')]
                ).interactive()
                st.altair_chart(sc, use_container_width=True)

                dl = df_daily.reset_index().melt('time', var_name='Serie', value_name='kWh_day')
                chd = alt.Chart(dl).mark_bar().encode(
                    x=alt.X('time:T', title='Giorno'),
                    y=alt.Y('kWh_day:Q', title='kWh (giorno)'),
                    color='Serie:N'
                )
                st.altair_chart(chd, use_container_width=True)

                import io
                buf15 = io.StringIO(); df_eval.to_csv(buf15, index=True)
                st.download_button("⬇️ Scarica confronto 15‑min (CSV)", buf15.getvalue(), file_name="confronto_15min.csv", mime="text/csv")
                bufd = io.StringIO(); df_daily.to_csv(bufd, index=True)
                st.download_button("⬇️ Scarica confronto giornaliero (CSV)", bufd.getvalue(), file_name="confronto_daily.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Errore nel caricamento o confronto: {e}")
