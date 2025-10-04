import os, io, requests, joblib
import pandas as pd, numpy as np
from datetime import datetime, timedelta, timezone
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

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

def train_model():
    df = load_data().dropna(subset=["E_INT_Daily_kWh","G_M0_Wm2"] if "E_INT_Daily_kWh" in load_data().columns else ["E_INT_Daily_KWh","G_M0_Wm2"])
    # normalizza nome colonna kWh
    if "E_INT_Daily_KWh" in df.columns and "E_INT_Daily_kWh" not in df.columns:
        df = df.rename(columns={"E_INT_Daily_KWh":"E_INT_Daily_kWh"})
    if df.empty:
        return float("nan"), float("nan"), None, None
    train = df[df["Date"] < "2025-01-01"]
    test  = df[df["Date"] >= "2025-01-01"]
    X_train, y_train = train[["G_M0_Wm2"]], train["E_INT_Daily_kWh"]
    model = LinearRegression().fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    mae = r2 = float("nan")
    if len(test) > 0:
        y_pred = model.predict(test[["G_M0_Wm2"]])
        mae = float(mean_absolute_error(test["E_INT_Daily_KWh"] if "E_INT_Daily_KWh" in test.columns else test["E_INT_Daily_kWh"], y_pred))
        r2  = float(r2_score(test["E_INT_Daily_KWh"] if "E_INT_Daily_KWh" in test.columns else test["E_INT_Daily_kWh"], y_pred))
    coef = float(model.coef_[0]) if hasattr(model, "coef_") else None
    intercept = float(model.intercept_) if hasattr(model, "intercept_") else None
    return mae, r2, coef, intercept

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
    r = requests.get(url, auth=(MM_USER, MM_PASS), timeout=25)
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
        return None, 0.0, 0.0, 0.0
    df = df.copy().sort_values("time")
    # forza notturno a zero (valori negativi o troppo bassi portati a 0)
    df["GlobalRad_W"] = df["GlobalRad_W"].clip(lower=0)
    df["CloudCover_P"] = df["CloudCover_P"].clip(lower=0, upper=100)
    df["rad_corr"] = df["GlobalRad_W"] * (1 - df["CloudCover_P"]/100.0)
    sum_rad = df["rad_corr"].sum()
    pred_kwh = float(model.predict([[sum_rad]])[0]) if sum_rad > 0 else 0.0
    if sum_rad > 0:
        df["kWh_curve"] = pred_kwh * (df["rad_corr"]/sum_rad)
    else:
        df["kWh_curve"] = 0.0
    # potenza istantanea stimata (kWh su 15 min * 4 = kW)
    df["kW_inst"] = df["kWh_curve"] * 4.0
    peak_kW = float(df["kW_inst"].max()) if len(df) else 0.0
    peak_pct = float(peak_kW/plant_kw*100.0) if plant_kw > 0 else 0.0
    cloud_mean = float(df["CloudCover_P"].mean()) if "CloudCover_P" in df.columns else float("nan")
    return df, pred_kwh, peak_kW, peak_pct, cloud_mean

def forecast_for_day(lat, lon, offset_days, label, model, tilt, orient, provider_pref, plant_kw):
    day = (datetime.now(timezone.utc).date() + timedelta(days=offset_days))
    start_iso = f"{day}T00:00:00Z"; end_iso = f"{day + timedelta(days=1)}T00:00:00Z"
    provider = "Meteomatics"; status = "OK"; url = ""
    df = None

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
        return None, 0.0, 0.0, 0.0, provider, status, url

    df2, pred_kwh, peak_kW, peak_pct, cloud_mean = compute_curve_and_daily(df, model, plant_kw)

    write_log(timestamp=datetime.utcnow().isoformat(), day_label=label,
              provider=provider, status=status, url=url, lat=lat, lon=lon,
              tilt=tilt, orient=orient, sum_rad_corr=float(df2["rad_corr"].sum()), pred_kwh=float(pred_kwh),
              peak_kW=float(peak_kW), cloud_mean=float(cloud_mean), note="")

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

    return df2, pred_kwh, peak_kW, peak_pct, provider, status, url

# -------- UI ---------
st.title("☀️ Solar Forecast - ROBOTRONIX for IMEPOWER")

ensure_log_file()
st.sidebar.header("Impostazioni")
provider_pref = st.sidebar.selectbox("Fonte meteo:", ["Auto","Meteomatics","Open-Meteo"])
plant_kw = st.sidebar.number_input("Potenza di targa impianto (kW)", value=1000.0, step=50.0, min_value=0.0)

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

tab1, tab2, tab3 = st.tabs(["📊 Storico","🛠️ Modello","🔮 Previsioni 4 giorni (15 min)"])

with tab1:
    try:
        df = load_data()
        st.subheader("Storico produzione (kWh) e irradianza (W/m²)")
        if "E_INT_Daily_KWh" in df.columns and "E_INT_Daily_kWh" not in df.columns:
            df = df.rename(columns={"E_INT_Daily_KWh":"E_INT_Daily_kWh"})
        st.line_chart(df.set_index("Date")[["E_INT_Daily_kWh","G_M0_Wm2"]])
    except Exception as e:
        st.error(f"Impossibile caricare dataset: {e}")

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
    cc1, cc2, cc3, cc4 = st.columns(4)
    lat = cc1.number_input("Latitudine", value=DEFAULT_LAT, format="%.6f")
    lon = cc2.number_input("Longitudine", value=DEFAULT_LON, format="%.6f")
    st.session_state["tilt"] = cc3.slider("Tilt (°)", min_value=0, max_value=90, value=st.session_state["tilt"], step=1)
    st.session_state["orient"] = cc4.slider("Orientation (°, 0=N, 90=E, 180=S, 270=W)", min_value=0, max_value=360, value=st.session_state["orient"], step=5)
    st.caption("Se Tilt > 0 uso `global_rad_tilt_<tilt>_orientation_<orient>:W`, altrimenti `global_rad:W`.")

    go = st.button("Calcola previsioni (Ieri/Oggi/Domani/Dopodomani)")
    model = load_model()
    results = {}

    sections = {"Ieri": st.container(), "Oggi": st.container(), "Domani": st.container(), "Dopodomani": st.container()}

    if go:
        for label, off in [("Ieri",-1),("Oggi",0),("Domani",1),("Dopodomani",2)]:
            with sections[label]:
                st.markdown(f"### {label}")
                dfp, energy, peak_kW, peak_pct, provider, status, url = forecast_for_day(
                    lat, lon, off, label, model, st.session_state["tilt"], st.session_state["orient"], provider_pref, plant_kw
                )
                results[label] = dfp
                st.caption(f"Provider: **{provider}** | Stato: **{status}**")
                if url: st.code(url, language="text")
                if dfp is None or dfp.empty:
                    st.warning("Nessun dato disponibile.")
                else:
                    metr1, metr2, metr3 = st.columns(3)
                    metr1.metric(f"Energia stimata {label}", f"{energy:.1f} kWh")
                    metr2.metric("Picco stimato", f"{peak_kW:.1f} kW")
                    metr3.metric("% della targa", f"{peak_pct:.1f}%")
                    chart_df = dfp.set_index("time")[["kWh_curve"]].rename(columns={"kWh_curve":"Produzione stimata (kWh/15min)"})
                    st.line_chart(chart_df)

                    # Download curve 15-min
                    csv_buf = io.StringIO()
                    out_df = dfp[["time","GlobalRad_W","CloudCover_P","rad_corr","kWh_curve","kW_inst"]].copy()
                    out_df.to_csv(csv_buf, index=False)
                    st.download_button(f"⬇️ Scarica curva 15-min ({label})", csv_buf.getvalue(),
                                       file_name=f"curve_{label.replace('ò','o').lower()}_15min.csv", mime="text/csv")

                    # Scarica aggregato giornaliero
                    daily_buf = io.StringIO()
                    pd.DataFrame([{"date": str((datetime.now(timezone.utc).date() + timedelta(days=off))),
                                   "energy_kWh": energy, "peak_kW": peak_kW}]).to_csv(daily_buf, index=False)
                    st.download_button(f"⬇️ Scarica aggregato ({label})", daily_buf.getvalue(),
                                       file_name=f"daily_{label.replace('ò','o').lower()}.csv", mime="text/csv")

        # confronto
        st.subheader("📊 Confronto curve (4 giorni, 15 min)")
        comp = pd.DataFrame()
        for lbl, dfp in results.items():
            if dfp is not None and not dfp.empty:
                tmp = dfp[["time","kWh_curve"]].rename(columns={"kWh_curve": lbl})
                comp = tmp if comp.empty else pd.merge(comp, tmp, on="time", how="outer")
        if not comp.empty:
            comp = comp.set_index("time")
            st.line_chart(comp)
        else:
            st.info("Nessuna curva disponibile per il confronto.")
