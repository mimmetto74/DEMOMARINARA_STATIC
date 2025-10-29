# -*- coding: utf-8 -*-
"""
Solar Forecast - ROBOTRONIX for IMEPOWER (Versione v2 – con Coerenza Energetica)
- Build completa con struttura ordinata, modalità Debug, training RF,
  previsioni 15-min (Ieri/Oggi/Domani/Dopodomani), mappa e validazione DOMANI.
- Nuovo indicatore nel Tab "Validazione DOMANI": 🔋 Coerenza energetica.
"""

import os, json, joblib, requests, numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
import streamlit as st
import plotly.graph_objects as go

# --------------------------- Config Streamlit -------------------------------
st.set_page_config(page_title='Solar Forecast - ROBOTRONIX (v2 DEBUG)', layout='wide')
_PORT = os.environ.get('PORT')
if _PORT:
    try:
        st.set_option('server.headless', True)
        st.set_option('server.address', '0.0.0.0')
        st.set_option('server.port', int(_PORT))
    except Exception:
        pass

# ------------------------------- Globals -----------------------------------
DATA_PATH = os.environ.get('PV_DAILY_DATA', 'Dataset_Daily_EnergiaSeparata_2020_2025.csv')
MODEL_PATH = os.environ.get('PV_MODEL_PATH', 'rf_model.joblib')
LOG_DIR    = os.environ.get('PV_LOG_DIR', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

DEFAULT_LAT = float(os.environ.get('PV_LAT', '40.643278'))
DEFAULT_LON = float(os.environ.get('PV_LON', '16.986083'))
DEFAULT_TILT = float(os.environ.get('PV_TILT', '10'))
DEFAULT_ORIENT = float(os.environ.get('PV_ORIENT', '180'))
DEFAULT_PLANT_KW = float(os.environ.get('PV_PLANT_KW', '947.25'))

MM_USER = os.environ.get('MM_USER', 'robotronixsrl_daniello_fabio')
MM_PASS = os.environ.get('MM_PASS', 'xaRVRh8sa5EV70F0B88u')

APP_USER = os.environ.get('APP_USER', 'admin')
APP_PASS = os.environ.get('APP_PASS', 'robotronix')

# ------------------------------ Utils --------------------------------------

import pytz
from dateutil import tz

def fix_timezone(df, column="Date", tz_local="Europe/Rome"):
    """
    Allinea il timestamp all'ora locale italiana (Europe/Rome).
    - Se il dataset è in UTC, lo converte a Europe/Rome.
    - Se è già locale, lo mantiene coerente.
    """
    if column not in df.columns:
        return df

    df[column] = pd.to_datetime(df[column], errors="coerce")

    try:
        # Se la colonna non ha timezone (tipico dei CSV locali)
        if df[column].dt.tz is None:
            df[column] = df[column].dt.tz_localize("UTC").dt.tz_convert(tz_local)
        else:
            df[column] = df[column].dt.tz_convert(tz_local)
    except Exception as e:
        print(f"[WARN] Timezone conversion failed for {column}: {e}")

    return df

def write_log(**kwargs):
    try:
        ts = datetime.utcnow().strftime('%Y%m%d')
        with open(os.path.join(LOG_DIR, f'forecast_log_{ts}.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(kwargs, default=str) + '\n')
    except Exception:
        pass

@st.cache_data(show_spinner=False, ttl=600)
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['Date'])
    # ---- Correzione timezone automatica ----
    df = fix_timezone(df, column='Date')

    # ✅ RIENTRA queste righe (devono stare dentro la funzione)
    if 'E_INT_Daily_KWh' in df.columns and 'E_INT_Daily_kWh' not in df.columns:
        df = df.rename(columns={'E_INT_Daily_KWh': 'E_INT_Daily_kWh'})
    
    return df

def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        return None
    obj = joblib.load(path)
    if isinstance(obj, dict) and 'model' in obj:  # compat dict
        return obj['model']
    return obj

# ----------------------------- Providers -----------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_meteomatics_pt15m(lat, lon, start_iso, end_iso, tilt=None, orient=None):
    if not MM_USER or not MM_PASS:
        raise RuntimeError('Credenziali Meteomatics mancanti')
    if tilt is not None and orient is not None and float(tilt) > 0:
        rad_param = f'global_rad_tilt_{int(round(float(tilt)))}_orientation_{int(round(float(orient)))}:W'
    else:
        rad_param = 'global_rad:W'
    params = f'{rad_param},total_cloud_cover:p,t_2m:C'
    url = f'https://api.meteomatics.com/{start_iso}--{end_iso}:PT15M/{params}/{lat},{lon}/json'
    try:
        r = requests.get(url, auth=(MM_USER, MM_PASS), timeout=30)
        if r.status_code == 401:
            raise RuntimeError('401 Unauthorized: credenziali Meteomatics non valide')
        if r.status_code == 404:
            raise RuntimeError('404: Nessun dato disponibile per il periodo richiesto')
        if not r.ok:
            raise RuntimeError(f'HTTP {r.status_code}: {r.text[:200]}')
        j = r.json()
    except requests.Timeout as e:
        raise RuntimeError('Timeout contattando Meteomatics (30s)') from e
    except Exception as e:
        raise RuntimeError(f'Errore rete Meteomatics: {e}') from e

    rows = []
    for blk in j.get('data', []):
        prm = blk.get('parameter')
        col = 'GlobalRad_W'
        if prm == 'total_cloud_cover:p':
            col = 'CloudCover_P'
        elif prm == 't_2m:C':
            col = 'Temp_Air'
        for d in blk.get('coordinates', [{}])[0].get('dates', []):
            rows.append({'time': d.get('date'), col: d.get('value')})

    df = pd.DataFrame(rows)
df = pd.DataFrame(rows)

# ---- Correzione timezone automatica per dati meteo ---
df_meteo = fix_timezone(df_meteo, column='datetime')

if df.empty:
    return '', df

df['time'] = pd.to_datetime(df['time'])
df = df.groupby('time', as_index=False).mean().sort_values('time')

for c in ['GlobaRad_W', 'CloudCover_P', 'Temp_Air']:
    if c not in df.columns:
        df[c] = np.nan

df['provider'] = 'Meteomatics'
return '', df

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_meteomatics(lat: float, lon: float, start_date: datetime, end_date: datetime):
    """
    Recupera dati meteo da Meteomatics e li restituisce come DataFrame.
    """
    import requests

    try:
        url = f"https://api.meteomatics.com/{start_date:%Y-%m-%dT%H:%M:%SZ}--{end_date:%Y-%m-%dT%H:%M:%SZ}:PT1H/global_rad:W,total_cloud_cover:p,t_2m:C/{lat},{lon}/json"
        auth = (MM_USER, MM_PASS)
        r = requests.get(url, auth=auth, timeout=30)
        r.raise_for_status()
        j = r.json()
    except Exception as e:
        raise RuntimeError(f"Errore rete Meteomatics: {e}") from e

    # --- Parsing JSON ---
    rows = []
    for blk in j.get("data", []):
        prm = blk.get("parameter")
        col = "GlobaRad_W"
        if prm == "total_cloud_cover:p":
            col = "CloudCover_P"
        elif prm == "t_2m:C":
            col = "Temp_Air"

        for d in blk.get("coordinates", [{}])[0].get("dates", []):
            rows.append({"time": d.get("date"), col: d.get("value")})

    # --- Creazione DataFrame ---
    df = pd.DataFrame(rows)

    # --- Correzione timezone automatica ---
    df = fix_timezone(df, column="time")

    if df.empty:
        return "", df

    # --- Pulizia e aggregazione ---
    df["time"] = pd.to_datetime(df["time"])
    df = df.groupby("time", as_index=False).mean().sort_values("time")

    for c in ["GlobaRad_W", "CloudCover_P", "Temp_Air"]:
        if c not in df.columns:
            df[c] = np.nan

    df["provider"] = "Meteomatics"
    return "", df

# -------------------------- Forecasting core --------------------------------
def compute_curve_and_daily(df, model, plant_kw):
    import pytz
    if df is None or df.empty:
        return df, 0.0, 0.0, 0.0, 0.0

    if 'time' not in df.columns:
        for alt in ['Date', 'datetime', 'timestamp']:
            if alt in df.columns:
                df = df.rename(columns={alt: 'time'}); break
    if 'time' not in df.columns:
        return df, 0.0, 0.0, 0.0, 0.0

    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['time']).sort_values('time')

    try:
        tz_local = pytz.timezone("Europe/Rome")
        if df['time'].dt.tz is None:
            df['time'] = df['time'].dt.tz_localize(pytz.UTC)
        df['time'] = df['time'].dt.tz_convert(tz_local).dt.tz_localize(None)
    except Exception:
        pass

    for col in df.columns:
        if col != 'time':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'GlobalRad_W' in df.columns:
        df['rad_corr'] = df['GlobalRad_W'].fillna(0)
    elif 'G_M0_Wm2' in df.columns:
        df['rad_corr'] = df['G_M0_Wm2'].fillna(0)
        df = df.rename(columns={'G_M0_Wm2': 'GlobalRad_W'})
    else:
        df['rad_corr'] = 0.0

    df = df.set_index('time').resample('15T').mean(numeric_only=True).reset_index()

    if model is not None:
        X = pd.DataFrame()
# --- Correzione timezone automatica per dati meteo ---
df_meteo = fix_timezone(df_meteo, column='datetime')
        feats = getattr(model, "feature_names_in_", [])
        col_map = {'G_M0_Wm2':'GlobalRad_W','GlobalRad_W':'G_M0_Wm2'}
        for f_ in feats:
            if f_ in df.columns:
                X[f_] = df[f_]
            elif f_ in col_map and col_map[f_] in df.columns:
                X[f_] = df[col_map[f_]]
            else:
                X[f_] = 0.0
        try:
            df['kWh_curve'] = model.predict(X)
        except ValueError:
            cols = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else X.columns
            X = X.reindex(columns=cols, fill_value=0)
            df['kWh_curve'] = model.predict(X)
        if df['kWh_curve'].max() > 0:
            df['kWh_curve'] = (df['kWh_curve'] / df['kWh_curve'].max()) * plant_kw
    else:
        ref = 'GlobalRad_W' if 'GlobalRad_W' in df.columns else 'rad_corr'
        max_val = max(df[ref].max(), 1.0)
        df['kWh_curve'] = df[ref] * (plant_kw/1000.0) / max_val

    df['kWh_curve'] = df['kWh_curve'].rolling(window=3, center=True, min_periods=1).mean()

    energy_kWh = float(df['kWh_curve'].sum() * 0.25)
    peak_kW = float(df['kWh_curve'].max())
    peak_pct = float(100 * peak_kW / plant_kw) if plant_kw > 0 else 0.0
    cloud_mean = float(df['CloudCover_P'].mean()) if 'CloudCover_P' in df.columns else np.nan
    return df, energy_kWh, peak_kW, peak_pct, cloud_mean

def forecast_physical(df, plant_kw, eff=0.17):
    if df is None or df.empty or 'GlobalRad_W' not in df.columns:
        return df, 0.0, 0.0, 0.0, 0.0
    df = df.copy()
    df['Energy_kWh'] = df['GlobalRad_W'].fillna(0) * eff * (15*60) / 3.6e6 * (plant_kw/1000)
    df['kWh_curve'] = df['Energy_kWh'] / 0.25
    total = float(df['Energy_kWh'].sum())
    peak_kW = float(df['kWh_curve'].max())
    peak_pct = float(100 * peak_kW / plant_kw) if plant_kw else 0.0
    cloud_mean = float(df['CloudCover_P'].mean()) if 'CloudCover_P' in df.columns else 0.0
    return df, total, peak_kW, peak_pct, cloud_mean

def forecast_hybrid(df, model, plant_kw, w_ml=0.7):
    if df is None or df.empty:
        return df, 0.0, 0.0, 0.0, 0.0
    df_ml, _, _, _, cloud = compute_curve_and_daily(df, model, plant_kw)
    df_phys, _, _, _, _ = forecast_physical(df, plant_kw)
    if df_ml is None or df_ml.empty or df_phys is None or df_phys.empty:
        return df, 0.0, 0.0, 0.0, 0.0
    n = min(len(df_ml), len(df_phys))
    df_h = df_ml.iloc[:n].copy()
    df_h['kWh_curve'] = w_ml * df_ml['kWh_curve'].iloc[:n].values + (1 - w_ml) * df_phys['kWh_curve'].iloc[:n].values
    total = float(df_h['kWh_curve'].sum() * 0.25)
    peak = float(df_h['kWh_curve'].max())
    pct = float(100 * peak / plant_kw) if plant_kw else 0.0
    return df_h, total, peak, pct, cloud

def forecast_for_day(lat, lon, offset_days, label, model, tilt, orient, provider_pref, plant_kw, autosave=True):
    day = (datetime.now(timezone.utc).date() + timedelta(days=offset_days))
    start_iso = f'{day}T00:00:00Z'; end_iso = f'{day + timedelta(days=1)}T00:00:00Z'
    provider, status, url, df = 'Meteomatics', 'OK', '', None

    def try_meteomatics():
        try: return (*fetch_meteomatics_pt15m(lat, lon, start_iso, end_iso, tilt=tilt, orient=orient), 'Meteomatics', 'OK')
        except Exception as e: return ('', None, 'Meteomatics', f'ERROR: {e}')

    def try_openmeteo():
        try: return (*fetch_openmeteo_hourly(lat, lon, str(day), str(day + timedelta(days=1))), 'Open-Meteo', 'OK')
        except Exception as e: return ('', None, 'Open-Meteo', f'ERROR: {e}')

    if provider_pref == 'Meteomatics':
        url, df, provider, status = try_meteomatics()
        if df is None or df.empty: url, df, provider, status = try_openmeteo()
    elif provider_pref == 'Open-Meteo':
        url, df, provider, status = try_openmeteo()
    else:
        url, df, provider, status = try_meteomatics()
        if df is None or df.empty: url, df, provider, status = try_openmeteo()

    if df is None or df.empty:
        write_log(timestamp=datetime.utcnow().isoformat(), day_label=label, provider=provider, status=status,
                  lat=lat, lon=lon, tilt=tilt, orient=orient, note='no data')
        return None, 0.0, 0.0, 0.0, float('nan'), provider, status, ''

    df2, energy, peak_kW, peak_pct, cloud_mean = compute_curve_and_daily(df, model, plant_kw)

    write_log(timestamp=datetime.utcnow().isoformat(), day_label=label, provider=provider, status=status,
              lat=lat, lon=lon, tilt=tilt, orient=orient,
              sum_rad_corr=float(df2['rad_corr'].sum() if 'rad_corr' in df2.columns else 0.0),
              energy_kWh=float(energy), peak_kW=float(peak_kW), cloud_mean=float(cloud_mean))

    if autosave:
        cols = [c for c in ['time','GlobalRad_W','CloudCover_P','Temp_Air','rad_corr','kWh_curve'] if c in df2.columns]
        df2[cols].to_csv(os.path.join(LOG_DIR, f'curve_{label.lower()}_15min.csv'), index=False)
        pd.DataFrame([{'date': str(day), 'energy_kWh': float(energy), 'peak_kW': float(peak_kW), 'cloud_mean': float(cloud_mean)}])
# --- Correzione timezone automatica per dati meteo ---
df_meteo = fix_timezone(df_meteo, column='datetime') \
          .to_csv(os.path.join(LOG_DIR, f'daily_{label.lower()}.csv'), index=False)

    return df2, energy, peak_kW, peak_pct, cloud_mean, provider, status, ''

# ----------------------- Model training & eval ------------------------------
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def train_model():
    df0 = load_data()
    if 'E_INT_Daily_KWh' in df0.columns and 'E_INT_Daily_kWh' not in df0.columns:
        df0 = df0.rename(columns={'E_INT_Daily_KWh': 'E_INT_Daily_kWh'})
    for col in ['CloudCover_P', 'Temp_Air']:
        if col not in df0.columns: df0[col] = np.nan
    df = df0.dropna(subset=['E_INT_Daily_kWh','G_M0_Wm2']).copy()
    X = df[['G_M0_Wm2','CloudCover_P','Temp_Air']].fillna(df[['G_M0_Wm2','CloudCover_P','Temp_Air']].mean())
    y = df['E_INT_Daily_kWh']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    mae = float(mean_absolute_error(yte, pred)); r2 = float(r2_score(yte, pred))
    joblib.dump({'model': model, 'features': X.columns.tolist()}, MODEL_PATH)
    pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False) \
      .to_csv(os.path.join(LOG_DIR, 'feature_importances.csv'))
    return mae, r2

def compare_forecast_vs_real(day_label, forecast_df, data_path=DATA_PATH):
    try:
        df_real = pd.read_csv(data_path, parse_dates=['Date'])
# --- Correzione timezone automatica ---
df = fix_timezone(df, column='Date')
    except Exception as e:
        st.warning(f'⚠️ Errore caricamento storico: {e}'); return None, None, None
    if forecast_df is None or forecast_df.empty or 'time' not in forecast_df.columns:
        return None, None, None
    f_date = pd.to_datetime(forecast_df['time'].iloc[0]).date()
    df_real_day = df_real[df_real['Date'].dt.date == f_date]
    if df_real_day.empty:
        st.info(f'Nessun dato reale per {f_date}'); return None, None, None

    dfp = forecast_df.copy()
    dfp['Potenza_kW_prevista'] = dfp['kWh_curve'] * 4
    mae = float('nan'); mape = float('nan')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dfp['time'], y=dfp['Potenza_kW_prevista'], mode='lines', name='Previsione (kW)',
                             line=dict(color='orange', width=2)))
    if 'Potenza_reale_kW' in df_real_day.columns:
        y_true = df_real_day['Potenza_reale_kW'].values[:len(dfp)]
        y_pred = dfp['Potenza_kW_prevista'].values[:len(y_true)]
        mae = float(mean_absolute_error(y_true, y_pred))
        mape = float(np.mean(np.abs((y_true - y_pred)/np.maximum(y_true, 1e-3))) * 100)
        fig.add_trace(go.Scatter(x=dfp['time'], y=y_true, mode='lines', name='Reale (kW)',
                                 line=dict(color='blue', width=2)))
    fig.update_layout(title=f'📊 Confronto previsione vs reale – {day_label}', xaxis_title='Ora',
                      yaxis_title='Potenza (kW)', template='plotly_white', height=350)
    return fig, mae, mape

# ------------------------------ Login --------------------------------------
def render_login():
    st.title('🔐 Accesso richiesto')
    st.write('Inserisci le credenziali per accedere all\'applicazione.')
    u = st.text_input('Username', value='', key='login_user')
    p = st.text_input('Password', value='', type='password', key='login_pass')
    go_btn = st.button('Accedi')
    if go_btn:
        if u == APP_USER and p == APP_PASS:
            st.session_state['authenticated'] = True; st.success('Accesso effettuato!'); st.rerun()
        else:
            st.error('❌ Credenziali non valide')

if not st.session_state.get('authenticated', False):
    render_login(); st.stop()

DEBUG = st.sidebar.checkbox("🔍 Modalità Debug", value=False)
st.sidebar.caption("La modalità Debug mostra info tecniche su dati, modello e forecast. Nessuna URL provider viene mostrata.")

# Defaults
for k,v in {'lat':DEFAULT_LAT,'lon':DEFAULT_LON,'tilt':DEFAULT_TILT,'orient':DEFAULT_ORIENT,
            'provider_pref':'Auto','plant_kw':DEFAULT_PLANT_KW}.items():
    st.session_state.setdefault(k,v)

# ------------------------------- UI Tabs -----------------------------------
st.title('☀️ Solar Forecast - ROBOTRONIX for IMEPOWER (v2)')
tab1, tab2, tab3, tab4, tab5 = st.tabs(['📊 Storico','🧠 Modello','🔮 Previsioni 4 giorni (15 min)','🗺️ Mappa','🛡️ Validazione DOMANI'])

# ---- TAB 1 ----------------------------------------------------------------
with tab1:
    try:
        df = load_data()
        st.subheader('Storico produzione (kWh) e irradianza (W/m²) — grafici separati')
        c1,c2 = st.columns(2)
        with c1:
            fig1 = go.Figure()
            if 'E_INT_Daily_kWh' in df.columns:
                fig1.add_trace(go.Scatter(x=df['Date'], y=df['E_INT_Daily_kWh'], mode='lines',
                                          name='Produzione (kWh)', line=dict(color='orange', width=2)))
            fig1.update_layout(template='plotly_white', height=350, title='⚡ Produzione giornaliera (kWh)',
                               xaxis_title='Data', yaxis_title='Energia (kWh)')
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = go.Figure()
            if 'G_M0_Wm2' in df.columns:
                fig2.add_trace(go.Scatter(x=df['Date'], y=df['G_M0_Wm2'], mode='lines',
                                          name='Irradianza (W/m²)', line=dict(color='deepskyblue', width=2)))
            fig2.update_layout(template='plotly_white', height=350, title='☀️ Irradianza giornaliera (W/m²)',
                               xaxis_title='Data', yaxis_title='W/m²')
            st.plotly_chart(fig2, use_container_width=True)

        if DEBUG:
            st.markdown("#### 🔎 Dataset info")
            st.write(f"Righe: {len(df)} | Colonne: {list(df.columns)}")
            st.dataframe(df.head(5))
            st.json({'na_count': df.isna().sum().to_dict()})
    except Exception as e:
        st.error(f'Errore caricamento storico: {e}')

# ---- TAB 2 ----------------------------------------------------------------
with tab2:
    st.subheader('🧠 Modello di previsione')
    uploaded_files = st.file_uploader("📂 Carica CSV storici addizionali (opzionale)", type=['csv'], accept_multiple_files=True)
    if uploaded_files:
        base = load_data(); dfs = [base]
        for f in uploaded_files:
            try:
                df_new = pd.read_csv(f, parse_dates=['Date'])
# --- Correzione timezone automatica ---
df = fix_timezone(df, column='Date'); dfs.append(df_new)
                st.success(f"✅ Aggiunto: {f.name} ({len(df_new)} righe)")
            except Exception as e:
                st.error(f"Errore caricamento {f.name}: {e}")
        df_merged = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['Date'])
        merged_path = os.path.join(LOG_DIR, 'merged_dataset.csv'); df_merged.to_csv(merged_path, index=False)
        st.info(f"📊 Dataset unificato salvato: `{merged_path}` — {len(df_merged)} righe.")
        st.session_state['custom_dataset'] = merged_path

    c1, c2 = st.columns([1,1])
    if c1.button('Addestra / Riaddestra modello', use_container_width=True):
        data_path = st.session_state.get('custom_dataset', DATA_PATH)
        if os.path.exists(data_path):
            df_tmp = pd.read_csv(data_path, parse_dates=['Date'])
# --- Correzione timezone automatica ---
df = fix_timezone(df, column='Date'); df_tmp.to_csv(DATA_PATH, index=False)
        mae, r2 = train_model(); st.session_state['last_mae'] = mae; st.session_state['last_r2'] = r2
        st.success(f'✅ Modello addestrato!  MAE: {mae:.2f} | R²: {r2:.3f}')

    model = load_model()
    if model is None:
        st.warning('⚠️ Modello non addestrato.')
    else:
        dfm = load_data()
        if 'E_INT_Daily_KWh' in dfm.columns and 'E_INT_Daily_kWh' not in dfm.columns:
            dfm = dfm.rename(columns={'E_INT_Daily_KWh': 'E_INT_Daily_kWh'})
        for col in ['CloudCover_P', 'Temp_Air']:
            if col not in dfm.columns: dfm[col] = np.nan
        dfm = dfm.dropna(subset=['E_INT_Daily_kWh', 'G_M0_Wm2']).copy()
        Xp = dfm[['G_M0_Wm2','CloudCover_P','Temp_Air']].fillna(dfm[['G_M0_Wm2','CloudCover_P','Temp_Air']].mean())
        dfm['Predetto'] = model.predict(Xp)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dfm['E_INT_Daily_kWh'], y=dfm['Predetto'], mode='markers',
                                 marker=dict(size=5, opacity=0.6), name='Punti'))
        minv = float(min(dfm['E_INT_Daily_kWh'].min(), dfm['Predetto'].min()))
        maxv = float(max(dfm['E_INT_Daily_kWh'].max(), dfm['Predetto'].max()))
        fig.add_trace(go.Scatter(x=[minv, maxv], y=[minv, maxv], mode='lines',
                                 line=dict(color='orange', dash='dash'), name='y = x'))
        fig.update_layout(title='📈 Reale vs Predetto (kWh/giorno)',
                          xaxis_title='Reale (kWh)', yaxis_title='Predetto (kWh)',
                          template='plotly_white', height=400)
        st.plotly_chart(fig, use_container_width=True)

        if DEBUG:
            st.markdown("#### 🔎 Modello info")
            st.json({'feature_names_in_': getattr(model, 'feature_names_in_', []),
                     'last_MAE': st.session_state.get('last_mae', None),
                     'last_R2': st.session_state.get('last_r2', None)})
            fi_path = os.path.join(LOG_DIR, 'feature_importances.csv')
            if os.path.exists(fi_path):
                try: st.dataframe(pd.read_csv(fi_path, header=None, names=['feature','importance'])
# --- Correzione timezone automatica ---
df = fix_timezone(df, column='Date'))
                except Exception: st.write("Impossibile leggere feature_importances.csv")

# ---- TAB 3 ----------------------------------------------------------------
with tab3:
    st.subheader('🔮 Previsioni 4 giorni (15 min)')
    colA, colB, colC, colD = st.columns(4)
    st.session_state['lat'] = colA.number_input('Lat', value=float(st.session_state['lat']), step=0.0001, format='%.6f')
    st.session_state['lon'] = colB.number_input('Lon', value=float(st.session_state['lon']), step=0.0001, format='%.6f')
    st.session_state['tilt'] = colC.number_input('Tilt (°)', value=float(st.session_state['tilt']), step=1.0)
    st.session_state['orient'] = colD.number_input('Orient (°)', value=float(st.session_state['orient']), step=1.0)
    colE, colF = st.columns(2)
    st.session_state['provider_pref'] = colE.selectbox('Provider', ['Auto', 'Meteomatics', 'Open-Meteo'], index=0)
    st.session_state['plant_kw'] = colF.number_input('Taglia impianto (kW)', value=float(st.session_state['plant_kw']), step=10.0)

    method = st.selectbox("Metodo di previsione", ["Random Forest", "Fisico semplificato", "Ibrido ML + Fisico"],
                          index=0, key="method_select_tab3")
    model = load_model() if method != "Fisico semplificato" else load_model()

    if method != "Fisico semplificato" and model is None:
        st.warning('⚠️ Modello non addestrato. Vai al tab "🧠 Modello".')
    else:
        if st.button('Calcola previsioni (Ieri/Oggi/Domani/Dopodomani)', use_container_width=True):
            for label, off in [('Ieri', -1), ('Oggi', 0), ('Domani', 1), ('Dopodomani', 2)]:
                try:
                    dfp, energy, peak_kW, peak_pct, cloud_mean, provider, status, _ = forecast_for_day(
                        lat=st.session_state['lat'], lon=st.session_state['lon'],
                        offset_days=off, label=label, model=model,
                        tilt=st.session_state['tilt'], orient=st.session_state['orient'],
                        provider_pref=st.session_state['provider_pref'], plant_kw=st.session_state['plant_kw'],
                        autosave=False
                    )
                except Exception as e:
                    st.error(f"Errore durante la previsione per {label}: {e}"); continue

                if dfp is None or dfp.empty:
                    st.warning(f"Nessun dato disponibile per {label}."); continue

                if method == "Fisico semplificato":
                    dfp, energy, peak_kW, peak_pct, cloud_mean = forecast_physical(dfp, st.session_state['plant_kw'])
                elif method == "Ibrido ML + Fisico":
                    dfp, energy, peak_kW, peak_pct, cloud_mean = forecast_hybrid(dfp, model, st.session_state['plant_kw'])
                else:
                    dfp, energy, peak_kW, peak_pct, cloud_mean = compute_curve_and_daily(dfp, model, st.session_state['plant_kw'])

                if label == 'Domani':
                    try:
                        base_path = os.path.join(LOG_DIR, "forecast_domani_base.csv")
                        cols = [c for c in ['time','GlobalRad_W','CloudCover_P','Temp_Air','rad_corr','kWh_curve'] if c in dfp.columns]
                        dfp[cols].to_csv(base_path, index=False)
                        st.info(f"📦 Base di confronto DOMANI salvata: {base_path}")
                    except Exception as e:
                        st.warning(f"Impossibile salvare la base DOMANI: {e}")

                fig = go.Figure()
                if 'GlobalRad_W' in dfp.columns:
                    fig.add_trace(go.Scatter(x=dfp['time'], y=dfp['GlobalRad_W'], mode='lines', name='☀️ Irradianza (W/m²)', line=dict(color='royalblue', width=2)))
                if 'kWh_curve' in dfp.columns:
                    fig.add_trace(go.Scatter(x=dfp['time'], y=dfp['kWh_curve'], mode='lines', name='⚡ Produzione stimata (kW)', line=dict(color='orange', width=3)))
                fig.update_layout(title=f"📊 Andamento previsto — {label}", xaxis_title="Ora locale (Europe/Rome)",
                                  yaxis_title="Potenza / Irradianza", template="plotly_white", height=400, hovermode="x unified")
                st.markdown(f"### **{label}**")
                st.caption(f"Provider: {provider} | Stato: {status} | Energia: {energy:.1f} kWh | Picco: {peak_kW:.1f} kW ({peak_pct:.1f}%)")
                st.plotly_chart(fig, use_container_width=True)

                if DEBUG:
                    st.markdown("#### 🔎 Forecast debug")
                    st.json({'label': label, 'provider': provider, 'status': status, 'rows': len(dfp),
                             'cols': list(dfp.columns), 'energy_kWh': round(energy, 2), 'peak_kW': round(peak_kW, 2),
                             'cloud_mean': None if (cloud_mean is None or (isinstance(cloud_mean, float) and np.isnan(cloud_mean))) else round(float(cloud_mean), 2)})
                    st.dataframe(dfp.head(5))

                st.download_button(label=f"📥 Scarica CSV previsione {label}", data=dfp.to_csv(index=False).encode('utf-8'),
                                   file_name=f"previsione_{label.lower()}.csv", mime="text/csv", key=f"download_{label.lower()}",
                                   use_container_width=True)

# ---- TAB 4 ----------------------------------------------------------------
with tab4:
    st.subheader("🛰️ Localizzazione impianto fotovoltaico (vista satellitare)")
    st.write("Visualizzazione satellitare (Esri World Imagery).")
    lat = float(st.session_state.get('lat', DEFAULT_LAT)); lon = float(st.session_state.get('lon', DEFAULT_LON))
    st.markdown(f"**Coordinate attuali:** 🌍 {lat:.6f}, {lon:.6f}")
    try:
        from streamlit_folium import st_folium; import folium
        m = folium.Map(location=[lat, lon], zoom_start=17, tiles='Esri.WorldImagery', attr='Tiles © Esri')
        folium.Marker([lat, lon], popup=f"Impianto<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}", tooltip="Impianto FV",
                      icon=folium.Icon(color='orange', icon='bolt', prefix='fa')).add_to(m)
        st_folium(m, width=900, height=500)
    except Exception as e:
        st.info("Modulo mappa non disponibile (folium/streamlit_folium).")

# ---- TAB 5 ----------------------------------------------------------------
with tab5:
    st.subheader('🛡️ Validazione previsione DOMANI')
    st.caption('Confronto tra previsione base (salvata) e previsione aggiornata: differenza assoluta 15-min e integrale.')
    base_path = os.path.join(LOG_DIR, 'forecast_domani_base.csv')
    col1, col2 = st.columns([1,1])
    refresh_provider = col1.selectbox('Provider confronto', ['Auto', 'Meteomatics', 'Open-Meteo'], index=0)
    btn = col2.button('🔄 Aggiorna confronto', use_container_width=True)
    if not os.path.exists(base_path):
        st.warning('⚠️ Nessuna base DOMANI trovata. Calcola prima Domani in Tab 3.')
    else:
        try:
            df_base = pd.read_csv(base_path, parse_dates=['time'])
# --- Correzione timezone automatica ---
df = fix_timezone(df, column='Date')
            st.caption(f"✅ Base DOMANI: {len(df_base)} punti — dal {df_base['time'].min().strftime('%d/%m %H:%M')}")
        except Exception as e:
            st.error(f"Errore lettura base DOMANI: {e}"); df_base = None
        if btn and df_base is not None:
            try:
                df_new, _, _, _, _, provider, status, _ = forecast_for_day(
                    lat=st.session_state['lat'], lon=st.session_state['lon'],
                    offset_days=1, label='Domani', model=load_model(),
                    tilt=st.session_state['tilt'], orient=st.session_state['orient'],
                    provider_pref=refresh_provider, plant_kw=st.session_state['plant_kw'], autosave=False
                )
                if df_new is None or df_new.empty:
                    st.warning("Nessun dato nuovo disponibile.")
                else:
                    df_new['time'] = pd.to_datetime(df_new['time']); df_base['time'] = pd.to_datetime(df_base['time'])
                    dfm = pd.merge_asof(df_new.sort_values('time'), df_base.sort_values('time'),
                                        on='time', tolerance=pd.Timedelta('7min'), direction='nearest',
                                        suffixes=('_new','_base'))
                    if 'kWh_curve_new' in dfm.columns and 'kWh_curve_base' in dfm.columns:
                        # indice integrale esistente
                        dfm['diff_abs'] = (dfm['kWh_curve_new'] - dfm['kWh_curve_base']).abs()
                        quality_index = float(dfm['diff_abs'].sum() * 0.25)
                        max_val = float(max(dfm['kWh_curve_base'].max(), 1.0))
                        quality_pct = 100.0 * (1.0 - min(1.0, quality_index / max_val))

                        # nuovo: coerenza energetica
                        energy_base = float(dfm['kWh_curve_base'].sum() * 0.25)
                        energy_new  = float(dfm['kWh_curve_new'].sum() * 0.25)
                        if energy_base > 0:
                            energy_diff_pct = 100.0 * abs(energy_new - energy_base) / energy_base
                            energy_coherence = 100.0 - min(100.0, energy_diff_pct)
                        else:
                            energy_coherence = float('nan')

                        st.success(
                            f"📈 Qualità previsione (integrale): {quality_pct:.1f}%  \n"
                            f"🔋 Coerenza energetica: {energy_coherence:.1f}%  \n"
                            f"(Diff. integrale = {quality_index:.2f} kWh-eq, E_base = {energy_base:.1f} kWh, E_new = {energy_new:.1f} kWh)"
                        )

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=dfm['time'], y=dfm['kWh_curve_base'], mode='lines', name='🔵 Base DOMANI'))
                        fig.add_trace(go.Scatter(x=dfm['time'], y=dfm['kWh_curve_new'], mode='lines', name='🟠 Nuova previsione'))
                        fig.add_trace(go.Scatter(x=dfm['time'], y=dfm['diff_abs'], mode='lines', name='⚙️ Differenza assoluta', line=dict(dash='dot')))
                        fig.update_layout(template='plotly_white', height=420, title='📊 Confronto previsione aggiornata vs base DOMANI', yaxis_title='Potenza (kW)')
                        st.plotly_chart(fig, use_container_width=True)
                        if DEBUG:
                            st.markdown("#### 🔎 Merge debug (prime righe)")
                            st.dataframe(dfm[['time','kWh_curve_base','kWh_curve_new','diff_abs']].head(8))
                    else:
                        st.warning("Colonne richieste non trovate ('kWh_curve_base' / 'kWh_curve_new').")
            except Exception as e:
                st.error(f'Errore confronto: {e}')

# ---- DEBUG FOOTER ---------------------------------------------------------
if DEBUG:
    st.markdown("---"); st.header("🧪 Diagnostica globale (DEBUG)")
    ts = datetime.utcnow().strftime('%Y%m%d')
    log_path = os.path.join(LOG_DIR, f'forecast_log_{ts}.jsonl')
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[-10:]
            st.markdown("#### 📝 Ultimi 10 log"); st.code("".join(lines) if lines else "(log vuoto)")
        except Exception as e:
            st.write(f"Impossibile leggere log: {e}")
    else:
        st.write("Nessun log per oggi.")
    st.markdown("#### 🧰 Session state (chiavi principali)")
    st.json({k: st.session_state.get(k) for k in ['lat','lon','tilt','orient','provider_pref','plant_kw','last_mae','last_r2']})
    model = load_model(); st.markdown("#### 🧠 Stato modello")
    st.json({'exists': model is not None, 'feature_names_in_': getattr(model, 'feature_names_in_', []) if model is not None else None})
