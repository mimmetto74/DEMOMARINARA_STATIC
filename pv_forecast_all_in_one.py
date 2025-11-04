# -*- coding: utf-8 -*-
# Solar Forecast - ROBOTRONIX for IMEPOWER (SECURE build + Validation Tab)
# Includes: login page, embedded Meteomatics credentials, provider fallback, RF model,
# charts, exports, comparison, and NEW Tab 5 validation with Domani baseline.
import os, io, json, math, joblib, requests, numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
import streamlit as st

# ---- Railway-compatible log dir ----
try:
    LOG_DIR = '/tmp/logs' if not os.access('.', os.W_OK) else 'logs'
    os.makedirs(LOG_DIR, exist_ok=True)
except Exception:
    LOG_DIR = 'logs'
    os.makedirs(LOG_DIR, exist_ok=True)

import plotly.graph_objects as go

st.set_page_config(page_title='Solar Forecast - ROBOTRONIX for IMEPOWER', layout='wide')

# ---- Railway server settings ----
_PORT = os.environ.get('PORT')
if _PORT:
    try:
        st.set_option('server.headless', True)
        st.set_option('server.address', '0.0.0.0')
        st.set_option('server.port', int(_PORT))
    except Exception:
        pass

# ---------------------------- SECURITY / LOGIN ---------------------------- #
APP_USER = os.environ.get('APP_USER', 'admin')
APP_PASS = os.environ.get('APP_PASS', 'robotronix')

def render_login():
    st.title('🔐 Accesso richiesto')
    st.write('Inserisci le credenziali per accedere all\'applicazione.')
    u = st.text_input('Username', value='', key='login_user')
    p = st.text_input('Password', value='', type='password', key='login_pass')
    go_btn = st.button('Accedi')
    if go_btn:
        if u == APP_USER and p == APP_PASS:
            st.session_state['authenticated'] = True
            st.success('Accesso effettuato!')
            st.rerun()
        else:
            st.error('❌ Credenziali non valide')

if not st.session_state.get('authenticated', False):
    render_login()
    st.stop()

# ---------------------------- CONFIG ---------------------------- #
DATA_PATH = os.environ.get('PV_DAILY_DATA', 'Dataset_Daily_EnergiaSeparata_2020_2025.csv')
MODEL_PATH = os.environ.get('PV_MODEL_PATH', 'rf_model.joblib')
LOG_DIR = os.environ.get('PV_LOG_DIR', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

DEFAULT_LAT = float(os.environ.get('PV_LAT', '40.643278'))
DEFAULT_LON = float(os.environ.get('PV_LON', '16.986083'))
DEFAULT_TILT = float(os.environ.get('PV_TILT', '10'))
DEFAULT_ORIENT = float(os.environ.get('PV_ORIENT', '180'))
DEFAULT_PLANT_KW = float(os.environ.get('PV_PLANT_KW', '947.25'))

# ----------------------- METEOMATICS CREDENTIALS (embedded) ----------------------- #
MM_USER = 'robotronixsrl_daniello_fabio'
MM_PASS = 'xaRVRh8sa5EV70F0B88u'

# -------------------------- UTILITIES -------------------------- #
def write_log(**kwargs):
    try:
        ts = datetime.utcnow().strftime('%Y%m%d')
        with open(os.path.join(LOG_DIR, f'forecast_log_{ts}.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(kwargs, default=str) + '\n')
    except Exception:
        pass

@st.cache_data(show_spinner=False, ttl=600)
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
    if 'E_INT_Daily_KWh' in df.columns and 'E_INT_Daily_kWh' not in df.columns:
        df = df.rename(columns={'E_INT_Daily_KWh':'E_INT_Daily_kWh'})
    return df

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    obj = joblib.load(MODEL_PATH)
    if isinstance(obj, dict) and 'model' in obj:
        return obj['model']
    return obj

# ----------------------- FETCH PROVIDERS ------------------------ #
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_meteomatics_pt15m(lat, lon, start_iso, end_iso, tilt=None, orient=None):
    # Robust error handling with explicit messages
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
    except requests.Timeout as e:
        raise RuntimeError('Timeout contattando Meteomatics (30s)') from e
    except Exception as e:
        raise RuntimeError(f'Errore rete Meteomatics: {e}') from e
    if r.status_code == 401:
        raise RuntimeError('401 Unauthorized: credenziali Meteomatics non valide')
    if r.status_code == 404:
        raise RuntimeError('404: Nessun dato disponibile per il periodo richiesto')
    if not r.ok:
        raise RuntimeError(f'HTTP {r.status_code}: {r.text[:200]}')
    try:
        j = r.json()
    except Exception as e:
        raise RuntimeError('Risposta non in formato JSON valida da Meteomatics') from e
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
    if df.empty:
        return url, df
    df['time'] = pd.to_datetime(df['time'])
    df = df.groupby('time', as_index=False).mean().sort_values('time')
    for c in ['GlobalRad_W','CloudCover_P','Temp_Air']:
        if c not in df.columns: df[c] = np.nan
    df['provider'] = 'Meteomatics'
    return url, df

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_openmeteo_hourly(lat, lon, start_date, end_date):
    base = 'https://api.open-meteo.com/v1/forecast'
    params = (f'?latitude={lat}&longitude={lon}&hourly=direct_radiation,cloudcover,temperature_2m'
              f'&start_date={start_date}&end_date={end_date}&timezone=UTC')
    url = base + params
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    j = r.json()
    h = j.get('hourly', {})
    times = pd.to_datetime(h.get('time', []))
    if len(times) == 0:
        return url, pd.DataFrame()
    df = pd.DataFrame({'time': times})
    if 'direct_radiation' in h: df['GlobalRad_W'] = pd.Series(h['direct_radiation']).astype(float)
    if 'cloudcover' in h: df['CloudCover_P'] = pd.Series(h['cloudcover']).astype(float)
    if 'temperature_2m' in h: df['Temp_Air'] = pd.Series(h['temperature_2m']).astype(float)
    for c in ['GlobalRad_W','CloudCover_P','Temp_Air']:
        if c not in df.columns: df[c] = np.nan
    df = df.set_index('time').resample('15min').interpolate().reset_index()
    df['provider'] = 'Open-Meteo'
    return url, df
    
# ---------------------- STABILITY UTILS ---------------------- #
BASELINE_PATH = os.path.join(LOG_DIR, 'baseline_today.json')

def load_baseline():
    try:
        if os.path.exists(BASELINE_PATH):
            with open(BASELINE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return None

def save_baseline(payload: dict):
    try:
        with open(BASELINE_PATH, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, default=str, indent=2)
    except Exception as e:
        st.warning(f'Impossibile salvare baseline: {e}')

def send_alert_email(subject: str, body: str, rcpt: str) -> bool:
    """Invia email via SMTP se le variabili d'ambiente sono configurate."""
    host = os.environ.get('SMTP_SERVER')
    port = int(os.environ.get('SMTP_PORT', '587'))
    user = os.environ.get('SMTP_USER')
    pw   = os.environ.get('SMTP_PASS')
    if not (host and user and pw and rcpt):
        st.info('SMTP non configurato (SMTP_SERVER/SMTP_PORT/SMTP_USER/SMTP_PASS o destinatario mancanti).')
        return False
    try:
        from email.mime.text import MIMEText
        import smtplib
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = user
        msg['To'] = rcpt
        with smtplib.SMTP(host, port, timeout=20) as s:
            s.starttls()
            s.login(user, pw)
            s.sendmail(user, [rcpt], msg.as_string())
        return True
    except Exception as e:
        st.error(f'Errore invio email: {e}')
        return False

def compute_energy_for_today(model, lat, lon, tilt, orient, provider_pref, plant_kw):
    """Ricalcola la previsione per Oggi e ritorna (kWh, df, provider, status, url)."""
    dfp, energy, peak_kW, peak_pct, cloud_mean, provider, status, url = forecast_for_day(
        lat=lat, lon=lon, offset_days=0, label='Oggi', model=model,
        tilt=tilt, orient=orient, provider_pref=provider_pref, plant_kw=plant_kw, autosave=False
    )
    return float(energy), dfp, provider, status, url

# ----------------------- MODEL TRAINING (IMPROVED) ------------------------ #
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def train_model():
    df0 = load_data()
    if 'E_INT_Daily_KWh' in df0.columns and 'E_INT_Daily_kWh' not in df0.columns:
        df0 = df0.rename(columns={'E_INT_Daily_KWh':'E_INT_Daily_kWh'})

    # --- Feature engineering migliorata ---
    df = df0.dropna(subset=['E_INT_Daily_kWh', 'G_M0_Wm2']).copy()
    df['month'] = df['Date'].dt.month
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['sin_doy'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['cos_doy'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    df['Temp_Air'] = df.get('Temp_Air', pd.Series(0, index=df.index))
    df['CloudCover_P'] = df.get('CloudCover_P', pd.Series(0, index=df.index))
    df['avg_temp'] = df['Temp_Air'].rolling(window=3, min_periods=1).mean()
    df['cloud_trend'] = df['CloudCover_P'].diff().fillna(0)

    features = [
        'G_M0_Wm2', 'CloudCover_P', 'Temp_Air',
        'sin_doy', 'cos_doy', 'avg_temp', 'cloud_trend'
    ]
    X = df[features].fillna(df[features].mean())
    y = df['E_INT_Daily_kWh']

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    model = RandomForestRegressor(
        n_estimators=400, max_depth=14, random_state=42,
        min_samples_leaf=3, n_jobs=-1
    )
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    mae = float(mean_absolute_error(yte, pred))
    r2 = float(r2_score(yte, pred))

    # --- Salva modello e feature importance ---
    joblib.dump({'model': model, 'features': features}, MODEL_PATH)
    pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)\
        .to_csv(os.path.join(LOG_DIR, 'feature_importances.csv'))
    return mae, r2

# ------------------- DAILY CURVE & FORECAST ------------------- #

def compute_curve_and_daily(df, model, plant_kw):
    """
    Calcola la curva di produzione stimata a partire dai dati meteo.
    Funzionalità:
      - Conversione automatica del tempo in Europe/Rome
      - Mantiene i dati grezzi per diagnostica
      - Allineamento temporale corretto (nessuno shift)
      - Compatibilità completa con Meteomatics / Open-Meteo
      - Normalizzazione per potenza impianto
    """
    import numpy as np
    import pandas as pd
    import pytz

    # --- Validazione iniziale ---
    if df is None or df.empty:
        return df, 0.0, 0.0, 0.0, 0.0

    # --- Parsing e ordinamento ---
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['time']).sort_values('time')

    # ✅ Conversione automatica in ora locale
    try:
        tz_local = pytz.timezone("Europe/Rome")
        # Se il timestamp non ha timezone, assumiamo UTC
        if df['time'].dt.tz is None:
            df['time'] = df['time'].dt.tz_localize(pytz.UTC)
        df['time'] = df['time'].dt.tz_convert(tz_local).dt.tz_localize(None)
    except Exception as e:
        st.warning(f"⚠️ Errore conversione fuso orario: {e}")

    # --- Conversione robusta di colonne numeriche ---
    for col in df.columns:
        if col != 'time':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                pass

    # --- Calcolo irradianza corretta (mantiene dati grezzi) ---
    if 'GlobalRad_W' in df.columns:
        df['rad_corr'] = df['GlobalRad_W'].fillna(0)
    elif 'G_M0_Wm2' in df.columns:
        df['rad_corr'] = df['G_M0_Wm2'].fillna(0)
        df = df.rename(columns={'G_M0_Wm2': 'GlobalRad_W'})
    else:
        df['rad_corr'] = 0.0
        st.warning("⚠️ Nessuna colonna di irradianza trovata nei dati meteo.")

    # --- Resample ogni 15 minuti (preserva media delle numeriche) ---
    df = df.set_index('time').resample('15T').mean(numeric_only=True).reset_index()

    # --- Predizione modello se disponibile ---
    if model is not None:
        model_features = getattr(model, "feature_names_in_", [])
        X = pd.DataFrame()

        # Mappa automatica dei nomi delle feature
        col_map = {
            'G_M0_Wm2': 'GlobalRad_W',
            'GlobalRad_W': 'G_M0_Wm2'
        }

        for feat in model_features:
            if feat in df.columns:
                X[feat] = df[feat]
            elif feat in col_map and col_map[feat] in df.columns:
                X[feat] = df[col_map[feat]]
            else:
                X[feat] = 0.0

        try:
            df['kWh_curve'] = model.predict(X)
        except ValueError as e:
            st.warning(f"⚠️ Mismatch colonne modello: {e}. Riprovo adattando i nomi.")
            cols = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else X.columns
            X = X.reindex(columns=cols, fill_value=0)
            df['kWh_curve'] = model.predict(X)

        # Normalizza rispetto alla potenza impianto
        if df['kWh_curve'].max() > 0:
            df['kWh_curve'] = (
                df['kWh_curve'] / df['kWh_curve'].max()
            ) * plant_kw
    else:
        # --- Fallback proporzionale all'irradianza ---
        ref_col = 'GlobalRad_W' if 'GlobalRad_W' in df.columns else 'rad_corr'
        max_val = max(df[ref_col].max(), 1)
        df['kWh_curve'] = df[ref_col] * (plant_kw / 1000.0) / max_val

    # --- Smussamento centrato (no shift temporale) ---
    df['kWh_curve'] = df['kWh_curve'].rolling(window=3, center=True, min_periods=1).mean()

    # --- Metriche giornaliere ---
    pred_kwh = df['kWh_curve'].sum() * (15 / 60.0)  # ogni passo = 15 min
    peak_kW = df['kWh_curve'].max()
    peak_pct = 100 * peak_kW / plant_kw if plant_kw > 0 else np.nan
    cloud_mean = df['CloudCover_P'].mean() if 'CloudCover_P' in df.columns else np.nan

    return df, float(pred_kwh), float(peak_kW), float(peak_pct), float(cloud_mean)


def forecast_for_day(lat, lon, offset_days, label, model, tilt, orient, provider_pref, plant_kw, autosave=True):
    """Genera la previsione per un giorno specifico (ieri/oggi/domani...)"""
    day = (datetime.now(timezone.utc).date() + timedelta(days=offset_days))
    start_iso = f'{day}T00:00:00Z'
    end_iso = f'{day + timedelta(days=1)}T00:00:00Z'
    provider, status, url, df = 'Meteomatics', 'OK', '', None

    # --- Provider fallback ---
    def try_meteomatics():
        try:
            return (*fetch_meteomatics_pt15m(lat, lon, start_iso, end_iso, tilt=tilt, orient=orient), 'Meteomatics', 'OK')
        except Exception as e:
            return ('', None, 'Meteomatics', f'ERROR: {e}')

    def try_openmeteo():
        try:
            return (*fetch_openmeteo_hourly(lat, lon, str(day), str(day + timedelta(days=1))), 'Open-Meteo', 'OK')
        except Exception as e:
            return ('', None, 'Open-Meteo', f'ERROR: {e}')

    # --- Provider selection ---
    if provider_pref == 'Meteomatics':
        url, df, provider, status = try_meteomatics()
        if df is None or df.empty:
            url, df, provider, status = try_openmeteo()
    elif provider_pref == 'Open-Meteo':
        url, df, provider, status = try_openmeteo()
    else:  # Auto
        url, df, provider, status = try_meteomatics()
        if df is None or df.empty:
            url, df, provider, status = try_openmeteo()

    # --- Se nessun dato disponibile ---
    if df is None or df.empty:
        write_log(
            timestamp=datetime.utcnow().isoformat(),
            day_label=label,
            provider=provider,
            status=status,
            url=url,
            lat=lat,
            lon=lon,
            tilt=tilt,
            orient=orient,
            note='no data'
        )
        return None, 0.0, 0.0, 0.0, float('nan'), provider, status, url

    # ✅ Normalizzazione oraria (UTC → Europe/Rome → naive)
    tz_status = "unknown"
    try:
        import pytz
        tz_local = pytz.timezone("Europe/Rome")
        times = pd.to_datetime(df['time'], errors='coerce')
        if times.dt.tz is None:
            times = times.dt.tz_localize(pytz.UTC)
        times = times.dt.tz_convert(tz_local)
        df['time'] = times.dt.tz_localize(None)
        tz_status = "UTC→Europe/Rome (naive per grafico)"
    except Exception as e:
        tz_status = f"error: {e}"
        st.warning(f"⚠️ Normalizzazione oraria non riuscita: {e}")

    # --- Calcolo curve e parametri ---
    df2, pred_kwh, peak_kW, peak_pct, cloud_mean = compute_curve_and_daily(df, model, plant_kw)

    # --- Log dettagliato ---
    write_log(
        timestamp=datetime.utcnow().isoformat(),
        day_label=label,
        provider=provider,
        status=status,
        url=url,
        lat=lat,
        lon=lon,
        tilt=tilt,
        orient=orient,
        tz_status=tz_status,
        sum_rad_corr=float(df2['rad_corr'].sum() if 'rad_corr' in df2.columns else 0.0),
        pred_kwh=float(pred_kwh),
        peak_kW=float(peak_kW),
        cloud_mean=float(cloud_mean)
    )

    # --- Autosave opzionale ---
    if autosave:
        cols = [c for c in ['time', 'GlobalRad_W', 'CloudCover_P', 'Temp_Air', 'rad_corr', 'kWh_curve'] if c in df2.columns]
        df2[cols].to_csv(os.path.join(LOG_DIR, f'curve_{label.lower()}_15min.csv'), index=False)
        pd.DataFrame([{
            'date': str(day),
            'energy_kWh': float(pred_kwh),
            'peak_kW': float(peak_kW),
            'cloud_mean': float(cloud_mean)
        }]).to_csv(os.path.join(LOG_DIR, f'daily_{label.lower()}.csv'), index=False)

    return df2, pred_kwh, peak_kW, peak_pct, cloud_mean, provider, status, url


# --------------- COMPARISON: FORECAST VS REAL ----------------- #
from sklearn.metrics import mean_absolute_error
def compare_forecast_vs_real(day_label, forecast_df, data_path=DATA_PATH):
    try: df_real = pd.read_csv(data_path, parse_dates=['Date'])
    except Exception as e: st.warning(f'⚠️ Errore caricamento storico: {e}'); return None, None, None
    if forecast_df is None or forecast_df.empty or 'time' not in forecast_df.columns: return None, None, None
    f_date = pd.to_datetime(forecast_df['time'].iloc[0]).date()
    df_real_day = df_real[df_real['Date'].dt.date == f_date]
    if df_real_day.empty: st.info(f'Nessun dato reale per {f_date}'); return None, None, None
    dfp = forecast_df.copy(); dfp['Potenza_kW_prevista'] = dfp['kWh_curve'] * 4
    mae = float('nan'); mape = float('nan')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dfp['time'], y=dfp['Potenza_kW_prevista'], mode='lines', name='Previsione (kW)', line=dict(color='orange', width=2)))
    if 'Potenza_reale_kW' in df_real_day.columns:
        y_true = df_real_day['Potenza_reale_kW'].values[:len(dfp)]
        y_pred = dfp['Potenza_kW_prevista'].values[:len(y_true)]
        mae = float(mean_absolute_error(y_true, y_pred))
        mape = float(np.mean(np.abs((y_true - y_pred)/np.maximum(y_true, 1e-3))) * 100)
        fig.add_trace(go.Scatter(x=dfp['time'], y=y_true, mode='lines', name='Reale (kW)', line=dict(color='blue', width=2)))
    fig.update_layout(title=f'📊 Confronto previsione vs reale – {day_label}', xaxis_title='Ora', yaxis_title='Potenza (kW)', template='plotly_white', height=350)
    return fig, mae, mape

# ----------------------------- UI ----------------------------- #
st.title('☀️ Solar Forecast - ROBOTRONIX for IMEPOWER')
for k,v in {'lat':DEFAULT_LAT,'lon':DEFAULT_LON,'tilt':DEFAULT_TILT,'orient':DEFAULT_ORIENT,'provider_pref':'Auto','plant_kw':DEFAULT_PLANT_KW}.items():
    st.session_state.setdefault(k,v)
tab1, tab2, tab3, tab4, tab5 = st.tabs(['📊 Storico','🧠 Modello','🔮 Previsioni 4 giorni (15 min)','🗺️ Mappa','🛡️ Validazione previsione DOMANI'])

# ---- Selezione metodo previsione ----
_method = st.selectbox("Metodo di previsione", ["Random Forest", "Fisico semplificato", "Ibrido ML + Fisico"], index=0, key="method_select_tab3")

# ---- TAB 1: Storico ---- #
with tab1:
    try:
        df = load_data()
        st.subheader('Storico produzione (kWh) e irradianza (W/m²) — grafici separati')
        c1,c2 = st.columns(2)
        with c1:
            fig1 = go.Figure(); fig1.add_trace(go.Scatter(x=df['Date'], y=df['E_INT_Daily_kWh'], mode='lines', name='Produzione (kWh)', line=dict(color='orange', width=2)))
            fig1.update_layout(template='plotly_white', height=350, title='⚡ Produzione giornaliera (kWh)', xaxis_title='Data', yaxis_title='Energia (kWh)')
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = go.Figure(); fig2.add_trace(go.Scatter(x=df['Date'], y=df['G_M0_Wm2'], mode='lines', name='Irradianza (W/m²)', line=dict(color='deepskyblue', width=2)))
            fig2.update_layout(template='plotly_white', height=350, title='☀️ Irradianza giornaliera (W/m²)', xaxis_title='Data', yaxis_title='W/m²')
            st.plotly_chart(fig2, use_container_width=True)
    except Exception as e: st.error(f'Errore caricamento storico: {e}')

# ---- TAB 2: Modello ---- #
with tab2:
    st.subheader('🧠 Modello di previsione')

    # --- Carica CSV aggiuntivi per ampliare il training ---
    st.markdown("📂 **Carica altri file CSV di dati storici per ampliare il training del modello:**")
    uploaded_files = st.file_uploader(
        "Trascina qui uno o più file CSV di dati storici aggiuntivi",
        type=['csv'],
        accept_multiple_files=True
    )

    if uploaded_files:
        df_base = load_data()
        dfs = [df_base]
        for f in uploaded_files:
            try:
                df_new = pd.read_csv(f, parse_dates=['Date'])
                dfs.append(df_new)
                st.success(f"✅ File aggiunto: {f.name} ({len(df_new)} righe)")
            except Exception as e:
                st.error(f"Errore caricamento {f.name}: {e}")

        df_merged = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['Date'])
        merged_path = os.path.join(LOG_DIR, 'merged_dataset.csv')
        df_merged.to_csv(merged_path, index=False)
        st.info(f"📊 Dataset unificato salvato in: `{merged_path}` — {len(df_merged)} righe totali.")
        st.session_state['custom_dataset'] = merged_path

    # --- Pulsante di addestramento ---
    c1, c2, c3 = st.columns([1, 1, 2])
    if c1.button('Addestra / Riaddestra modello', use_container_width=True):
        data_path = st.session_state.get('custom_dataset', DATA_PATH)
        if os.path.exists(data_path):
            df = pd.read_csv(data_path, parse_dates=['Date'])
            df.to_csv(DATA_PATH, index=False)
        mae, r2 = train_model()
        st.session_state['last_mae'] = mae
        st.session_state['last_r2'] = r2
        st.success(f'✅ Modello addestrato!  MAE: {mae:.2f} | R²: {r2:.3f}')

    # --- Visualizzazione risultati modello ---
    if os.path.exists(MODEL_PATH):
        model = load_model()
        dfm = load_data()
        if 'E_INT_Daily_KWh' in dfm.columns and 'E_INT_Daily_kWh' not in dfm.columns:
            dfm = dfm.rename(columns={'E_INT_Daily_KWh': 'E_INT_Daily_kWh'})
        for col in ['CloudCover_P', 'Temp_Air']:
            if col not in dfm.columns:
                dfm[col] = np.nan
        dfm = dfm.dropna(subset=['E_INT_Daily_kWh', 'G_M0_Wm2'])

        # --- Ricrea le feature usate nel modello ---
        dfm['month'] = dfm['Date'].dt.month
        dfm['dayofyear'] = dfm['Date'].dt.dayofyear
        dfm['sin_doy'] = np.sin(2 * np.pi * dfm['dayofyear'] / 365)
        dfm['cos_doy'] = np.cos(2 * np.pi * dfm['dayofyear'] / 365)
        dfm['avg_temp'] = dfm['Temp_Air'].rolling(window=3, min_periods=1).mean()
        dfm['cloud_trend'] = dfm['CloudCover_P'].diff().fillna(0)

        # --- Usa le stesse feature del modello addestrato ---
        features = getattr(model, "feature_names_in_", [
             'G_M0_Wm2', 'CloudCover_P', 'Temp_Air',
             'sin_doy', 'cos_doy', 'avg_temp', 'cloud_trend'
        ])
        Xp = dfm.reindex(columns=features, fill_value=0)

        dfm['Predetto'] = model.predict(Xp)


        # Grafico Reale vs Predetto
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dfm['E_INT_Daily_kWh'], y=dfm['Predetto'],
            mode='markers', marker=dict(size=5, opacity=0.6), name='Punti'
        ))
        minv = float(min(dfm['E_INT_Daily_kWh'].min(), dfm['Predetto'].min()))
        maxv = float(max(dfm['E_INT_Daily_kWh'].max(), dfm['Predetto'].max()))
        fig.add_trace(go.Scatter(
            x=[minv, maxv], y=[minv, maxv],
            mode='lines', line=dict(color='orange', dash='dash'), name='y = x'
        ))
        fig.update_layout(
            title='📈 Reale vs Predetto (kWh/giorno)',
            xaxis_title='Reale (kWh)',
            yaxis_title='Predetto (kWh)',
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# ---- TAB 3: Previsioni (4 giorni) ---- #
with tab3:
    st.subheader('🔮 Previsioni 4 giorni (15 min)')

    # --- Input parametri ---
    colA, colB, colC, colD = st.columns(4)
    st.session_state['lat'] = colA.number_input('Lat', value=float(st.session_state['lat']), step=0.0001, format='%.6f')
    st.session_state['lon'] = colB.number_input('Lon', value=float(st.session_state['lon']), step=0.0001, format='%.6f')
    st.session_state['tilt'] = colC.number_input('Tilt (°)', value=float(st.session_state['tilt']), step=1.0)
    st.session_state['orient'] = colD.number_input('Orient (°)', value=float(st.session_state['orient']), step=1.0)

    colE, colF = st.columns(2)
    st.session_state['provider_pref'] = colE.selectbox('Provider', ['Auto', 'Meteomatics', 'Open-Meteo'], index=0)
    st.session_state['plant_kw'] = colF.number_input('Taglia impianto (kW)', value=float(st.session_state['plant_kw']), step=10.0)

    model = load_model()
    if model is None:
        st.warning('⚠️ Modello non addestrato. Vai al tab "🧠 Modello".')
    else:
        if st.button('Calcola previsioni (Ieri/Oggi/Domani/Dopodomani)', use_container_width=True):
            for label, off in [('Ieri', -1), ('Oggi', 0), ('Domani', 1), ('Dopodomani', 2)]:
                try:
                    dfp, energy, peak_kW, peak_pct, cloud_mean, provider, status, url = forecast_for_day(
                        lat=st.session_state['lat'], lon=st.session_state['lon'],
                        offset_days=off, label=label, model=model,
                        tilt=st.session_state['tilt'], orient=st.session_state['orient'],
                        provider_pref=st.session_state['provider_pref'],
                        plant_kw=st.session_state['plant_kw'],
                        autosave=False
                    )
                except Exception as e:
                    st.error(f"Errore durante la previsione per {label}: {e}")
                    continue

                # --- Controllo dati ---
                if dfp is None or dfp.empty:
                    st.warning(f"Nessun dato disponibile per {label}.")
                    continue

                # --- Normalizza colonne temporali ---
                import pytz
                dfp = dfp.copy()
                if 'time' not in dfp.columns:
                    for alt in ['Date', 'datetime', 'timestamp']:
                        if alt in dfp.columns:
                            dfp.rename(columns={alt: 'time'}, inplace=True)
                            break
                if 'time' not in dfp.columns:
                    st.error("❌ Colonna 'time' mancante nel dataset.")
                    continue

                dfp['time'] = pd.to_datetime(dfp['time'], errors='coerce')
                dfp = dfp.dropna(subset=['time']).sort_values('time')
                if dfp['time'].dt.tz is None:
                    dfp['time'] = dfp['time'].dt.tz_localize(pytz.UTC)
                dfp['time'] = dfp['time'].dt.tz_convert("Europe/Rome").dt.tz_localize(None)
                # --- Prepara le nuove feature per il modello ML ---
                if model is not None and _method == "Random Forest":
                   try:
                        dfp['month'] = dfp['time'].dt.month
                        dfp['dayofyear'] = dfp['time'].dt.dayofyear
                        dfp['sin_doy'] = np.sin(2 * np.pi * dfp['dayofyear'] / 365)
                        dfp['cos_doy'] = np.cos(2 * np.pi * dfp['dayofyear'] / 365)
                        dfp['avg_temp'] = dfp['Temp_Air'].rolling(window=3, min_periods=1).mean()
                        dfp['cloud_trend'] = dfp['CloudCover_P'].diff().fillna(0)
                    except Exception as e:
                        st.warning(f"⚠️ Errore creazione feature avanzate: {e}")

                # --- Applica metodo selezionato ---
                if _method == "Fisico semplificato":
                    dfp, energy, peak_kW, peak_pct, cloud_mean = forecast_physical(dfp, st.session_state['plant_kw'])
                elif _method == "Ibrido ML + Fisico":
                    dfp, energy, peak_kW, peak_pct, cloud_mean = forecast_hybrid(dfp, model, st.session_state['plant_kw'])
                else:
                    dfp, energy, peak_kW, peak_pct, cloud_mean = compute_curve_and_daily(dfp, model, st.session_state['plant_kw'])

                # --- Salvataggio base DOMANI ---
                if label == 'Domani':
                    try:
                        base_path = os.path.join(LOG_DIR, "forecast_domani_base.csv")
                        cols = [c for c in ['time','GlobalRad_W','CloudCover_P','Temp_Air','rad_corr','kWh_curve'] if c in dfp.columns]
                        dfp[cols].to_csv(base_path, index=False)
                        st.info(f"📦 Base di confronto DOMANI salvata: {base_path}")
                    except Exception as e:
                        st.warning(f"Impossibile salvare la base DOMANI: {e}")

                # --- Grafico principale ---
                import plotly.graph_objects as go
                fig = go.Figure()
                if 'GlobalRad_W' in dfp.columns:
                    fig.add_trace(go.Scatter(
                        x=dfp['time'], y=dfp['GlobalRad_W'],
                        mode='lines', name='☀️ Irradianza (W/m²)', line=dict(color='royalblue', width=2)
                    ))
                if 'kWh_curve' in dfp.columns:
                    fig.add_trace(go.Scatter(
                        x=dfp['time'], y=dfp['kWh_curve'],
                        mode='lines', name='⚡ Produzione stimata (kW)', line=dict(color='orange', width=3)
                    ))

                # --- Picchi e allineamento ---
                try:
                    idx_rad = dfp['GlobalRad_W'].idxmax() if 'GlobalRad_W' in dfp.columns else None
                    idx_prod = dfp['kWh_curve'].idxmax() if 'kWh_curve' in dfp.columns else None
                    if idx_rad is not None and idx_prod is not None:
                        t_rad = dfp.loc[idx_rad, 'time']
                        t_prod = dfp.loc[idx_prod, 'time']
                        delta_min = abs((t_rad - t_prod).total_seconds()) / 60
                        if delta_min > 30:
                            st.warning(f"⚠️ Differenza picchi: {int(delta_min)} minuti (☀️ {t_rad.strftime('%H:%M')} vs ⚡ {t_prod.strftime('%H:%M')})")
                        else:
                            st.success(f"✅ Picchi allineati ({t_prod.strftime('%H:%M')})")
                except Exception as e:
                    st.warning(f"Errore calcolo picchi: {e}")

                # --- Linea ora attuale ---
                try:
                    import pandas as pd
                    import pytz
                    from datetime import datetime

                    now_local = datetime.now(pytz.timezone("Europe/Rome")).replace(tzinfo=None)
                    dfp['time'] = pd.to_datetime(dfp['time'], errors='coerce')

                    if dfp['time'].min() <= now_local <= dfp['time'].max():
                        try:
                            # FIX linea oraria robusta
                            fig.add_vline(
                                x=pd.Timestamp(now_local),
                                line_width=2,
                                line_dash="dot",
                                line_color="red",
                                annotation_text=f"🕒 Ora attuale {now_local.strftime('%H:%M')}",
                                annotation_position="top right"
                            )
                        except Exception:
                            # Fallback se add_vline fallisce
                            xval = pd.Timestamp(now_local)
                            fig.add_shape(
                                type="line",
                                x0=xval, x1=xval,
                                y0=0, y1=1,
                                xref="x", yref="paper",
                                line=dict(width=2, dash="dot", color="red")
                            )
                            fig.add_annotation(
                                x=xval, y=1, xref="x", yref="paper",
                                text=f"🕒 Ora attuale {now_local.strftime('%H:%M')}",
                                showarrow=False, yshift=10
                            )
                except Exception as e:
                    st.warning(f"Errore linea oraria: {e}")

                # --- Layout e risultati ---
                fig.update_layout(
                    title=f"📊 Andamento previsto — {label}",
                    xaxis_title="Ora locale (Europe/Rome)",
                    yaxis_title="Potenza / Irradianza",
                    template="plotly_white", height=400, hovermode="x unified"
                )
                st.markdown(f"### **{label}**")
                st.caption(f"Provider: {provider} | Stato: {status} | Energia: {energy:.1f} kWh | Picco: {peak_kW:.1f} kW ({peak_pct:.1f}%)")
                st.plotly_chart(fig, use_container_width=True)

                # --- Download CSV ---
                csv_data = dfp.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"📥 Scarica CSV previsione {label}",
                    data=csv_data,
                    file_name=f"previsione_{label.lower()}.csv",
                    mime="text/csv",
                    key=f"download_{label.lower()}",
                    use_container_width=True
                )

# ============================================================
# ⚙️ Metodo fisico semplificato
# ============================================================
def forecast_physical(df, plant_kw, eff=0.17):
    if df is None or df.empty:
        return df, 0, 0, 0, 0
    df = df.copy()
    if 'GlobalRad_W' not in df.columns:
        st.warning("⚠️ Colonna 'GlobalRad_W' mancante nei dati meteo.")
        return df, 0, 0, 0, 0
    df['Energy_kWh'] = df['GlobalRad_W'] * eff * (15 * 60) / 3.6e6 * (plant_kw / 1000)
    df['kWh_curve'] = df['Energy_kWh'] / (15 / 60)
    total = float(df['Energy_kWh'].sum())
    peak_kW = float(df['kWh_curve'].max())
    peak_pct = float(100 * peak_kW / plant_kw)
    cloud_mean = float(df['CloudCover_P'].mean()) if 'CloudCover_P' in df.columns else 0.0
    return df, total, peak_kW, peak_pct, cloud_mean


# ============================================================
# 🌦️ Metodo ibrido (ML + fisico)
# ============================================================
def forecast_hybrid(df, model, plant_kw, w_ml=0.7):
    if df is None or df.empty:
        return df, 0, 0, 0, 0
    df_ml, _, _, _, cloud = compute_curve_and_daily(df, model, plant_kw)
    df_phys, _, _, _, _ = forecast_physical(df, plant_kw)
    if df_ml.empty or df_phys.empty:
        return df, 0, 0, 0, 0
    n = min(len(df_ml), len(df_phys))
    df_h = df_ml.iloc[:n].copy()
    df_h['kWh_curve'] = w_ml * df_ml['kWh_curve'].iloc[:n] + (1 - w_ml) * df_phys['kWh_curve'].iloc[:n]
    total = float(df_h['kWh_curve'].sum() * (15 / 60))
    peak = float(df_h['kWh_curve'].max())
    peak_pct = float(100 * peak / plant_kw)
    return df_h, total, peak, peak_pct, cloud

# ---- TAB 4: Mappa satellitare (Folium, senza chiavi API) ---- #
with tab4:
    st.subheader("🛰️ Localizzazione impianto fotovoltaico (vista satellitare)")
    st.write("Visualizzazione satellitare ad alta risoluzione tramite Esri World Imagery.")

    # Recupera coordinate correnti dalla sessione
    lat = float(st.session_state.get('lat', DEFAULT_LAT))
    lon = float(st.session_state.get('lon', DEFAULT_LON))

    # Mostra coordinate testuali
    st.markdown(f"**Coordinate attuali:** 🌍 {lat:.6f}, {lon:.6f}")

    # Import Folium e Streamlit-Folium
    from streamlit_folium import st_folium
    import folium

    # Crea la mappa satellitare
    m = folium.Map(
        location=[lat, lon],
        zoom_start=17,
        tiles='Esri.WorldImagery',  # layer satellitare ad alta risoluzione
        attr='Tiles © Esri'
    )

    # Aggiungi marker dell’impianto
    folium.Marker(
        [lat, lon],
        popup=f"Impianto fotovoltaico<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}",
        tooltip="Impianto fotovoltaico",
        icon=folium.Icon(color='orange', icon='bolt', prefix='fa')
    ).add_to(m)

    # Mostra la mappa nella pagina Streamlit
    st_folium(m, width=900, height=500)


# ---- TAB 5: Validazione con dati reali (Analysis) ---- #
with tab5:
    st.subheader("🛡️ Validazione previsioni con dati reali (Analysis)")
    st.caption("Confronto tra previsione salvata e produzione reale estratta dal modello Teseo.")

    forecast_path = os.path.join(LOG_DIR, "forecast_domani_base.csv")
    uploaded_file = st.file_uploader("📂 Carica file dati reali (analisis.csv o .xlsx)", type=["csv", "xls", "xlsx"])

    if uploaded_file is None:
        st.info("📥 Carica il file dei dati reali (ad esempio 'analisis.csv').")
    elif not os.path.exists(forecast_path):
        st.warning("⚠️ Nessuna previsione trovata. Prima salva la previsione di domani dal Tab 3.")
    else:
        try:
            # --- Legge file reale (robusto per CSV con ; e virgola decimale) ---
            def _read_real_file(uploaded_file):
                name = uploaded_file.name.lower()

                # 1) Excel diretto
                if name.endswith((".xls", ".xlsx")):
                    df = pd.read_excel(uploaded_file)
                else:
                    # 2) CSV – proviamo vari separator/decimal
                    tried = [
                        dict(sep=None, engine="python"),
                        dict(sep=";", decimal=","),
                        dict(sep=";", decimal="."),
                        dict(sep=",", decimal=","),
                        dict(sep=",", decimal=".")
                    ]
                    last_err = None
                    for opts in tried:
                        try:
                            uploaded_file.seek(0)
                            df = pd.read_csv(
                                uploaded_file,
                                **opts,
                                skip_blank_lines=True,
                                comment="#"
                            )
                            break
                        except Exception as e:
                            last_err = e
                            continue
                    else:
                        raise last_err

                # 3) Normalizza nomi colonne
                df.columns = [c.strip().lower() for c in df.columns]

                # 4) Se è una sola colonna “grezza”, prova a splittare
                if df.shape[1] == 1 and df.iloc[:, 0].dtype == object:
                    s = df.iloc[:, 0].astype(str)
                    df = s.str.replace("\t", ";", regex=False).str.split(r"[;,]", expand=True)
                    df.columns = [f"col{i}" for i in range(df.shape[1])]

                # 5) Scegli la colonna reale in kW
                if "meteorologica" in df.columns:
                    serie = df["meteorologica"].astype(str)
                else:
                    num_cols = []
                    for c in df.columns:
                        vals = pd.to_numeric(df[c], errors="coerce")
                        if vals.notna().sum() > 0:
                            num_cols.append(c)
                    if not num_cols:
                        raise ValueError("Nessuna colonna numerica trovata nel file reale.")
                    serie = df[num_cols[0]].astype(str)

                # 6) Pulisci numeri: rimuovi separatore migliaia e usa punto decimale
                serie = (serie.str.replace(".", "", regex=False)
                               .str.replace(",", ".", regex=False))
                serie = pd.to_numeric(serie, errors="coerce")

                out = pd.DataFrame({"Reale_kW": serie})
                out = out.dropna(subset=["Reale_kW"]).reset_index(drop=True)
                return out

            df_real = _read_real_file(uploaded_file)


            # --- Legge previsione salvata ---
            df_fore = pd.read_csv(forecast_path)
            if "kWh_curve" not in df_fore.columns:
                st.error("❌ Colonna 'kWh_curve' non trovata nella previsione salvata.")
                st.stop()

            df_fore = df_fore.rename(columns={"kWh_curve": "Previsto_kW"})
            df_fore = df_fore.dropna(subset=["Previsto_kW"]).reset_index(drop=True)

            # --- Allineamento automatico (stesso numero di punti) ---
            n = min(len(df_real), len(df_fore))
            if n == 0:
                st.error("Nessun punto confrontabile tra reale e previsto.")
                st.stop()

            df_merge = pd.DataFrame({
                "Previsto_kW": df_fore["Previsto_kW"].iloc[:n].values,
                "Reale_kW": df_real["Reale_kW"].iloc[:n].values
            })

            # --- Metriche ---
            from sklearn.metrics import mean_absolute_error, r2_score
            mae = float(mean_absolute_error(df_merge["Reale_kW"], df_merge["Previsto_kW"]))
            mape = float(np.mean(np.abs((df_merge["Reale_kW"] - df_merge["Previsto_kW"]) / np.maximum(df_merge["Reale_kW"], 1e-3))) * 100)
            r2 = float(r2_score(df_merge["Reale_kW"], df_merge["Previsto_kW"]))

            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"{mae:.3f} kW")
            c2.metric("MAPE", f"{mape:.2f} %")
            c3.metric("R²", f"{r2:.3f}")

            # --- Grafico comparativo ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=df_merge["Reale_kW"], mode="lines", name="Reale (Teseo)"))
            fig.add_trace(go.Scatter(y=df_merge["Previsto_kW"], mode="lines", name="Previsto (Modello)"))
            fig.update_layout(
                title="📊 Confronto Reale (Teseo) vs Previsto (Modello)",
                xaxis_title="Indice temporale (punti)",
                yaxis_title="Potenza (kW)",
                template="plotly_white",
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Errore durante la validazione: {e}")

# ---- Added forecast methods (physical + hybrid) ----

def forecast_physical(df, plant_kw, eff=0.17):
    import pandas as pd, numpy as np
    if df is None or df.empty or 'GlobalRad_W' not in df.columns:
        return df, 0, 0, 0, 0
    df = df.copy()
    df['Energy_kWh'] = df['GlobalRad_W'].fillna(0) * eff * (15*60) / 3.6e6 * (plant_kw/1000)
    df['kWh_curve'] = df['Energy_kWh'] / (15/60)
    total = float(df['Energy_kWh'].sum())
    peak_kW = float(df['kWh_curve'].max())
    peak_pct = float(100 * peak_kW / plant_kw) if plant_kw else 0.0
    cloud_mean = float(df['CloudCover_P'].mean()) if 'CloudCover_P' in df.columns else 0.0
    return df, total, peak_kW, peak_pct, cloud_mean

def forecast_hybrid(df, model, plant_kw, w_ml=0.7):
    if df is None or df.empty: return df, 0, 0, 0, 0
    df_ml,_,_,_,cloud = compute_curve_and_daily(df, model, plant_kw)
    df_phys,_,_,_,_ = forecast_physical(df, plant_kw)
    if df_ml is None or df_ml.empty or df_phys is None or df_phys.empty:
        return df, 0, 0, 0, 0
    n = min(len(df_ml), len(df_phys))
    df_h = df_ml.iloc[:n].copy()
    df_h['kWh_curve'] = w_ml * df_ml['kWh_curve'].iloc[:n].values + (1 - w_ml) * df_phys['kWh_curve'].iloc[:n].values
    total = float(df_h['kWh_curve'].sum() * (15/60))
    peak = float(df_h['kWh_curve'].max())
    pct = float(100 * peak / plant_kw) if plant_kw else 0.0
    return df_h, total, peak, pct, cloud




