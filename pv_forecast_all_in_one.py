# -*- coding: utf-8 -*-
# Solar Forecast - ROBOTRONIX for IMEPOWER (SECURE build)
# Includes: login page, embedded Meteomatics credentials, provider fallback, RF model, charts, exports, comparison.
import os, io, json, math, joblib, requests, numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title='Solar Forecast - ROBOTRONIX for IMEPOWER (SECURE)', layout='wide')

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

# ----------------------- MODEL TRAINING ------------------------ #
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def train_model():
    df0 = load_data()
    if 'E_INT_Daily_KWh' in df0.columns and 'E_INT_Daily_kWh' not in df0.columns:
        df0 = df0.rename(columns={'E_INT_Daily_KWh':'E_INT_Daily_kWh'})
    for col in ['CloudCover_P','Temp_Air']:
        if col not in df0.columns: df0[col] = np.nan
    df = df0.dropna(subset=['E_INT_Daily_kWh','G_M0_Wm2']).copy()
    X = df[['G_M0_Wm2','CloudCover_P','Temp_Air']].fillna(df[['G_M0_Wm2','CloudCover_P','Temp_Air']].mean())
    y = df['E_INT_Daily_kWh']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    mae = float(mean_absolute_error(yte, pred))
    r2 = float(r2_score(yte, pred))
    joblib.dump({'model': model, 'features': X.columns.tolist()}, MODEL_PATH)
    pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).to_csv(os.path.join(LOG_DIR,'feature_importances.csv'))
    return mae, r2

# ------------------- DAILY CURVE & FORECAST ------------------- #
def compute_curve_and_daily(df, model, plant_kw):
    if df is None or df.empty:
        return None, 0.0, 0.0, 0.0, float('nan')
    df = df.copy().sort_values('time')
    for col in ['GlobalRad_W','CloudCover_P','Temp_Air']:
        if col not in df.columns: df[col] = np.nan
    df['GlobalRad_W'] = df['GlobalRad_W'].clip(lower=0)
    df['CloudCover_P'] = df['CloudCover_P'].clip(lower=0, upper=100)
    df['Temp_Air'] = df['Temp_Air'].fillna(df['Temp_Air'].mean())
    df['rad_corr'] = df['GlobalRad_W'] * (1 - df['CloudCover_P']/100.0)
    sum_rad = float(df['rad_corr'].sum())
    cloud_mean = float(df['CloudCover_P'].mean())
    temp_mean = float(df['Temp_Air'].mean())
    pred_kwh = float(model.predict([[sum_rad, cloud_mean, temp_mean]])[0]) if sum_rad>0 else 0.0
    df['kWh_curve'] = pred_kwh * (df['rad_corr']/sum_rad) if sum_rad>0 else 0.0
    peak_kW = float(df['kWh_curve'].max()*4.0)
    peak_pct = float((peak_kW/plant_kw*100.0) if plant_kw>0 else 0.0)
    return df, pred_kwh, peak_kW, peak_pct, cloud_mean

def forecast_for_day(lat, lon, offset_days, label, model, tilt, orient, provider_pref, plant_kw, autosave=True):
    day = (datetime.now(timezone.utc).date() + timedelta(days=offset_days))
    start_iso = f'{day}T00:00:00Z'
    end_iso = f'{day + timedelta(days=1)}T00:00:00Z'
    provider, status, url, df = 'Meteomatics', 'OK', '', None
    def try_meteomatics():
        try: return (*fetch_meteomatics_pt15m(lat, lon, start_iso, end_iso, tilt=tilt, orient=orient), 'Meteomatics', 'OK')
        except Exception as e: return ('', None, 'Meteomatics', f'ERROR: {e}')
    def try_openmeteo():
        try: return (*fetch_openmeteo_hourly(lat, lon, str(day), str(day+timedelta(days=1))), 'Open-Meteo', 'OK')
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
        write_log(timestamp=datetime.utcnow().isoformat(), day_label=label, provider=provider, status=status, url=url, lat=lat, lon=lon, tilt=tilt, orient=orient, note='no data')
        return None, 0.0, 0.0, 0.0, float('nan'), provider, status, url
    df2, pred_kwh, peak_kW, peak_pct, cloud_mean = compute_curve_and_daily(df, model, plant_kw)
    write_log(timestamp=datetime.utcnow().isoformat(), day_label=label, provider=provider, status=status, url=url, lat=lat, lon=lon, tilt=tilt, orient=orient, sum_rad_corr=float(df2['rad_corr'].sum()), pred_kwh=float(pred_kwh), peak_kW=float(peak_kW), cloud_mean=float(cloud_mean))
    if autosave:
        cols = [c for c in ['time','GlobalRad_W','CloudCover_P','Temp_Air','rad_corr','kWh_curve'] if c in df2.columns]
        df2[cols].to_csv(os.path.join(LOG_DIR, f'curve_{label.lower()}_15min.csv'), index=False)
        pd.DataFrame([{'date': str(day), 'energy_kWh': float(pred_kwh), 'peak_kW': float(peak_kW), 'cloud_mean': float(cloud_mean)}]).to_csv(os.path.join(LOG_DIR, f'daily_{label.lower()}.csv'), index=False)
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
st.title('☀️ Solar Forecast - ROBOTRONIX for IMEPOWER (SECURE)')
for k,v in {'lat':DEFAULT_LAT,'lon':DEFAULT_LON,'tilt':DEFAULT_TILT,'orient':DEFAULT_ORIENT,'provider_pref':'Auto','plant_kw':DEFAULT_PLANT_KW}.items():
    st.session_state.setdefault(k,v)
tab1, tab2, tab3, tab4 = st.tabs(['📊 Storico','🧠 Modello','🔮 Previsioni 4 giorni (15 min)','🗺️ Mappa'])

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
    st.subheader('🧠 Modello di previsione – Random Forest (multivariato)')
    c1,c2,c3 = st.columns([1,1,2])
    if c1.button('Addestra / Riaddestra modello', use_container_width=True):
        mae,r2 = train_model(); st.session_state['last_mae']=mae; st.session_state['last_r2']=r2
        st.success(f'✅ Modello addestrato!  MAE: {mae:.2f} | R²: {r2:.3f}')
    if os.path.exists(MODEL_PATH):
        model = load_model(); dfm = load_data()
        if 'E_INT_Daily_KWh' in dfm.columns and 'E_INT_Daily_kWh' not in dfm.columns: dfm = dfm.rename(columns={'E_INT_Daily_KWh':'E_INT_Daily_kWh'})
        for col in ['CloudCover_P','Temp_Air']:
            if col not in dfm.columns: dfm[col]=np.nan
        dfm = dfm.dropna(subset=['E_INT_Daily_kWh','G_M0_Wm2'])
        Xp = dfm[['G_M0_Wm2','CloudCover_P','Temp_Air']].fillna(dfm[['G_M0_Wm2','CloudCover_P','Temp_Air']].mean())
        dfm['Predetto'] = model.predict(Xp)
        fig = go.Figure(); fig.add_trace(go.Scatter(x=dfm['E_INT_Daily_kWh'], y=dfm['Predetto'], mode='markers', marker=dict(size=5, opacity=0.6), name='Punti'))
        minv = float(min(dfm['E_INT_Daily_kWh'].min(), dfm['Predetto'].min())); maxv = float(max(dfm['E_INT_Daily_kWh'].max(), dfm['Predetto'].max()))
        fig.add_trace(go.Scatter(x=[minv,maxv], y=[minv,maxv], mode='lines', line=dict(color='orange', dash='dash'), name='y = x'))
        fig.update_layout(title='📈 Reale vs Predetto (kWh/giorno)', xaxis_title='Reale (kWh)', yaxis_title='Predetto (kWh)', template='plotly_white', height=400)
        st.plotly_chart(fig, use_container_width=True)
        fi_path = os.path.join(LOG_DIR, 'feature_importances.csv')
        if os.path.exists(fi_path):
            feat = pd.read_csv(fi_path, index_col=0);
            if feat.shape[1]==1: feat.columns=['importance']
            figf = go.Figure(); figf.add_trace(go.Bar(x=feat.index, y=feat.iloc[:,0]))
            figf.update_layout(title='🔍 Importanza variabili', xaxis_title='Feature', yaxis_title='Importanza', template='plotly_white', height=350)
            st.plotly_chart(figf, use_container_width=True)
    else: st.info('Addestra il modello per abilitare le previsioni.')

# ---- TAB 3: Previsioni (4 giorni) ---- #
with tab3:
    st.subheader('Previsioni (PT15M, tilt/orient, provider toggle)')
    colA,colB,colC,colD = st.columns(4)
    st.session_state['lat'] = colA.number_input('Lat', value=float(st.session_state['lat']), step=0.0001, format='%.6f')
    st.session_state['lon'] = colB.number_input('Lon', value=float(st.session_state['lon']), step=0.0001, format='%.6f')
    st.session_state['tilt'] = colC.number_input('Tilt (°)', value=float(st.session_state['tilt']), step=1.0)
    st.session_state['orient'] = colD.number_input('Orient (°)', value=float(st.session_state['orient']), step=1.0)
    colE,colF = st.columns(2)
    st.session_state['provider_pref'] = colE.selectbox('Provider', ['Auto','Meteomatics','Open-Meteo'], index=0)
    st.session_state['plant_kw'] = colF.number_input('Taglia impianto (kW)', value=float(st.session_state['plant_kw']), step=10.0)
    model = load_model()
    if model is None:
        st.warning('⚠️ Modello non addestrato. Vai al tab \'Modello\'.')
    else:
        if st.button('Calcola previsioni (Ieri/Oggi/Domani/Dopodomani)'):
            results = {}
            for label,off in [('Ieri',-1),('Oggi',0),('Domani',1),('Dopodomani',2)]:
                dfp, energy, peak_kW, peak_pct, cloud_mean, provider, status, url = forecast_for_day(
                    lat=st.session_state['lat'], lon=st.session_state['lon'],
                    offset_days=off, label=label, model=model,
                    tilt=st.session_state['tilt'], orient=st.session_state['orient'],
                    provider_pref=st.session_state['provider_pref'], plant_kw=st.session_state['plant_kw'],
                    autosave=False)
                results[label] = dfp
                st.markdown(f'### {label}')
                st.caption(f'Provider: {provider} | Stato: {status}')
                # st.code(url or '', language='text')
                if dfp is not None and not dfp.empty:
                    # 📈 Anteprima diagnostica dati provider
                    with st.expander('📈 Meteomatics/Open-Meteo: anteprima dati grezzi'):
                        try:
                            fig_diag = go.Figure()
                            fig_diag.add_trace(go.Scatter(x=dfp['time'], y=dfp['GlobalRad_W'], name='GlobalRad_W', mode='lines'))
                            if 'CloudCover_P' in dfp.columns: fig_diag.add_trace(go.Scatter(x=dfp['time'], y=dfp['CloudCover_P'], name='CloudCover_P', mode='lines'))
                            if 'Temp_Air' in dfp.columns: fig_diag.add_trace(go.Scatter(x=dfp['time'], y=dfp['Temp_Air'], name='Temp_Air', mode='lines'))
                            fig_diag.update_layout(template='plotly_white', height=220, margin=dict(l=10,r=10,t=30,b=10))
                            st.plotly_chart(fig_diag, use_container_width=True)
                        except Exception as e:
                            st.info(f'Diagnostica non disponibile: {e}')
                    dfp = dfp.copy(); dfp['Potenza_kW'] = dfp['kWh_curve'] * 4
                    fig = go.Figure(); fig.add_trace(go.Scatter(x=dfp['time'], y=dfp['Potenza_kW'], mode='lines', fill='tozeroy', name='Potenza prevista (kW)', line=dict(color='orange', width=2), fillcolor='rgba(255,165,0,0.3)'))
                    fig.add_hline(y=peak_kW, line_dash='dash', line_color='red', annotation_text=f'Picco: {peak_kW:.1f} kW', annotation_position='top left')
                    fig.update_layout(template='plotly_white', height=300, xaxis_title='Ora', yaxis_title='kW')
                    st.plotly_chart(fig, use_container_width=True)
                    st.info(f'**Energia stimata {label}**: {energy:.1f} kWh | **Picco stimato**: {peak_kW:.1f} kW | **% targa**: {peak_pct:.1f}% | **Nuvolosità media**: {cloud_mean:.0f}%')
                    csv_buf = io.StringIO();
                    cols_ok = [c for c in ['time','GlobalRad_W','CloudCover_P','Temp_Air','rad_corr','kWh_curve','Potenza_kW'] if c in dfp.columns]
                    dfp[cols_ok].to_csv(csv_buf, index=False)
                    st.download_button(f'⬇️ Scarica curva 15-min ({label})', csv_buf.getvalue(), file_name=f'curve_{label.lower()}_15min.csv', mime='text/csv')
                    daily_buf = io.StringIO();
                    pd.DataFrame([{'date': str(datetime.now(timezone.utc).date() + timedelta(days=off)), 'energy_kWh': energy, 'peak_kW': peak_kW, 'cloud_mean': cloud_mean}]).to_csv(daily_buf, index=False)
                    st.download_button(f'⬇️ Scarica aggregato ({label})', daily_buf.getvalue(), file_name=f'daily_{label.lower()}.csv', mime='text/csv')
                    with st.expander(f'Confronto con reale – {label}'):
                        fig_real, mae, mape = compare_forecast_vs_real(label, dfp)
                        if fig_real is not None:
                            st.plotly_chart(fig_real, use_container_width=True)
                            if not (math.isnan(mae) or math.isnan(mape)): st.success(f'MAE: {mae:.2f} kW | MAPE: {mape:.1f}%')
                        else: st.info('Nessun dato reale disponibile per questo giorno.')
                else: st.warning('Nessun dato disponibile per questa giornata.')
            st.subheader('📈 Confronto curve (4 giorni, 15 min)')
            comp = pd.DataFrame(); all_curves = pd.DataFrame()
            for lbl, dfp in results.items():
                if dfp is not None and not dfp.empty:
                    tmp = dfp.copy(); tmp['Potenza_kW'] = tmp['kWh_curve'] * 4
                    comp = pd.concat([comp, tmp.set_index('time')['Potenza_kW'].rename(lbl)], axis=1)
                    cols_ok = [c for c in ['time','GlobalRad_W','CloudCover_P','Temp_Air','rad_corr','kWh_curve','Potenza_kW'] if c in tmp.columns]
                    tmp2 = tmp[cols_ok].copy(); tmp2['giorno'] = lbl
                    all_curves = pd.concat([all_curves, tmp2], ignore_index=True)
            if not comp.empty: st.line_chart(comp)
            if not all_curves.empty:
                buf_all = io.StringIO(); all_curves.to_csv(buf_all, index=False)
                st.download_button('📦 Scarica TUTTE le curve (CSV unico)', buf_all.getvalue(), 'all_curves_15min.csv', 'text/csv')
            else: st.info('Nessuna curva disponibile per il confronto.')

# ---- TAB 4: Map (placeholder) ---- #
# ---- TAB 4: Mappa ---- #
# ---- TAB 4: Mappa (satellitare) ---- #
with tab4:
    st.subheader('🗺️ Localizzazione impianto fotovoltaico')
    st.write('Visualizzazione satellitare del sito di produzione.')

    lat = st.session_state.get('lat', DEFAULT_LAT)
    lon = st.session_state.get('lon', DEFAULT_LON)

    # Mostra le coordinate attuali
    st.markdown(f"**Coordinate attuali:** 🌍 {lat:.6f}, {lon:.6f}")

    # Layer Pydeck: punto posizione impianto
    import pydeck as pdk
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame({'lat': [lat], 'lon': [lon]}),
        get_position='[lon, lat]',
        get_color='[255, 165, 0, 220]',  # arancione semi-trasparente
        get_radius=60,
    )

    # Stato della vista (zoom centrato sul sito)
    view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=16, pitch=0)

    # Mappa con sfondo satellitare
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/satellite-v9',
        initial_view_state=view_state,
        layers=[layer],
        tooltip={"text": "Impianto fotovoltaico\nLat: {lat}\nLon: {lon}"}
    ))
