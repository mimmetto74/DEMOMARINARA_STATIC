# PROVA – Solar Forecast (V7 base V4)

- Login: **FVMANAGER** / **MIMMOFABIO**
- Storico + Modello (LinearRegression su `G_M0_Wm2` → `E_INT_Daily_kWh`)
- Previsioni 4 giorni a **15 minuti** (preferenza Meteomatics, fallback Open‑Meteo)
- Toggle provider, tilt/orientamento, stima picco %, export CSV (curva 15’ + aggregato)
- Mappa satellitare Folium con popup leggibile

## Meteomatics (opzionale)
Impostare variabili d'ambiente su Railway:
- `METEO_USER`
- `METEO_PASS`

Se assenti → fallback automatico Open‑Meteo (oraria interpolata a 15’).

## Avvio locale
```
pip install -r requirements.txt
streamlit run pv_forecast_all_in_one.py
```
