# Solar Forecast – ROBOTRONIX (Railway Ready)

Questa app Streamlit include:
- Login con credenziali (FVMANAGER / MIMMOFABIO)
- Addestramento modello dai dati storici (CSV incluso)
- Previsioni 4 giorni (ieri, oggi, domani, dopodomani) con **Meteomatics** (fallback Open‑Meteo)
- Grafici singoli + comparativo
- Log CSV scaricabile dalla sidebar

## Avvio locale
```bash
pip install -r requirements.txt
streamlit run pv_forecast_all_in_one.py
```

## Railway
- Carica la cartella su GitHub e collega a Railway. Il `Procfile` avvia Streamlit automaticamente.
