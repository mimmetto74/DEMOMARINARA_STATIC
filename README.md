# Solar Forecast – ROBOTRONIX (Railway Ready)

Questa app Streamlit:
- Addestra un modello lineare su dati storici (`Dataset_Daily_EnergiaSeparata_2020_2025.csv`)
- Calcola previsioni per **Ieri, Oggi, Domani, Dopodomani** usando **Meteomatics** (fallback **Open‑Meteo**)
- Mostra 4 grafici separati e salva un log CSV scaricabile dalla sidebar.

## Avvio locale
```bash
pip install -r requirements.txt
streamlit run pv_forecast_all_in_one.py
```

## Railway
- Carica la cartella su GitHub e collega a Railway. Il `Procfile` avvia Streamlit automaticamente.
