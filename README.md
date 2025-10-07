# ROBOTRONIX – Solar Forecast (PT15M)

- Provider toggle: Meteomatics / Open-Meteo  
- PT15M, tilt/orient, peak %, export CSV (curva + aggregato giornaliero)  
- Storico/Modello: training su `Dataset_Daily_EnergiaSeparata_2020_2025.csv`  
- Mappa satellite con box descrittivo  
- Log nascosti in `logs/app.log` (non in UI)

## Deploy su Railway
1. Carica questa cartella su GitHub.
2. Railway → New Project → Deploy from GitHub → seleziona repo.
3. Aggiungi **(opzionale)** variabili:
   - `METEO_USER`
   - `METEO_PASS`
4. Deploy.

## Avvio locale
```bash
pip install -r requirements.txt
streamlit run pv_forecast_all_in_one.py
