# ☀️ Solar Forecast - ROBOTRONIX for IMEPOWER

Questa demo mostra un sistema di previsione della produzione fotovoltaica.

## 🔑 Accesso
- Username: `FVMANAGER`
- Password: `MIMMOFABIO`

## 🌤️ Meteo
- Per le previsioni usa **Meteomatics** (radiazione + copertura nuvolosa)
- Se Meteomatics non è disponibile → fallback **Open-Meteo** (solo radiazione).

## 📂 Contenuto
- `pv_forecast_all_in_one.py` → app principale
- `Dataset_Daily_EnergiaSeparata_2020_2025.csv` → dati storici
- `Procfile`, `requirements.txt`, `runtime.txt`
- `.streamlit/config.toml`

## 🚀 Deploy su Railway
1. Carica i file su una repo GitHub.
2. Collega la repo a Railway.
3. L'app sarà raggiungibile online.
