# ☀️ Solar Forecast - ROBOTRONIX for IMEPOWER

- Training: sempre su Dataset_Daily_EnergiaSeparata_2020_2025.csv
- Previsioni: Meteomatics (`direct_rad:W` + `total_cloud_cover:p`) con formula G_eff = direct_rad*(1 - cloud_cover/100)
- Fallback: Open-Meteo se Meteomatics non risponde
- Output: curve 15-min + valori giornalieri domani e dopodomani
- Login: FVMANAGER / MIMMOFABIO

Dataset status in this ZIP: included_real.
