# ☀️ Solar Forecast - ROBOTRONIX for IMEPOWER (Meteomatics + fallback + log)

- Training: sempre su Dataset_Daily_EnergiaSeparata_2020_2025.csv
- Previsioni: Meteomatics (`direct_rad:W` + `total_cloud_cover:p`) con formula G_eff = direct_rad*(1 - cloud_cover/100)
- Fallback automatico: se Meteomatics non risponde, usa Open-Meteo (`shortwave_radiation` + `cloudcover`)
- Log: ogni previsione viene registrata in forecast_log.csv (provider, stato, eventuale errore, coordinate)
- Output: 4 grafici separati (Ieri, Oggi, Domani, Dopodomani)
- Login: FVMANAGER / MIMMOFABIO

Dataset status in this ZIP: included_real.
