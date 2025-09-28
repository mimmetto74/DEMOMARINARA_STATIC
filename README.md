# ☀️ Solar Forecast - ROBOTRONIX for IMEPOWER

Questa applicazione prevede la produzione giornaliera di un impianto fotovoltaico:

- Addestramento modello su dati storici (`Dataset_Daily_EnergiaSeparata_2020_2025.csv`).
- Previsioni per Ieri, Oggi, Domani, Dopodomani.
- Meteomatics come fonte principale, con fallback automatico a Open-Meteo.
- Log CSV delle previsioni scaricabile.

## Deploy su Railway
```bash
git init
git add .
git commit -m "Deploy ROBOTRONIX package"
git branch -M main
git remote add origin <repo_url>
git push -u origin main
```
