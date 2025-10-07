# ROBOTRONIX – PROVA V7 (PT15M, tilt/orient)

App Streamlit per previsioni FV:
- Provider toggle: Meteomatics (POA `global_rad_tilt_x_orientation_y`) o Open-Meteo (simulato).
- Risoluzione 15 minuti, fuso `Europe/Rome`.
- Stima kW, kWh/giorno, picco e % della targa.
- Download CSV curve 15 min e aggregato giornaliero.
- Log API **non** esposti in UI.

## Deploy (Railway)
1. Crea una nuova repo GitHub con questi file.
2. Su Railway → New Project → Deploy from GitHub → seleziona repo.
3. Aggiungi **Variables**:
   - `METEO_USER` = *tuo utente Meteomatics*
   - `METEO_PASS` = *tua password Meteomatics*
4. Deploy.

## Avvio locale
```bash
pip install -r requirements.txt
streamlit run pv_forecast_all_in_one.py
