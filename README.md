
# Solar Forecast – ROBOTRONIX (Railway ready)

- Modello lineare addestrato su dataset storico (G_M0_Wm2 → E_INT_Daily_kWh)
- Previsioni per Ieri/Oggi/Domani/Dopodomani con **Meteomatics** (fallback **Open‑Meteo**)
- 4 grafici separati (curva 15 min/ora prevista)
- **Log CSV** con URL, provider, esito e produzione stimata
- **Sidebar** per scaricare il log filtrato (Tutti / Meteomatics / Open‑Meteo / Errori)

## Deploy
1. Carica questi file in una repo GitHub
2. Collega la repo a Railway → Deploy
3. Porta esposta: gestita da Procfile (`$PORT`)

Credenziali Meteomatics da impostare nel codice o tramite Variabili d'Ambiente:
- `MM_USER`
- `MM_PASS`
