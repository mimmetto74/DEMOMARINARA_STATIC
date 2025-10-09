# 🛰️ Robotronix PT15M — Web App Streamlit (Ottimizzata per Railway)

## 🚀 Descrizione
Questa applicazione **Streamlit** elabora e visualizza dati fotovoltaici da sensori, previsioni e log CSV.
È progettata per essere eseguita su **Railway.app**, con deploy automatico da **GitHub**.

---

## 📁 Struttura del progetto

```
robotronix_pt15m_tilt_zip_v4_OPTIMIZED_RAILWAY/
│
├── pv_forecast_all_in_one.py      # Applicazione principale Streamlit
├── requirements.txt               # Dipendenze Python
├── Procfile                       # Istruzioni di avvio per Railway
├── runtime.txt                    # Versione Python
├── railway.toml                   # Configurazione Railway
├── Dataset_Daily_EnergiaSeparata_2020_2025.csv
├── forecast_log.csv
└── .streamlit/
    └── config.toml                # Configurazione Streamlit (tema, layout, ecc.)
```

---

## ⚙️ Requisiti principali

- Python 3.11+
- Librerie:
  ```
  streamlit
  pandas
  numpy
  matplotlib
  scikit-learn
  ```

---

## 🌐 Deploy su Railway (passo per passo)

1. **Accedi** a [https://railway.app](https://railway.app)
2. **Crea un nuovo progetto → Deploy from GitHub Repo**
3. **Seleziona la tua repository** con questi file
4. Railway rileverà automaticamente:
   - il file `Procfile`
   - il comando di avvio Streamlit
   - l’ambiente Python da `railway.toml`
5. Attendi il completamento del **build automatico**
6. Dopo il deploy, clicca su **“Open App”** per vedere l’interfaccia Streamlit online 🎉

---

## 🧠 Suggerimenti

- Per modifiche locali:
  ```bash
  pip install -r requirements.txt
  streamlit run pv_forecast_all_in_one.py
  ```
- Railway assegna automaticamente una variabile d’ambiente `PORT`; non modificare la riga:
  ```python
  streamlit run pv_forecast_all_in_one.py --server.port $PORT --server.address 0.0.0.0
  ```

---

## 💬 Supporto
Per problemi di deploy o configurazione:
- Controlla i log su Railway → “Deployments → View Logs”
- Verifica che tutti i file CSV e `.streamlit/config.toml` siano inclusi nel repository
