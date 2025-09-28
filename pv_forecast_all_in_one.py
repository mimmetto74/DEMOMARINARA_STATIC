import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# =============================
# Config
# =============================
DATA_PATH = "Dataset_Daily_EnergiaSeparata_2020_2025.csv"
MODEL_PATH = "pv_model.joblib"
LOG_PATH = "log.csv"

# Funzione logging
def log_event(event, message):
    with open(LOG_PATH, "a") as f:
        f.write(f"{datetime.utcnow().isoformat()},{event},{message}\n")

# Carica dataset
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.dropna(subset=["E_INT_Daily_kWh", "G_M0_Wm2"])

# Split train/test
train = df[df["Date"] < "2025-01-01"]
test = df[df["Date"] >= "2025-01-01"]

X_train, y_train = train[["G_M0_Wm2"]], train["E_INT_Daily_kWh"]
X_test, y_test = test[["G_M0_Wm2"]], test["E_INT_Daily_kWh"]

# Addestramento
model = LinearRegression()
model.fit(X_train, y_train)
joblib.dump(model, MODEL_PATH)

# Previsioni test
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# =============================
# Streamlit App
# =============================
st.title("🔆 Solar Forecast - ROBOTRONIX for IMEPOWER")

st.subheader("📊 Analisi Storica")
st.write("MAE Test:", round(mae, 2))
st.write("R² Test:", round(r2, 3))

st.line_chart(df.set_index("Date")[["E_INT_Daily_kWh", "G_M0_Wm2"]])

# Previsioni Meteo
st.subheader("🔮 Previsioni Ieri + Oggi + Domani + Dopodomani")

lat = st.number_input("Latitudine", value=40.643278)
lon = st.number_input("Longitudine", value=16.986083)

if st.button("Calcola previsioni"):
    for offset, label in zip([-1,0,1,2], ["Ieri","Oggi","Domani","Dopodomani"]):
        date = (datetime.utcnow() + timedelta(days=offset)).date()
        try:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=direct_radiation,cloudcover&start={date}&end={date}"
            r = requests.get(url, timeout=10)
            data = r.json()
            log_event("forecast", f"{label} OK")
            st.success(f"Previsione {label} ({date}) disponibile")
        except Exception as e:
            log_event("error", f"{label}: {e}")
            st.error(f"Errore {label}: {e}")

# Download log
with open(LOG_PATH, "rb") as f:
    st.download_button("📥 Scarica log CSV", f, file_name="log.csv", mime="text/csv")
