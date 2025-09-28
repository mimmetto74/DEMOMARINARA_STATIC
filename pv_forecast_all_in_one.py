import streamlit as st
import pandas as pd
import os

LOG_PATH = "log_forecast.csv"

def read_log():
    if not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0:
        with open(LOG_PATH, "w") as f:
            f.write("date,parameter,value\n")
        return pd.DataFrame(columns=["date", "parameter", "value"])
    else:
        return pd.read_csv(LOG_PATH)

def append_to_log(date, parameter, value):
    with open(LOG_PATH, "a") as f:
        f.write(f"{date},{parameter},{value}\n")

st.title("⚡ Solar Forecast - ROBOTRONIX")

# Sidebar per gestione log
st.sidebar.subheader("📥 Scarica Log")

log_options = st.sidebar.multiselect(
    "Seleziona quali log scaricare:",
    ["Ieri", "Oggi", "Domani", "Dopodomani"]
)

if st.sidebar.button("Scarica log selezionati"):
    log_df = read_log()
    if not log_df.empty:
        export_df = log_df[log_df["parameter"].isin(log_options)]
        st.sidebar.download_button(
            label="📂 Download CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="forecast_log.csv",
            mime="text/csv"
        )
    else:
        st.sidebar.warning("⚠️ Nessun log disponibile")

st.write("Demo app con gestione log e download selettivo.")
