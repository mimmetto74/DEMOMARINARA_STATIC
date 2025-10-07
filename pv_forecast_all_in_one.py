import streamlit as st
import pandas as pd

st.set_page_config(page_title="ROBOTRONIX PROVA V7", page_icon="☀️", layout="wide")
st.title("ROBOTRONIX PROVA V7")

st.success("App pronta: questo è un placeholder minimale per partire subito.")
st.write("Sostituisci questo file con la versione completa del tuo script quando vuoi.")

# Carica dataset se presente
try:
    df = pd.read_csv("Dataset_Daily_EnergiaSeparata_2020_2025.csv", parse_dates=["Date"])
    st.line_chart(df.set_index("Date")[["E_INT_Daily_kWh"]])
except Exception as e:
    st.info("Carica il dataset reale 'Dataset_Daily_EnergiaSeparata_2020_2025.csv' nella root.")
