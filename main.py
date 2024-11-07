import streamlit as st
import pandas as pd

# Título de la app
st.title("Sarc_Predictor: predecir sarcopenia mediante Machine-Learning")

# URL del archivo de Excel en formato raw
file_url = "https://raw.githubusercontent.com/SArcD/SARC_Predictor/main/Base%202019%20Santiago%20Arceo.xlsx"

# Leer y mostrar el archivo de Excel
try:
    datos = pd.read_excel(file_url)
    st.write("Datos cargados con éxito:")
    st.dataframe(datos)
except Exception as e:
    st.error(f"Error al cargar el archivo: {e}")
