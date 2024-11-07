import streamlit as st
import pandas as pd
import requests
from io import BytesIO

# Título de la app
st.title("Sarc_Predictor: predecir sarcopenia mediante Machine-Learning")

# URL del archivo de Excel en formato raw
file_url = "https://raw.githubusercontent.com/SArcD/SARC_Predictor/dc0862d02d4cc1187e1b8dd5a991427132cd55c1/Base%202019%20Santiago%20Arceo.xlsx"

# Descargar el archivo y cargarlo en un DataFrame
try:
    response = requests.get(file_url)
    response.raise_for_status()  # Verificar si la descarga fue exitosa
    datos = pd.read_excel(BytesIO(response.content), engine='openpyxl')
    st.write("Datos cargados con éxito:")
    st.dataframe(datos)
except Exception as e:
    st.error(f"Error al cargar el archivo: {e}")
