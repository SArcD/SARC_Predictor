import streamlit as st
import pandas as pd

# Cargar archivo de Excel
file_path = "Base 2019 Santiago Arceo.xlsx"

# Título de la app
st.title("Sarc_Predictor: predecir sarcopenia mediante Machine-Learning")

# Leer y mostrar el archivo de Excel
try:
    datos = pd.read_excel(file_path)
    st.write("Datos cargados con éxito:")
    st.dataframe(data)
except FileNotFoundError:
    st.error("No se encontró el archivo en la ruta especificada.")
