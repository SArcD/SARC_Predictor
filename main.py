# streamlit_app.py

import streamlit as st
import pandas as pd

st.title("Cargador de base de datos para modelo")

# Subida del archivo Excel
archivo = st.file_uploader("Selecciona tu archivo Excel (.xlsx)", type=["xlsx"])

if archivo is not None:
    # Cargar el archivo para mostrar las hojas disponibles
    excel_file = pd.ExcelFile(archivo)
    hojas = excel_file.sheet_names

    # Menú desplegable para seleccionar la hoja
    hoja_seleccionada = st.selectbox("Selecciona la hoja de datos", hojas)

    # Cargar la hoja seleccionada como DataFrame
    datos = pd.read_excel(archivo, sheet_name=hoja_seleccionada)
    
    st.success(f"Datos cargados desde la hoja: {hoja_seleccionada}")
    
    # Mostrar una vista previa del DataFrame
    st.dataframe(datos.head())
