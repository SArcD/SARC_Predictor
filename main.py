import streamlit as st
import pandas as pd

st.title("Selector de Base de Datos desde GitHub")

# URL del archivo en GitHub (formato raw)
url_github = "https://raw.githubusercontent.com/SArcD/SARC_Predictor/main/Base%202019%20Santiago%20Arceo.xlsx"

# Cargar el archivo
try:
    excel_file = pd.ExcelFile(url_github)
    hojas = excel_file.sheet_names

    # Menú desplegable para seleccionar la hoja
    hoja_seleccionada = st.selectbox("Selecciona una hoja del archivo:", hojas)

    # Leer los datos de la hoja seleccionada
    datos = pd.read_excel(excel_file, sheet_name=hoja_seleccionada)

    # Mostrar una vista previa de los datos
    st.write(f"Vista previa de la hoja **{hoja_seleccionada}**:")
    st.dataframe(datos.head())

except Exception as e:
    st.error(f"Ocurrió un error al intentar cargar el archivo: {e}")
