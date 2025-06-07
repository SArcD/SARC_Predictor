import streamlit as st
import pandas as pd
import requests
from io import BytesIO

st.title("📊 Selector de Base de Datos desde GitHub")

# URL raw del archivo
url_github = "https://github.com/SArcD/SARC_Predictor/raw/main/Base%202019%20Santiago%20Arceo.xlsx"

try:
    # Descargar archivo como binario
    r = requests.get(url_github)
    r.raise_for_status()  # Verifica error HTTP

    archivo = BytesIO(r.content)

    # Leer archivo Excel con engine explícito
    excel_file = pd.ExcelFile(archivo, engine="openpyxl")

    # Obtener lista de hojas
    hojas = excel_file.sheet_names
    hoja_seleccionada = st.selectbox("Selecciona una hoja:", hojas)

    # Cargar la hoja seleccionada
    df = pd.read_excel(excel_file, sheet_name=hoja_seleccionada, engine="openpyxl")

    st.success(f"Hoja cargada: {hoja_seleccionada}")
    st.dataframe(df.head())

except requests.exceptions.RequestException as req_err:
    st.error(f"Error al descargar el archivo: {req_err}")

except Exception as e:
    st.error(f"Error al procesar el archivo: {e}")
