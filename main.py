import streamlit as st
import pandas as pd

# Título de la app
st.title("Sarc_Predictor: predecir sarcopenia mediante Machine-Learning")

# Subir archivo de Excel
uploaded_file = st.file_uploader("Cargar archivo de Excel", type="xlsx")

# Leer y mostrar el archivo de Excel
if uploaded_file is not None:
    try:
        datos = pd.read_excel(uploaded_file)
        st.write("Datos cargados con éxito:")
        st.dataframe(datos)
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
else:
    st.info("Por favor, sube un archivo de Excel.")
