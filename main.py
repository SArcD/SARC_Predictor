import streamlit as st
import pandas as pd
import requests
from io import BytesIO

st.title("📊 Selector de Base de Datos desde GitHub")

st.subheader("Carga de los datos")
    
st.markdown("""
En el siguiente menú puede elegir entre las bases de datos disponibles""")
# Seleccionar el año de la base de datos
selected_year = st.selectbox("Por favor, seleccione la base de datos:", ["2019", "2022"])

# Definir la ruta del archivo en función de la selección
if selected_year == "2022":
    file_path = "Base 2022 Santiago Arceo.xls"
else:
    file_path = "Base 2019 Santiago Arceo.xls"

# Intento de cargar el archivo de Excel usando `xlrd` para archivos `.xls`
try:
    datos = pd.read_excel(file_path)  # Rellenar NaN con espacios
    #st.write(f"Datos de la base {selected_year} cargados con éxito:")
    #st.dataframe(datos)
    st.success(f"✅ Datos de la base {selected_year} cargados con éxito.")

    # Mostrar vista previa opcional en un expander
    with st.expander("📂 Ver datos cargados"):
        #st.dataframe(datos)
        st.dataframe(datos.describe())
        import matplotlib.pyplot as plt
        import streamlit as st

        # Reemplazar valores numéricos por etiquetas de sexo
        datos['sexo'] = datos['sexo'].replace({1.0: 'Hombre', 2.0: 'Mujer'})

        # Conteo por categoría
        sexo_counts = datos['sexo'].value_counts()

        # Crear la gráfica de pastel
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            sexo_counts,
            labels=sexo_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=['skyblue', 'lightpink']
        )
        ax.set_title('Proporción de Hombres vs Mujeres')
        ax.axis('equal')

        # Mostrar en Streamlit
        st.pyplot(fig)




except Exception as e:
    st.error(f"Ocurrió un error al intentar cargar el archivo: {e}")

