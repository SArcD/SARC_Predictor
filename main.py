import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Ruta del archivo de Excel
file_path = "Base 2019 Santiago Arceo.xlsx"

# Título de la app
st.title("Sarc_Predictor: predecir sarcopenia mediante Machine-Learning")

# Leer y mostrar el archivo de Excel
try:
    datos = pd.read_excel(file_path, engine='openpyxl')
    st.write("Datos cargados con éxito:")
    st.dataframe(datos)

    # Reemplazar valores numéricos con etiquetas 'Hombre' y 'Mujer' en la columna 'sexo'
    datos['sexo'] = datos['sexo'].replace({1.0: 'Hombre', 2.0: 'Mujer'})

    # Generar gráfico de pastel para la columna 'sexo'
    sexo_counts = datos['sexo'].value_counts()

    # Crear la figura y mostrarla en Streamlit
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sexo_counts, labels=sexo_counts.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightpink'])
    ax.set_title('Proporción de Hombres vs Mujeres')
    ax.axis('equal')  # Asegurar que el gráfico sea un círculo.

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)

except FileNotFoundError:
    st.error("No se encontró el archivo en la ruta especificada.")
except Exception as e:
    st.error(f"Error al cargar el archivo: {e}")
