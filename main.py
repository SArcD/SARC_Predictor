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
#-----------------------------------------------------------------------------------

        import matplotlib.pyplot as plt
        import numpy as np
        import streamlit as st


        # --- 1. Definir columnas y etiquetas ---
        columns_to_check = ['P44_3', 'P44_5', 'P44_7', 'P44_8', 'P44_9', 'P44_11', 'P44_12',
                        'P44_13', 'P44_14', 'P44_20', 'P44_21', 'P44_24', 'P44_27', 'P44_31']

        comorbidities_labels = ['VIH', 'Anemia', 'Arritmia', 'Artritis Reumatoide', 'Cáncer', 'Depresión', 'Diabetes Complicada',
                            'Diabetes Leve', 'Enfermedad Cerebro Vascular', 'Hipertensión Complicada', 'Hipertensión Sin Complicación',
                            'Insuficiencia Renal', 'Obesidad', 'Pérdida de Peso']

        # --- 2. Procesamiento de datos ---
        datos[columns_to_check] = datos[columns_to_check].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
        datos['Sin comorbilidades'] = (datos[columns_to_check].sum(axis=1) == 0).astype(int)

        comorbidities_counts = datos[columns_to_check + ['Sin comorbilidades']].sum()
        comorbidities_labels_with_none = comorbidities_labels + ['Sin comorbilidades']

        
        # --- Preparar proporciones y etiquetas

        total = comorbidities_counts.sum()
        proportions = comorbidities_counts / total

        column_to_label = dict(zip(columns_to_check + ['Sin comorbilidades'], comorbidities_labels_with_none))

        sorted_indices = proportions.sort_values(ascending=False).index
        proportions_sorted = proportions.loc[sorted_indices]
        labels_sorted = [column_to_label[i] for i in sorted_indices]

        cumulative = proportions_sorted.cumsum()
        cutoff_index = np.argmax(cumulative > 0.9)

        main_props = proportions_sorted.iloc[:cutoff_index]
        main_labels = labels_sorted[:cutoff_index]

        small_props = proportions_sorted.iloc[cutoff_index:]
        small_labels = labels_sorted[cutoff_index:]

        small_sum = small_props.sum()
        small_label = 'Otras Comorbilidades'

        final_props = list(main_props) + [small_sum]
        final_labels = list(main_labels) + [small_label]

        # --- Graficar

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), gridspec_kw={'height_ratios': [3, 1]}, dpi=100)

        color_map = plt.cm.tab20(np.linspace(0, 1, 20))
        colors_main = color_map[:len(final_labels)]
        colors_small = color_map[len(final_labels):len(final_labels)+len(small_labels)]

        # Gráfico 1: Principales + Otras
        left = 0
        for prop, label, color in zip(final_props, final_labels, colors_main):
            ax1.barh(0, prop, left=left, label=label, color=color, edgecolor='white')
            if prop > 0.02:
                ax1.text(left + prop/2, 0, f'{prop*100:.1f}%', ha='center', va='center', fontsize=9)
            left += prop

        ax1.set_xlim(0, 1)
        ax1.set_yticks([])
        ax1.set_xlabel('Proporción del Total de Pacientes')
        ax1.set_title('Distribución de Comorbilidades (Principales + "Otras Comorbilidades")')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, frameon=False)

        # Gráfico 2: Detalle de "Otras"
        left = 0
        for prop, label, color in zip(small_props, small_labels, colors_small):
            proportion_within_small = prop / small_sum
            ax2.barh(0, proportion_within_small, left=left, label=label, color=color, edgecolor='white')
            if proportion_within_small > 0.05:
                ax2.text(left + proportion_within_small/2, 0, f'{proportion_within_small*100:.1f}%', ha='center', va='center', fontsize=8)
            left += proportion_within_small

        ax2.set_xlim(0, 1)
        ax2.set_yticks([])
        ax2.set_xlabel('Proporción dentro de "Otras Comorbilidades"')
        ax2.set_title('Detalle de "Otras Comorbilidades" (Relativo al grupo "Otras")')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, frameon=False)

        plt.tight_layout()

        # Mostrar en Streamlit
        st.pyplot(fig)




except Exception as e:
    st.error(f"Ocurrió un error al intentar cargar el archivo: {e}")

