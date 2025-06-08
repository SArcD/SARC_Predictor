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
#------------------------------Comparación por sexo

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        # Asegurar que la columna 'sexo' esté bien codificada
        datos['sexo'] = datos['sexo'].replace({1.0: 'Hombre', 2.0: 'Mujer'})

        # --- Separar por sexo
        hombres = datos[datos['sexo'] == 'Hombre'].copy()
        mujeres = datos[datos['sexo'] == 'Mujer'].copy()

        # Sumar comorbilidades
        hombres_counts = hombres[columns_to_check].sum()
        mujeres_counts = mujeres[columns_to_check].sum()

        # Añadir 'Sin comorbilidades'
        hombres['Sin comorbilidades'] = (hombres[columns_to_check].sum(axis=1) == 0).astype(int)
        mujeres['Sin comorbilidades'] = (mujeres[columns_to_check].sum(axis=1) == 0).astype(int)

        hombres_counts['Sin comorbilidades'] = hombres['Sin comorbilidades'].sum()
        mujeres_counts['Sin comorbilidades'] = mujeres['Sin comorbilidades'].sum()

        # Crear DataFrames
        df_hombres = pd.DataFrame({
            'Comorbilidad': comorbidities_labels + ['Sin Comorbilidades'],
            'Conteo': hombres_counts.values
        })

        df_mujeres = pd.DataFrame({
            'Comorbilidad': comorbidities_labels + ['Sin Comorbilidades'],
            'Conteo': mujeres_counts.values
        })

        # --- Definir orden personalizado
        orden_personalizado = [
            'Sin Comorbilidades',
            'Diabetes Leve',
            'Diabetes Complicada',
            'Hipertensión Complicada',
            'Hipertensión Sin Complicación',
            'Otras'
        ]

        comorbilidades_principales = ['Diabetes Leve', 'Diabetes Complicada', 'Hipertensión Complicada', 'Hipertensión Sin Complicación']

        def preparar(df):
            principales = df[df['Comorbilidad'].isin(comorbilidades_principales + ['Sin Comorbilidades'])]
            restantes = df[~df['Comorbilidad'].isin(comorbilidades_principales + ['Sin Comorbilidades'])]
            otras = pd.DataFrame({'Comorbilidad': ['Otras'], 'Conteo': [restantes['Conteo'].sum()]})
            final = pd.concat([principales, otras], ignore_index=True)
            final['Comorbilidad'] = pd.Categorical(final['Comorbilidad'], categories=orden_personalizado, ordered=True)
            return final.sort_values('Comorbilidad')

        top_hombres = preparar(df_hombres)
        top_mujeres = preparar(df_mujeres)

        # --- Graficar
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

        # Paleta de colores
        color_dict = {
            'Sin Comorbilidades': '#ADD8E6',
            'Diabetes Leve': '#66C2A5',
            'Diabetes Complicada': '#A6D854',
            'Hipertensión Complicada': '#FC8D62',
            'Hipertensión Sin Complicación': '#8DA0CB',
            'Otras': '#E78AC3'
        }

        # Totales
        total_hombres = hombres.shape[0]
        total_mujeres = mujeres.shape[0]

        # Hombres
        bottom = 0
        for _, row in top_hombres.iterrows():
            ax.bar('Hombres', row['Conteo'], bottom=bottom, color=color_dict.get(row['Comorbilidad'], 'gray'), edgecolor='white')
            porcentaje = row['Conteo'] / total_hombres * 100
            if porcentaje > 2:
                ax.text('Hombres', bottom + row['Conteo']/2, f'{porcentaje:.1f}%', ha='center', va='center', fontsize=8)
            bottom += row['Conteo']

        # Mujeres
        bottom = 0
        for _, row in top_mujeres.iterrows():
            ax.bar('Mujeres', row['Conteo'], bottom=bottom, color=color_dict.get(row['Comorbilidad'], 'gray'), edgecolor='white')
            porcentaje = row['Conteo'] / total_mujeres * 100
            if porcentaje > 2:
                ax.text('Mujeres', bottom + row['Conteo']/2, f'{porcentaje:.1f}%', ha='center', va='center', fontsize=8)
            bottom += row['Conteo']

        # Ajustes finales
        ax.set_ylabel('Número de Pacientes')
        ax.set_title('Distribución de Comorbilidades Principales por Sexo')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(0, max(total_hombres, total_mujeres) * 1.2)

        # Totales
        ax.text('Hombres', total_hombres * 0.05, f'n={total_hombres}', ha='center', va='top', fontsize=9, fontweight='bold')
        ax.text('Mujeres', total_mujeres * 0.05, f'n={total_mujeres}', ha='center', va='top', fontsize=9, fontweight='bold')

        # Leyenda
        handles_labels = {label: plt.Rectangle((0,0),1,1, color=color_dict[label]) for label in orden_personalizado}
        ax.legend(handles_labels.values(), handles_labels.keys(), bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, frameon=False, title="Comorbilidades")

        plt.tight_layout()
        st.pyplot(fig)

    with st.expander("Varianzas"):
        # Definir las columnas que deben ser iguales a 0
        columns_to_check = ['P44_3', 'P44_5', 'P44_7', 'P44_8', 'P44_9', 'P44_11', 'P44_12',
                    'P44_13', 'P44_14', 'P44_20', 'P44_21', 'P44_24', 'P44_27', 'P44_31']

        # Filtrar las filas en las que todas las columnas especificadas son iguales a 0
        filtered_data = datos[(datos[columns_to_check] == 0).all(axis=1)]

        # Mostrar el DataFrame resultante
        #filtered_data

        df=filtered_data[['P117_1','P117_2','P117_3','P118_1','P118_2','P118_3','P119_1','P119_2','P119_3','P120_1','P120_2','P120_3','P121_1','P121_2','P121_3','P122_1','P122_2','P122_3','P123_1','P123_2','P123_3','P124_1','P124_2','P124_3','P125_1','P125_2','P125_3','P126_1','P126_2','P126_3','P127_1','P127_2','P127_3','P128_1','P128_2','P128_3','P129_1','P129_2','P129_3','P130_1','P130_2','P130_3']]

        # Corrigiendo la advertencia al agrupar columnas
        df_grouped = df.T.groupby(lambda x: x.split('_')[0]).mean().T
        # Calculando el IMC: Peso / (Altura^2)
        df_grouped['IMC'] = df_grouped['P117'] / ((df_grouped['P118']*0.01) ** 2)

        #df_grouped
        # Promediar los valores de las columnas P113_1, P113_3, y P113_5
        df_2=filtered_data[['P113_1', 'P113_3', 'P113_5']]
        df_2['P113_iz'] = filtered_data[['P113_1', 'P113_3', 'P113_5']].mean(axis=1)
        # Promediar los valores de las columnas P113_1, P113_3, y P113_5
        df_2 = df_2.drop(columns=['P113_1', 'P113_3', 'P113_5'])
        #df_2
        # Promediar los valores de las columnas P113_1, P113_3, y P113_5
        df_3=filtered_data[['P113_2', 'P113_4', 'P113_6']]
        df_3['P113_der'] = filtered_data[['P113_2', 'P113_4', 'P113_6']].mean(axis=1)
        # Promediar los valores de las columnas P113_1, P113_3, y P113_5
        df_3 = df_3.drop(columns=['P113_2', 'P113_4', 'P113_6'])
        #df_3
        df_3b = pd.concat([df_2,df_3], axis=1)
        df_3b['P113']=(df_2['P113_iz']+df_3['P113_der'])/2
        df_3b = df_3b.drop(columns=['P113_iz', 'P113_der'])
        #df_3b
        # Seleccionar las columnas y eliminar los valores que sean 0 antes de calcular el promedio
        df_4 = filtered_data[['P112_4_1', 'P112_4_2']].replace(0, np.nan).dropna()
        # Calcular el promedio
        df_4['P112'] = df_4.mean(axis=1)
        # Verificar los valores únicos en P112 para asegurarse de que no sean todos iguales
        unique_values = df_4['P112'].unique()
        #df_4, unique_values
        df_4['P112_vel'] = 4 / df_4['P112']
        df_4 = df_4.drop(columns=['P112_4_1', 'P112_4_2', 'P112'])
        #df_4
        df_datos=filtered_data[['folio_paciente','edad_am','sexo','nacio_en_mexico']]
        #df_datos
        # Concatenating df_grouped with df_r to create a single DataFrame
        df_combined = pd.concat([df_datos, df_grouped, df_3b, df_4], axis=1)
        #df_combined
        # Hombre# Standardizing the columns from the 4th column onwards in df_combined
        #df_combined = df_combined[df_combined['sexo'] == "Hombre"]

        columns_to_standardize = df_combined.columns[4:]  # Selecting columns from the 4th column onwards

        # Calculating variance using the provided method: dividing by the mean and then calculating the variance
        features = df_combined[columns_to_standardize]  # Selecting the standardized features
        variances = (features / features.mean()).dropna().var()

        variances=variances.sort_values(ascending=False)
        variances
        variances_df = pd.DataFrame({'Variable': variances.index, 'Normalized Variance': variances.values})
        #import matplotlib.pyplot as plt

        # Diccionario de traducción
        column_labels_en = {
            'P112_vel': 'Gait Speed',
            'P113': 'Grip Strength',
            'P125': 'Triceps Skinfold',
            'P126': 'Subscapular Skinfold',
            'P128': 'Calf Circumference',
            'P127': 'Biceps Skinfold',
            'P117': 'Weight',
            'IMC': 'BMI',
            'P123': 'Thigh Circumference',
            'P121': 'Waist Circumference',
            'P120': 'Arm Circumference',
            'P124': 'Calf Skinfold',
            'P122': 'Abdomen Circumference',
            'P119': 'Chest Circumference',
            'P129': 'Neck Circumference',
            'P130': 'Wrist Circumference',
            'P118': 'Hip Circumference'
        }

        # --- 1. Traducir nombres de variable en variances_df ---
        variances_df['Variable_English'] = variances_df['Variable'].map(column_labels_en)

        # Si alguna variable no está en el diccionario, deja el nombre original
        variances_df['Variable_English'] = variances_df['Variable_English'].fillna(variances_df['Variable'])

        # --- 2. Aplicar umbral 0.02 ---
        variances_filtered = variances_df[variances_df['Normalized Variance'] >= 0.02]

        # --- 3. Graficar ---
        #plt.figure(figsize=(10, 6), dpi=300)
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        plt.barh(variances_filtered['Variable_English'], variances_filtered['Normalized Variance'], color='skyblue', edgecolor='black')    
        plt.xlabel('Normalized Variance')
        plt.title('Normalized Variances of Variables (Men)')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        #plt.tight_layout()
        fig.tight_layout()


        # Guardar en alta resolución
        #plt.savefig('/content/Normalized_Variances_Men_Filtered.png', dpi=300, bbox_inches='tight')
        st.pyplot(fig)




except Exception as e:
    st.error(f"Ocurrió un error al intentar cargar el archivo: {e}")

