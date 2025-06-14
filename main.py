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
        fig_1, ax = plt.subplots(figsize=(6, 6))
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
        st.pyplot(fig_1)
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
        if 'fig_comorbilidades' not in st.session_state:

            fig_2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), gridspec_kw={'height_ratios': [3, 1]}, dpi=100)

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
            st.session_state.fig_comorbilidades = fig_2
        st.pyplot(st.session_state.fig_comorbilidades)

        # Mostrar en Streamlit
        #st.pyplot(fig_2)
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
        
        if 'fig_comorbilidades_sexo' not in st.session_state:
            # --- Graficar
            fig_3, ax = plt.subplots(figsize=(10, 6), dpi=100)

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
            st.session_state.fig_comorbilidades_sexo = fig_3
        st.pyplot(st.session_state.fig_comorbilidades_sexo)
        #st.pyplot(fig_3)

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
        if 'fig_varianza_men' not in st.session_state:

            fig_4, ax = plt.subplots(figsize=(10, 6), dpi=150)
            ax.barh(
                variances_filtered['Variable_English'],
                variances_filtered['Normalized Variance'],
                color='skyblue', edgecolor='black'
            )
            ax.set_xlabel('Normalized Variance')
            ax.set_title('Normalized Variances of Variables (Men)')
            ax.invert_yaxis()
            ax.grid(axis='x', linestyle='--', alpha=0.7)
        
            fig_4.tight_layout()
            st.session_state.fig_varianza_men = fig_4



        # Guardar en alta resolución
        #plt.savefig('/content/Normalized_Variances_Men_Filtered.png', dpi=300, bbox_inches='tight')
        st.pyplot(st.session_state.fig_varianza_men)
        #st.pyplot(fig)

        # --- 1. Separar hombres y mujeres
        df_hombres = df_combined[df_combined['sexo'] == 'Hombre']
        df_mujeres = df_combined[df_combined['sexo'] == 'Mujer']

        # --- 2. Columnas a estandarizar
        columns_to_standardize = df_combined.columns[4:]

        # --- 3. Calcular varianzas normalizadas
        features_hombres = df_hombres[columns_to_standardize]
        variances_hombres = (features_hombres / features_hombres.mean()).dropna().var()
        variances_hombres = variances_hombres.sort_values(ascending=False)

        features_mujeres = df_mujeres[columns_to_standardize]
        variances_mujeres = (features_mujeres / features_mujeres.mean()).dropna().var()
        variances_mujeres = variances_mujeres.sort_values(ascending=False)
    
        # --- 4. Diccionario de nombres en inglés
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

        # --- 5. Traducir nombres
        variances_hombres.index = variances_hombres.index.map(lambda x: column_labels_en.get(x, x))    
        variances_mujeres.index = variances_mujeres.index.map(lambda x: column_labels_en.get(x, x))

        # --- 6. Crear DataFrame combinado
        merged_df = pd.merge(
            variances_hombres.rename('Men'),
            variances_mujeres.rename('Women'),
            left_index=True,
            right_index=True,
            how='inner'
        )

        # --- 7. Ordenar por promedio de varianza        
        merged_df['Mean Variance'] = merged_df[['Men', 'Women']].mean(axis=1)
        merged_df = merged_df.sort_values('Mean Variance', ascending=False)

        # --- 8. Gráfica comparativa
        if 'fig_varianza' not in st.session_state:

            fig_5, ax = plt.subplots(figsize=(10, 8), dpi=150)

            bar_width = 0.4
            y_pos = np.arange(len(merged_df))

            # Barras de hombres
            ax.barh(y_pos - bar_width/2, merged_df['Men'], height=bar_width, label='Men', color='steelblue', edgecolor='black')

            # Barras de mujeres
            ax.barh(y_pos + bar_width/2, merged_df['Women'], height=bar_width, label='Women', color='lightcoral', edgecolor='black')

            ax.set_yticks(y_pos)
            ax.set_yticklabels(merged_df.index)
            ax.set_xlabel('Normalized Variance')
            ax.set_title('Comparison of Normalized Variances by Sex')
            ax.invert_yaxis()
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            ax.legend()
            fig_5.tight_layout()
            st.session_state.fig_varianza = fig_5

        st.pyplot(st.session_state.fig_varianza)
        #st.pyplot(fig)
        ################################ Histogramas comparativos de variables

        #import streamlit as st
        import plotly.graph_objects as go
        import plotly.subplots as sp

        # Diccionario de etiquetas en español
        column_labels = {
            'P112_vel': 'Marcha (m/s)',
            'P113': 'Fuerza (kg)',
            'P125': 'P. Tricipital (mm)',
            'P128': 'P. Pantorrilla (mm)',
            'P127': 'P. Biceps (mm)',
            'P126': 'P. Subescapular (mm)',
            'IMC': 'IMC',
            'P121': 'Cintura (cm)',
            'P123': 'Muslo (cm)',
            'P120': 'Brazo (cm)',
            'P124': 'Pantorrilla (cm)',
            'P117': 'Peso (kg)'
        }

        # Renombrar columnas
        df_combined_renamed = df_combined.rename(columns=column_labels)

        # Crear subplots
        if 'fig_histogramas' not in st.session_state:

            fig_6 = sp.make_subplots(rows=4, cols=3, subplot_titles=list(column_labels.values()))

            # Iterar por cada variable
            row, col = 1, 1
            for column in column_labels.values():
                # Determinar tamaño del bin
                min_val = df_combined_renamed[column].min()
                max_val = df_combined_renamed[column].max()
                bin_size = (max_val - min_val) / 10 if max_val != min_val else 1

                # Histograma para Hombres
                histogram_male = go.Histogram(
                    x=df_combined_renamed[df_combined['sexo'] == "Hombre"][column],
                    xbins=dict(size=bin_size),
                    name=f'Hombres - {column}',
                    opacity=0.6,
                    marker=dict(line=dict(width=1, color='blue')),
                    showlegend=False
                )

                # Histograma para Mujeres
                histogram_female = go.Histogram(
                    x=df_combined_renamed[df_combined['sexo'] == "Mujer"][column],
                    xbins=dict(size=bin_size),
                    name=f'Mujeres - {column}',
                    opacity=0.6,
                    marker=dict(line=dict(width=1, color='deeppink')),
                    showlegend=False
                )

                # Agregar al subplot
                fig_6.add_trace(histogram_male, row=row, col=col)
                fig_6.add_trace(histogram_female, row=row, col=col)

                # Avanzar en la cuadrícula
                col += 1
                if col > 3:
                    col = 1
                    row += 1

            # Configurar layout
            fig_6.update_layout(
                title_text="Histogramas por Sexo (Hombres vs Mujeres)",
                height=900,
                barmode='overlay',
                showlegend=True,
                legend=dict(x=1.02, y=1, traceorder='normal', borderwidth=0),
                margin=dict(r=120)  # margen derecho extra para leyenda
                )
            st.session_state.fig_histogramas = fig_6

        # Mostrar en Streamlit
        st.plotly_chart(st.session_state.fig_histogramas)

        #st.pyplot(st.session_state.fig_histogramas)

        #st.plotly_chart(fig, use_container_width=True)


########### mascara de varianzas

        from sklearn.feature_selection import VarianceThreshold

        # Standardizing the columns from the 4th column onwards in df_combined
        columns_to_standardize = df_combined.columns[4:]  # Selecting columns from the 4th column onwards

        # Calculating variance using the provided method: dividing by the mean and then calculating the variance
        features = df_combined[columns_to_standardize]  # Selecting the features to be standardized
        variances = (features / features.mean()).dropna().var()

        # Sorting variances in descending order
        variances = variances.sort_values(ascending=False)

        # Applying the variance threshold mask
        sel = VarianceThreshold(threshold=0.005)
        sel.fit(features / features.mean())

        # Creating a boolean mask based on the variance    
        mask = variances >= 0.005

        # Applying the mask to create a reduced DataFrame
        reduced_df = features.loc[:, mask]
        reduced_df.to_excel('reduced_df_hombres.xlsx', index=False)

        ##matriz de correlacion


        import streamlit as st
        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import MinMaxScaler
        import numpy as np
        import pandas as pd
    
        # --- Columnas seleccionadas ---
        selected_columns = ['P112_vel','P113', 'P125', 'P128', 'P127','P126','IMC','P121','P123','P120', 'P124']

        # --- Filtrar el DataFrame con esas columnas ---    
        numeric_df = reduced_df[selected_columns]

        # --- Normalizar con MinMaxScaler ---
        scaler = MinMaxScaler()    
        normalized_array = scaler.fit_transform(numeric_df)
        #normalized_df = pd.DataFrame(normalized_array, columns=selected_columns)
        normalized_df = pd.DataFrame(normalized_array, columns=[column_labels[col] for col in selected_columns])

        # --- Calcular matriz de correlación ---
        corr = normalized_df.corr(method='pearson')

        # --- Crear máscara triangular superior ---
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # --- Crear figura y graficar ---
        if 'fig_mapas' not in st.session_state:

            fig_7, ax = plt.subplots(figsize=(10, 8), dpi=150)
            cmap = sns.diverging_palette(h_neg=10, h_pos=240, as_cmap=True)
            sns.heatmap(
                corr,
                mask=mask,
                center=0,
                cmap=cmap,
                linewidths=1,
                annot=True,
                fmt='.2f',
                square=True,
                ax=ax
            )
            st.session_state.fig_mapas = fig_7  # Guardar en session_state            
            st.subheader("🔗 Mapa de correlaciones normalizadas")
        st.pyplot(st.session_state.fig_mapas)

        # Concatenating df_grouped with df_r to create a single DataFrame
        df_combined = pd.concat([df_datos, reduced_df], axis=1)
        df_combined = df_combined.dropna()
        #df_combined

        import networkx as nx
        import numpy as np
        import pandas as pd

        # 1. Definir las variables de interés
        selected_vars = ['P112_vel', 'P113', 'P117', 'P120','P121','P122','P123','P124','P125', 'P128', 'P127', 'P126', 'IMC']
        threshold = 0.3

        # 2. Separar por sexo    
        df_men = df_combined[df_combined['sexo'] == 'Hombre']    
        df_women = df_combined[df_combined['sexo'] == 'Mujer']

        # 3. Calcular matrices de correlación
        corr_men = df_men[selected_vars].corr()
        corr_women = df_women[selected_vars].corr()

        # 4. Crear sets de aristas
        edges_men = set()
        edges_women = set()

        for i, var1 in enumerate(selected_vars):
            for j, var2 in enumerate(selected_vars):
                if i < j:
                    if abs(corr_men.loc[var1, var2]) > threshold:
                        edges_men.add((var1, var2))
                    if abs(corr_women.loc[var1, var2]) > threshold:
                        edges_women.add((var1, var2))

        # 5. Determinar intersecciones
        edges_both = edges_men & edges_women
        edges_men_only = edges_men - edges_both
        edges_women_only = edges_women - edges_both

        # 6. Crear grafo final con etiquetas de grupo
        G = nx.Graph()

        for u, v in edges_men_only:
            weight = corr_men.loc[u, v]
            G.add_edge(u, v, weight=weight, group='Men')

        for u, v in edges_women_only:
            weight = corr_women.loc[u, v]
            G.add_edge(u, v, weight=weight, group='Women')

        for u, v in edges_both:
            weight = np.mean([corr_men.loc[u, v], corr_women.loc[u, v]])
            G.add_edge(u, v, weight=weight, group='Both')

        # 7. Identificar nodos por grupo
        nodes_men = {n for e in edges_men_only for n in e}
        nodes_women = {n for e in edges_women_only for n in e}
        nodes_both = {n for e in edges_both for n in e}

        # Eliminar duplicados
        nodes_men -= nodes_both    
        nodes_women -= nodes_both

        import networkx as nx
        import matplotlib.patches as mpatches

        # Layout Kamada-Kawai
        pos = nx.kamada_kawai_layout(G)

        # Colores de nodos
        node_colors = []
        for node in G.nodes():
            if node in nodes_both:
                node_colors.append('mediumvioletred')
            elif node in nodes_men:
                node_colors.append('steelblue')
            elif node in nodes_women:
                node_colors.append('lightcoral')
            else:
                node_colors.append('grey')

        # Tamaño de nodos por grado
        degree_dict = dict(G.degree())
        node_sizes = [400 + degree_dict[n] * 150 for n in G.nodes()]

        # Crear figura
        if 'fig_red_correlacion' not in st.session_state:

            fig_8, ax = plt.subplots(figsize=(14, 10), dpi=150, facecolor='white')

            # Dibujar nodos    
            nx.draw_networkx_nodes(
                G, pos,
                node_size=node_sizes,
                node_color=node_colors,
                edgecolors='black',
                linewidths=1.5,
                ax=ax
            )

            # Agrupar aristas por grupo
            edges_men = [(u, v, d) for u, v, d in G.edges(data=True) if d['group'] == 'Men']
            edges_women = [(u, v, d) for u, v, d in G.edges(data=True) if d['group'] == 'Women']
            edges_both = [(u, v, d) for u, v, d in G.edges(data=True) if d['group'] == 'Both']

            # Función para dibujar aristas según rangos
            def draw_edges(edges_list, color):
            # Punteadas: < 0.5
                dotted_edges = [(u, v) for u, v, d in edges_list if abs(d['weight']) < 0.5]
                dotted_widths = [abs(d['weight'])*4 for u, v, d in edges_list if abs(d['weight']) < 0.5]

                # Sólidas medias: 0.5 – 0.7
                medium_edges = [(u, v) for u, v, d in edges_list if 0.5 <= abs(d['weight']) <= 0.7]
                medium_widths = [abs(d['weight'])*5 for u, v, d in edges_list if 0.5 <= abs(d['weight']) <= 0.7]

                # Sólidas gruesas: > 0.7
                strong_edges = [(u, v) for u, v, d in edges_list if abs(d['weight']) > 0.7]
                strong_widths = [abs(d['weight'])*6 for u, v, d in edges_list if abs(d['weight']) > 0.7]

                # Dibujar
                nx.draw_networkx_edges(G, pos, edgelist=dotted_edges, width=dotted_widths,
                           edge_color=color, style='dashed', alpha=0.5, ax=ax)

                nx.draw_networkx_edges(G, pos, edgelist=medium_edges, width=medium_widths,
                           edge_color=color, style='solid', alpha=0.7, ax=ax)

                nx.draw_networkx_edges(G, pos, edgelist=strong_edges, width=strong_widths,
                           edge_color=color, style='solid', alpha=0.9, ax=ax)

            # Dibujar aristas por grupo
            draw_edges(edges_men, color='steelblue')
            draw_edges(edges_women, color='lightcoral')
            draw_edges(edges_both, color='mediumvioletred')

            # Etiquetas
            labels_translated = {n: column_labels_en.get(n, n) for n in G.nodes()}
            nx.draw_networkx_labels(
                G, pos,
                labels=labels_translated,
                font_size=11,
                font_family="sans-serif",
                ax=ax
            )

            # Leyend    a        
            legend_handles = [
                mpatches.Patch(color='steelblue', label='Hombres'),
                mpatches.Patch(color='lightcoral', label='Mujeres'),
                mpatches.Patch(color='mediumvioletred', label='Ambos')
            ]
            ax.legend(handles=legend_handles, loc='lower left', bbox_to_anchor=(1.05, 0.5), frameon=False, fontsize=11)

            # Título y ajustes    
            ax.set_title("Red de Correlación con Aristas Diferenciadas\npor Rango de Fuerza y Sexo", fontsize=16)
            ax.axis('off')
            fig_8.tight_layout()
            st.session_state.fig_red_correlacion = fig_8

            st.subheader("🔗 Red de correlación")
        st.pyplot(st.session_state.fig_red_correlacion)

        # Calcular estatura en cm a partir de peso (P117) e IMC
        df_combined['P118'] = ((df_combined['P117'] / df_combined['IMC'])**0.5) * 100
    
        df_combined['sexo'] = df_combined['sexo'].replace({'Hombre': 1.0, 'Mujer': 0.0})
        #df_combied_2 = df_combined.copy()
        # Modificar la función para calcular el Índice de Masa Muscular Esquelética (IMME)
        def calcular_IMME(row):
            CP = row['P124']  # Circunferencia de Pantorrilla en cm
            FP = row['P113']  # Fuerza de Prensión de la Mano en kg
            P = row['P117']  # Peso corporal en kg
            Sexo = row['sexo']  # Sexo (1.0 para hombres, 2.0 para mujeres)
            IMC = row['IMC']  # Índice de Masa Corporal (IMC)

            # Calcular la Talla (Altura en cm) a partir del IMC y el Peso
            Talla = np.sqrt(P / IMC)  # Talla en metros (no es necesario convertir a cm aquí)

            # Estimar la masa muscular esquelética (puede adaptarse según la referencia)
            masa_muscular = (
                0.215 * CP +  # Estimación con circunferencia de pantorrilla
                0.093 * FP +  # Estimación con fuerza de prensión
                0.061 * P +   # Estimación con peso corporal
                3.637 * Sexo  # Ajuste según el sexo
            )

            # Calcular el IMME (masa muscular dividida por talla al cuadrado)
            imme = masa_muscular / (Talla ** 2)

            return imme

        # Aplicar la función a cada fila del DataFrame
        df_combined['IMME'] = df_combined.apply(calcular_IMME, axis=1)
        #df_combined

        #import streamlit as st
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error
        from itertools import combinations
        import pandas as pd

        st.subheader("🔍 Selección de combinaciones óptimas de variables para predecir IMME")

        # Variables disponibles
        variables = ["sexo",'P117', 'P118', 'P119', 'P120', 'P121', 'P122', 'P123', 'P124',
             'P125', 'P126', 'P127', 'P128', 'P129', 'IMC', 'P113', 'P112_vel']

        # Recalcular estatura si es necesario
        if 'P118' not in df_combined.columns:
            df_combined['P118'] = ((df_combined['P117'] / df_combined['IMC'])**0.5) * 100

        # Selección de longitud específica para combinaciones a mostrar
        selected_n = st.number_input("Selecciona el número de variables en cada combinación a mostrar", min_value=1, max_value=len(variables), value=3)

        # Máximo tamaño a combinar en evaluación general
        max_combinaciones = 5


        if 'errores_combinaciones' not in st.session_state:
            st.session_state.errores_combinaciones = {}
            st.session_state.mejor_combinacion = None
            st.session_state.mejor_error = None
            st.session_state.resultados_filtrados = None
            st.session_state.modelo_global = None
            st.session_state.modelo_n = None

        # Calcula combinaciones
        errores_combinaciones = {}
        for r in range(1, max_combinaciones + 1):
            for combinacion in combinations(variables, r):
                X = df_combined[list(combinacion)]
                y = df_combined['IMME']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = DecisionTreeRegressor(random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                errores_combinaciones[combinacion] = mse

        # Guardar resultados en session_state
        st.session_state.errores_combinaciones = errores_combinaciones
        st.session_state.mejor_combinacion = min(errores_combinaciones, key=errores_combinaciones.get)
        st.session_state.mejor_error = errores_combinaciones[st.session_state.mejor_combinacion]

        # Usar mejor combinación almacenada
        mejor_combinacion = st.session_state.mejor_combinacion
        mejor_error = st.session_state.mejor_error
        errores_combinaciones = st.session_state.errores_combinaciones


        # Mostrar la mejor combinación global
        mejor_combinacion = min(errores_combinaciones, key=errores_combinaciones.get)
        mejor_error = errores_combinaciones[mejor_combinacion]

        st.markdown(f"### 🏆 Mejor combinación global:")
        st.markdown(f"- **Variables**: `{mejor_combinacion}`")
        st.markdown(f"- **Error cuadrático medio (MSE)**: `{mejor_error:.4f}`")

        # Mostrar combinaciones con el número exacto de variables elegido
        st.markdown(f"### 📊 Combinaciones con exactamente {selected_n} variables:")

        filtered_results = {k: v for k, v in errores_combinaciones.items() if len(k) == selected_n}
        sorted_results = sorted(filtered_results.items(), key=lambda x: x[1])

        df_resultados = pd.DataFrame([
            {'Variables': ', '.join(k), 'MSE': v}
            for k, v in sorted_results
        ])

        st.dataframe(df_resultados)

        # Obtener la mejor combinación con n variables seleccionadas por el usuario
        if not df_resultados.empty:
            mejor_combinacion_n = df_resultados.iloc[0]['Variables'].split(', ')
        else:
            mejor_combinacion_n = None

        

        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np



        if mejor_combinacion_n is not None:
            # Datos para la mejor combinación global
            X_global = df_combined[list(mejor_combinacion)]
            y = df_combined['IMME']
            Xg_train, Xg_test, yg_train, yg_test = train_test_split(X_global, y, test_size=0.2, random_state=42)
            modelo_global = DecisionTreeRegressor(random_state=42).fit(Xg_train, yg_train)
            st.session_state.modelo_global = modelo_global
            y_pred_global = modelo_global.predict(Xg_test)
            mse_global = mean_squared_error(yg_test, y_pred_global)

            # Datos para la mejor combinación con n variables
            X_n = df_combined[mejor_combinacion_n]
            Xn_train, Xn_test, yn_train, yn_test = train_test_split(X_n, y, test_size=0.2, random_state=42)
            modelo_n = DecisionTreeRegressor(random_state=42).fit(Xn_train, yn_train)
            y_pred_n = modelo_n.predict(Xn_test)
            mse_n = mean_squared_error(yn_test, y_pred_n)

            # Crear DataFrames para comparación
            df_comparacion_global = pd.DataFrame({
                'IMME Real': yg_test,
                'IMME Predicho': y_pred_global
            })

            df_comparacion_n = pd.DataFrame({
                'IMME Real': yn_test,
                'IMME Predicho': y_pred_n
            })

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            for i, (df_cmp, title, mse_val) in enumerate([
                (df_comparacion_global, '🌳 Mejor Modelo Global', mse_global),
                (df_comparacion_n, f'📉 Mejor Modelo con {selected_n} variables', mse_n)
            ]):

                # Calcular errores individuales
                errores_individuales = np.abs(df_cmp['IMME Real'] - df_cmp['IMME Predicho'])

                # Dispersión con código de color por error
                sc = axes[i].scatter(df_cmp['IMME Real'], df_cmp['IMME Predicho'],
                     c=errores_individuales, cmap='RdYlGn_r', s=50, alpha=0.8)

                # Línea y = x
                min_val = min(df_cmp.min().min(), df_cmp.max().max())
                max_val = max(df_cmp.min().min(), df_cmp.max().max())
                axes[i].plot([min_val, max_val], [min_val, max_val],
                color='black', linestyle='--', label='y = x')

                # Líneas de error
                for j in range(len(df_cmp)):
                    real = df_cmp['IMME Real'].iloc[j]
                    pred = df_cmp['IMME Predicho'].iloc[j]
                    axes[i].plot([real, real], [real, pred], color='gray', alpha=0.3)
               

                # Títulos y etiquetas
                axes[i].set_title(f"{title}\nMSE: {mse_val:.4f}", fontsize=12)
                axes[i].set_xlabel('IMME Real (Fórmula)')
                axes[i].set_ylabel('IMME Predicho (Árbol)')
                axes[i].legend()

            st.pyplot(fig)

            st.markdown("### 🧪 Comparar predicciones con valores personalizados")

            # Inicializar session_state si no existe
            if 'valores_usuario' not in st.session_state:
                st.session_state.valores_usuario = {}
            if 'modelo_usado' not in st.session_state:
                st.session_state.modelo_usado = None
            if 'prediccion_realizada' not in st.session_state:
                st.session_state.prediccion_realizada = False
            if 'prediccion_valor' not in st.session_state:
                st.session_state.prediccion_valor = None


            # Selección del modelo a usar
            modelo_seleccionado = st.radio(
                "Selecciona el modelo para usar en la predicción:",
                ("Mejor combinación global", f"Mejor combinación con {selected_n} variables", f"Elegir manualmente {selected_n} variables"),
                horizontal=True
            )

            # Obtener modelo y variables según selección
            modelo = None
            variables_input = []

            if modelo_seleccionado == "Mejor combinación global":
                modelo = st.session_state.modelo_global
                variables_input = list(st.session_state.mejor_combinacion)

            elif modelo_seleccionado.startswith("Mejor combinación con"):
                modelo = st.session_state.modelo_n
                variables_input = list(st.session_state.mejor_combinacion_n)

            else:
                # Elección manual con multiselect
                variables_disponibles = variables  # debe estar definido antes
                seleccion_manual = st.multiselect(
                    f"Selecciona exactamente {selected_n} variables:",
                    options=variables_disponibles,
                    default=[],
                    key="manual_vars"
                )
                if len(seleccion_manual) != selected_n:
                    st.warning(f"Selecciona exactamente {selected_n} variables para continuar.")
                else:
                    variables_input = seleccion_manual
                    # Entrenar modelo si es nuevo
                    if "modelo_manual" not in st.session_state or st.session_state.variables_manual != seleccion_manual:
                        X_manual = df_combined[seleccion_manual]
                        y_manual = df_combined['IMME']
                        X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_manual, y_manual, test_size=0.2, random_state=42)
                        modelo_manual = DecisionTreeRegressor(random_state=42).fit(X_train_m, y_train_m)
                        st.session_state.modelo_manual = modelo_manual
                        st.session_state.variables_manual = seleccion_manual
                    modelo = st.session_state.modelo_manual

            # Diccionario de nombres amigables
            nombres_amigables = {
                'P117': 'Peso (kg)',
                'P118': 'Estatura (cm)',
                'P119': 'Circunferencia de cintura',
                'P120': 'Circunferencia de cadera',
                'P121': 'Circunferencia de brazo',
                'P122': 'Pliegue tricipital',
                'P123': 'Pliegue subescapular',
                'P124': 'Circunferencia de pantorrilla',
                'P125': 'Pliegue abdominal',
                'P126': 'Pliegue suprailíaco',
                'P127': 'Pliegue muslo',
                'P128': 'Pliegue pierna',
                'P129': 'Pliegue pectoral',
                'IMC': 'Índice de Masa Corporal',
                'P113': 'Fuerza de prensión',
                'P112_vel': 'Velocidad de marcha',
                'sexo': 'Sexo (Mujer/Hombre)'
            }

            # Formulario para ingresar valores
            st.markdown("### ✍️ Introduce los valores para las siguientes variables:")
            input_values = {}

            for var in variables_input:
                unique_key = f"input_{var}_{modelo_seleccionado.replace(' ', '_')}"

                if var == 'sexo':
                    sexo_valor = st.selectbox("Sexo", options=["Mujer", "Hombre"], key=unique_key)
                    input_values[var] = 1.0 if sexo_valor == "Hombre" else 0.0
                else:
                    label = nombres_amigables.get(var, var)
                    input_values[var] = st.number_input(
                        label=label,
                        key=unique_key,
                        value=0.0
                    )

            # Botón para predecir
            if st.button("Predecir IMME"):
                input_df = pd.DataFrame([input_values])
                pred = modelo.predict(input_df)[0]
                st.success(f"🧠 IMME estimado: **{pred:.2f}**")


    
        
        else:
                st.warning("⚠️ No hay combinaciones disponibles con ese número de variables.")



    with st.expander("Agrupación por clusters"):
        import streamlit as st
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import AgglomerativeClustering

        
        # Selección de sexo
        sexo = st.radio("Selecciona el sexo para el análisis", ('Hombres', 'Mujeres'))
        df_filtered = df_combined[df_combined['sexo'] == (0 if sexo == 'Mujeres' else 1)]

        # Selección del número de cluster    s        
        num_clusters = st.number_input("Número de clústeres", min_value=2, max_value=10, value=4)

        # Renombrar columnas necesarias
        df_filtered = df_filtered.rename(columns={
            'P113': 'Fuerza',
            'P112_vel': 'Marcha',
            'IMME': 'IMME'
        })

        def aplicar_clustering(df, variable, quintil_maximo, etiqueta):
            """
            Aplica clustering jerárquico y elimina los individuos por encima del quintil indicado.
            Devuelve el DataFrame filtrado, el eliminado y las etiquetas de clúster.
            """
            datos = df[[variable]].dropna()
            scaler = StandardScaler()
            datos_normalizados = scaler.fit_transform(datos)

            clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
            etiquetas = clustering.fit_predict(datos_normalizados)

            df_aux = df.copy()
            df_aux['Cluster'] = etiquetas

            # Calcular el segundo quintil global
            q2 = df_aux[variable].quantile(quintil_maximo)

            # Filtrar los que están por debajo o igual al segundo quintil
            df_filtrado = df_aux[df_aux[variable] <= q2].copy()
            df_eliminado = df_aux[df_aux[variable] > q2].copy()
            df_filtrado['Clasificación Sarcopenia'] = etiqueta

            return df_filtrado, df_eliminado

        # Paso 1: Clustering por Fuerza
        #df_fuerza, _ = aplicar_clustering(df_filtered, 'Fuerza', 0.40, 'Sarcopenia Sospechosa')
        df_fuerza, df_elim1 = aplicar_clustering(df_filtered, 'Fuerza', 0.40, 'Sarcopenia Sospechosa')
        pct_elim1 = 100 * len(df_elim1) / len(df_filtered)

        # Paso 2: Clustering por IMME
        if not df_fuerza.empty:
            #df_imme, _ = aplicar_clustering(df_fuerza, 'IMME', 0.40, 'Sarcopenia Probable')
            df_imme, df_elim2 = aplicar_clustering(df_fuerza, 'IMME', 0.40, 'Sarcopenia Probable')
            pct_elim2 = 100 * len(df_elim2) / len(df_fuerza)
        else:
            #df_imme = pd.DataFrame()
            df_imme, df_elim2, pct_elim2 = pd.DataFrame(), pd.DataFrame(), 0


        # Paso 3: Clustering por Marcha
        if not df_imme.empty:
            #df_marcha, _ = aplicar_clustering(df_imme, 'Marcha', 0.40, 'Sarcopenia Grave')
            df_marcha, df_elim3 = aplicar_clustering(df_imme, 'Marcha', 0.40, 'Sarcopenia Grave')
            pct_elim3 = 100 * len(df_elim3) / len(df_imme)
        else:
            #df_marcha = pd.DataFrame()
            df_marcha, df_elim3, pct_elim3 = pd.DataFrame(), pd.DataFrame(), 0


        # 7) Mostrar resumen de eliminación
        resumen = pd.DataFrame({
            'Etapa': ['Fuerza', 'IMME', 'Marcha'],
            'Eliminados': [len(df_elim1), len(df_elim2), len(df_elim3)],
            '% Eliminados': [pct_elim1, pct_elim2, pct_elim3]
        })
        st.subheader("Resumen de cribado por etapas")
        st.dataframe(resumen.style.format({'% Eliminados':'{:.1f}%'}))

        
        # Combinar resultados
        df_resultado = pd.concat([
            df_fuerza[~df_fuerza.index.isin(df_imme.index)][['Fuerza', 'IMME', 'Marcha', 'Clasificación Sarcopenia']],
            df_imme[~df_imme.index.isin(df_marcha.index)][['Fuerza', 'IMME', 'Marcha', 'Clasificación Sarcopenia']],
            df_marcha[['Fuerza', 'IMME', 'Marcha', 'Clasificación Sarcopenia']]
        ])

        # Agregar etiquetas a df_filtered (las que sobrevivieron al cribado)
        df_filtered['Clasificación Sarcopenia'] = df_resultado['Clasificación Sarcopenia']
        df_filtered
        
        # Mostrar resultados
        st.subheader("Resultados de Clasificación de Sarcopenia")
        if not df_resultado.empty:
            st.write(df_resultado)
        else:
            st.warning("No se identificaron individuos con criterios de sarcopenia bajo los filtros establecidos.")


        #df_combined
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        # Filtrar total por sexo (1 = hombres, 0 = mujeres)
        #sexo_elegido = 1  # Simular que el 
        #df_total_sexo = df_filtered.shape[0]
        #total_pacientes = len(df_total_sexo)
        total_pacientes = df_filtered.shape[0]
        total_pacientes
        # Contar pacientes por clasificación
        conteos = df_resultado['Clasificación Sarcopenia'].value_counts()
        sospechosa = conteos.get('Sarcopenia Sospechosa', 0)
        probable = conteos.get('Sarcopenia Probable', 0)
        grave = conteos.get('Sarcopenia Grave', 0)
        saludables = total_pacientes - sospechosa
    
        # Crear figura
        # === Crear gráfico con matplotlib ===
        fig, ax = plt.subplots(figsize=(8, 8))

        # Círculos concéntricos jerárquicos
        circle_salud = patches.Circle((0.5, 0.5), 0.45, color='green', alpha=0.2)
        circle_sospecha = patches.Circle((0.5, 0.5), 0.35, color='yellow', alpha=0.3)
        circle_probable = patches.Circle((0.5, 0.5), 0.25, color='orange', alpha=0.4)
        circle_grave = patches.Circle((0.5, 0.5), 0.15, color='red', alpha=0.5)

        # Agregar círculos
        for circle in [circle_salud, circle_sospecha, circle_probable, circle_grave]:
            ax.add_patch(circle)

        # Etiquetas con valores y porcentajes
        ax.text(0.5, 0.91, f'Saludables: {saludables} ({saludables/total_pacientes:.0%})',
        ha='center', fontsize=12, color='green')
        ax.text(0.5, 0.77, f'Sospechosa: {sospechosa} ({sospechosa/total_pacientes:.0%})',
        ha='center', fontsize=12, color='goldenrod')
        ax.text(0.5, 0.63, f'Probable: {probable} ({probable/total_pacientes:.0%})',
        ha='center', fontsize=12, color='darkorange')
        ax.text(0.5, 0.50, f'Grave: {grave} ({grave/total_pacientes:.0%})',
        ha='center', fontsize=12, color='darkred')

        # Estética del gráfico
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.title("Clasificación Jerárquica de Sarcopenia", fontsize=14)

        # === Mostrar en Streamlit ===
        st.subheader("Visualización Jerárquica de Sarcopenia")
        st.pyplot(fig)












        
#        df_combined

#        import matplotlib.pyplot as plt
#        from sklearn.preprocessing import StandardScaler
#        from sklearn.decomposition import PCA
#        from sklearn.cluster import AgglomerativeClustering
#        from scipy.spatial.distance import pdist, squareform
#        from sklearn.metrics import silhouette_score


#        # Selección del sexo
#        sexo = st.radio("Selecciona el sexo para el análisis", (0, 1), format_func=lambda x: 'Mujeres' if x == 0 else 'Hombres')
#        df_combined_2 = df_combined
#        # Filtrar los datos según el sexo seleccionado
#        df_filtered = df_combined_2[df_combined_2['sexo'] == sexo]

#        # Mostrar algunos datos filtrados para validación
#        st.write(f"Mostrando datos para: {'Mujeres' if sexo == 0 else 'Hombres'}")
#        st.write(df_filtered.head())

#        # Verificar si hay datos después del filtro
#        if df_filtered.empty:
#            st.error('No hay datos disponibles para este sexo. Intenta con otro.')
#        else:
#            # Seleccionar las columnas y normalizar los datos
#            selected_columns = ['P113']  # Aquí puedes seleccionar las columnas que necesitas
#            numeric_data_2 = df_filtered[selected_columns].dropna()
    
#            # Normalización de los datos
#            scaler = StandardScaler()
#            normalized_data_2 = scaler.fit_transform(numeric_data_2)

#            # Aplicar PCA para reducir la dimensionalidad
#            pca = PCA(n_components=1)
#            pca_data = pca.fit_transform(normalized_data_2)

#            # Calcular la matriz de distancias
#            distance_matrix = squareform(pdist(pca_data))

#            # Aplicar Agglomerative Clustering
#            avg_distances = []
#            silhouettes = []
#            K = range(2, 15)
    
#            for k in K:
#                clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
#                labels = clustering.fit_predict(pca_data)

#                # Calcular la distancia intra-cluster
#                intra_cluster_distances = []
#                for cluster in range(k):
#                    cluster_points = distance_matrix[np.ix_(labels == cluster, labels == cluster)]
#                    intra_cluster_distances.append(np.mean(cluster_points))

#                avg_distances.append(np.mean(intra_cluster_distances))

                # Calcular el Silhouette Score
#                silhouette_avg = silhouette_score(pca_data, labels)
#                silhouettes.append(silhouette_avg)

#            # Graficar el método del codo
#            st.subheader('Método del Codo')
#            fig, ax = plt.subplots(figsize=(8, 6))
#            ax.plot(K, avg_distances, 'bo-')
#            ax.set_xlabel('Número de clusters (k)')
#            ax.set_ylabel('Distancia intra-cluster promedio')
#            ax.set_title('Método del codo para Agglomerative Clustering')
#            st.pyplot(fig)


#            # Graficar el Silhouette Score
#            st.subheader('Silhouette Score')
#            fig, ax = plt.subplots(figsize=(8, 6))
#            ax.plot(K, silhouettes, 'go-')
#            ax.set_xlabel('Número de clusters (k)')
#            ax.set_ylabel('Silhouette Score')
#            ax.set_title('Silhouette Score para Agglomerative Clustering')
#            st.pyplot(fig)

#            import streamlit as st
#            import pandas as pd    
#            import numpy as np
#            from sklearn.preprocessing import StandardScaler
#            from sklearn.cluster import AgglomerativeClustering

            # Suponiendo que df_combined_2 ya está disponible como DataFrame
            # df_combined_2 = pd.read_csv('path_to_your_data.csv')

#            # Título de la aplicación
#            st.title('Clustering Jerárquico Aglomerativo')

#            # Opción de filtro por sexo
#            sexo = st.radio("Selecciona el sexo para el análisis", ('Hombres', 'Mujeres'))

#            # Filtrar el DataFrame según el sexo seleccionado (0 para mujeres, 1 para hombres)
#            if sexo == 'Mujeres':
#                df_filtered = df_combined_2[df_combined_2['sexo'] == 0]
#            else:
#                df_filtered = df_combined_2[df_combined_2['sexo'] == 1]

#            # Mostrar un resumen de los datos filtrados
#            st.write(f"Mostrando datos para: {sexo}")
#            st.write(df_filtered.head())

#            # Selección de las columnas para el análisis
#            selected_columns = ['P113']  # Cambia según las columnas que necesites
#            numeric_data_2 = df_filtered[selected_columns]

#            # Eliminar valores no numéricos y valores faltantes
#            numeric_data_2 = numeric_data_2.dropna()

#            # Normalizar los datos
#            scaler = StandardScaler()
#            normalized_data_2 = scaler.fit_transform(numeric_data_2)

#            # Entrada del número de clusters (por defecto 4)
#            num_clusters = st.number_input("Número de clusters:", min_value=2, max_value=10, value=4)

#            # Aplicar Agglomerative Clustering con el número de clusters especificado        
#            clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
#            labels_2019 = clustering.fit_predict(normalized_data_2)

#            # Agregar las etiquetas de clúster al DataFrame original
#            df_filtered['Cluster'] = labels_2019

#            # Mostrar el DataFrame con las etiquetas de los clústeres
#            st.write("Datos con Clustering aplicado:")
#            st.write(df_filtered)

#            # Opcional: Mostrar la cantidad de elementos en cada clúster
#            st.write("Cantidad de elementos por clúster:")
#            st.write(df_filtered['Cluster'].value_counts())


#            # Renombrar las columnas como se mencionó
#            df_filtered = df_filtered.rename(columns={
#                'P112_vel': 'Marcha',
#                'P113': 'Fuerza',
#                'P125': 'P. Tricipital',
#                'P128': 'P. Pantorrilla',
#                'IMC': 'IMC',
#                'P127': 'Biceps',
#                'P126': 'P. subescapular',
#                'P121': 'Cintura',
#                'P123': 'Muslo',
#                'P120': 'Brazo',
#                'P122': 'Cadera',
#                'P124': 'Pantorrilla',
#                'P117': 'Peso'
#            })

#            # Seleccionar las columnas específicas con los nuevos nombres
#            selected_columns_renamed = [
#                'Marcha', 'Fuerza', 'P. Tricipital', 'P. Pantorrilla',
#                'IMC', 'Biceps', 'P. subescapular', 'Cintura', 'Muslo', 'Brazo', 'Cadera', 'Pantorrilla', 'Peso', 'IMME'
#            ]

#            # Filtrar el DataFrame para incluir solo las columnas seleccionadas
#            numeric_columns = df_filtered[selected_columns_renamed]

#            # Crear un gráfico de caja individual para cada parámetro y comparar los clusters
#            for column in numeric_columns.columns:
                # Obtener los datos de cada cluster para el parámetro actual
#                cluster_data = [df_filtered[df_filtered['Cluster'] == cluster][column] for cluster in range(8)]

                # Calcular los quintiles (Q1=20%, Q2=40%, mediana=Q3=60%, Q4=80%)
#                quintile_1 = df_filtered[column].quantile(0.20)
#                quintile_2 = df_filtered[column].quantile(0.40)
#                quintile_3 = df_filtered[column].quantile(0.60)
#                quintile_4 = df_filtered[column].quantile(0.80)

#                # Crear una nueva figura para el gráfico de caja
#                fig = go.Figure()

#                # Agregar el gráfico de caja para cada cluster
#                for j in range(8):  # Cambié de 6 a 8, ya que usas hasta 8 clusters
#                    fig.add_trace(go.Box(y=cluster_data[j], boxpoints='all', notched=True, name=f'Cluster {j}'))

#                # Agregar líneas horizontales para los quintiles
#                fig.add_shape(type="line",
#                  x0=0, x1=1, y0=quintile_1, y1=quintile_1,
#                  line=dict(color="blue", width=2, dash="dash"),
#                  xref="paper", yref="y")  # Línea del primer quintil (Q1 = 20%)

#                fig.add_shape(type="line",
#                  x0=0, x1=1, y0=quintile_2, y1=quintile_2,
#                  line=dict(color="green", width=2, dash="dash"),
#                  xref="paper", yref="y")  # Línea del segundo quintil (Q2 = 40%)

#                fig.add_shape(type="line",
#                  x0=0, x1=1, y0=quintile_3, y1=quintile_3,
#                  line=dict(color="orange", width=2, dash="dash"),
#                  xref="paper", yref="y")  # Línea de la mediana (Q3 = 60%)

#                fig.add_shape(type="line",
#                  x0=0, x1=1, y0=quintile_4, y1=quintile_4,
#                  line=dict(color="red", width=2, dash="dash"),
#                  xref="paper", yref="y")  # Línea del cuarto quintil (Q4 = 80%)

#                # Actualizar el diseño y mostrar cada gráfico de caja individual
#                fig.update_layout(title_text=f'Comparación de Clusters - {column}',
#                      xaxis_title="Clusters",
#                      yaxis_title=column,
#                      showlegend=False)
#                st.plotly_chart(fig)  # Usamos st.plotly_chart para integrar el gráfico en Streamlit


#            import streamlit as st
#            import pandas as pd
#            import matplotlib.pyplot as plt

#            # Calcular el percentil 40 global de 'Fuerza' en todo df_filtered
#            percentile_40_fuerza = df_filtered['Fuerza'].quantile(0.40)

#            # Crear un DataFrame vacío para almacenar las filas que cumplen la condición
#            df_filtered_result = pd.DataFrame()
#            percentages_deleted = {}

#            # Iterar sobre cada clúster y aplicar el filtro
#            for cluster in df_filtered['Cluster'].unique():
#                # Filtrar el DataFrame por cada cluster
#                cluster_data = df_filtered[df_filtered['Cluster'] == cluster]
#                # Mantener solo las filas con 'Fuerza' menor o igual al percentil 40 global
#                filtered_cluster_data = cluster_data[cluster_data['Fuerza'] <= percentile_40_fuerza]
#                # Agregar las filas filtradas al nuevo DataFrame
#                df_filtered_result = pd.concat([df_filtered_result, filtered_cluster_data])
#                # Calcular el porcentaje de filas eliminadas en cada cluster
#                percentage_deleted = 100 * (1 - len(filtered_cluster_data) / len(cluster_data))
#                percentages_deleted[cluster] = percentage_deleted

#            # Convertir los porcentajes a un DataFrame para visualizar
#            percentages_df = pd.DataFrame(list(percentages_deleted.items()), columns=['Cluster', 'Percentage Deleted'])

#            # Mostrar el DataFrame con los datos filtrados en Streamlit
#            st.write("Datos filtrados según el percentil 40 de 'Fuerza':")
#            st.write(df_filtered_result)

#            # Mostrar el porcentaje de filas eliminadas por clúster    
#            st.write("Porcentaje de filas eliminadas por clúster:")
#            st.write(percentages_df)

#            # Crear el diagrama de barras para mostrar el porcentaje de filas eliminadas por clúster
#            fig, ax = plt.subplots(figsize=(10, 6))
#            ax.bar(percentages_df['Cluster'], percentages_df['Percentage Deleted'], color='purple', alpha=0.7)
#            ax.set_xlabel('Cluster')
#            ax.set_ylabel('Porcentaje de Filas Eliminadas')
#            ax.set_title('Porcentaje de Filas Eliminadas por Cluster (Fuerza <= Percentil 40%)')

#            # Mostrar el gráfico en Streamlit
#            st.pyplot(fig)


############################

#            import streamlit as st
#            import pandas as pd
#            import numpy as np
#            from sklearn.preprocessing import StandardScaler
#            from sklearn.cluster import AgglomerativeClustering



#            # Selección de columnas para el análisis
#            selected_columns = st.multiselect(
#                'Selecciona las columnas para el análisis de clustering',
#                options=df_filtered.columns,
#                default=['IMME']  # Esta es la columna por defecto
#            )

#            # Filtrar el DataFrame para incluir solo las columnas seleccionadas
#            numeric_data_2 = df_filtered[selected_columns]

#            # Eliminar valores no numéricos y valores faltantes
#            numeric_data_2 = numeric_data_2.dropna()

#            # Normalizar los datos
#            scaler = StandardScaler()
#            normalized_data_2 = scaler.fit_transform(numeric_data_2)

#            # Número de clusters, con valor por defecto 4
#            num_clusters = st.number_input("Núm de clusters:", min_value=2, max_value=10, value=4)

#            # Aplicar Agglomerative Clustering
#            clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
#            labels_2019 = clustering.fit_predict(normalized_data_2)

#            # Agregar las etiquetas al DataFrame original filtrado
#            df_filtered['Cluster'] = labels_2019

#            # Mostrar el DataFrame con los resultados del clustering
#            st.write("Datos con las etiquetas de clúster agregadas:")    
#            st.write(df_filtered.head())

#            # Opcional: Mostrar la cantidad de elementos por clúster
#            st.write("Cantidad de elementos por clúster:")
#            st.write(df_filtered['Cluster'].value_counts())


##################

#            import streamlit as st
#            import pandas as pd
#            import plotly.graph_objects as go

 

#            # Filtrar el DataFrame para incluir solo las columnas seleccionadas
#            numeric_columns_2 = df_filtered[selected_columns_renamed]

#            # Crear un gráfico de caja individual para cada parámetro y comparar los clusters
#            for column in numeric_columns_2.columns:
                # Obtener los datos de cada cluster para el parámetro actual
#                cluster_data = [df_filtered[df_filtered['Cluster'] == cluster][column] for cluster in range(df_filtered['Cluster'].nunique())]

#                # Calcular los quintiles (Q1=20%, Q2=40%, mediana=Q3=60%, Q4=80%)
#                quintile_1 = df_filtered[column].quantile(0.20)
#                quintile_2 = df_filtered[column].quantile(0.40)
#                quintile_3 = df_filtered[column].quantile(0.60)
#                quintile_4 = df_filtered[column].quantile(0.80)

#                # Crear una nueva figura para el gráfico de caja
#                fig = go.Figure()

#                # Agregar el gráfico de caja para cada cluster
#                for j in range(df_filtered['Cluster'].nunique()):
#                    fig.add_trace(go.Box(y=cluster_data[j], boxpoints='all', notched=True, name=f'Cluster {j}'))

#                # Agregar líneas horizontales para los quintiles
#                fig.add_shape(type="line",
#                  x0=0, x1=1, y0=quintile_1, y1=quintile_1,
#                  line=dict(color="blue", width=2, dash="dash"),
#                  xref="paper", yref="y")  # Línea del primer quintil (Q1 = 20%)

#                fig.add_shape(type="line",
#                  x0=0, x1=1, y0=quintile_2, y1=quintile_2,
#                  line=dict(color="green", width=2, dash="dash"),
#                  xref="paper", yref="y")  # Línea del segundo quintil (Q2 = 40%)

#                fig.add_shape(type="line",
#                  x0=0, x1=1, y0=quintile_3, y1=quintile_3,
#                  line=dict(color="orange", width=2, dash="dash"),
#                  xref="paper", yref="y")  # Línea de la mediana (Q3 = 60%)

#                fig.add_shape(type="line",
#                  x0=0, x1=1, y0=quintile_4, y1=quintile_4,
#                  line=dict(color="red", width=2, dash="dash"),
#                  xref="paper", yref="y")  # Línea del cuarto quintil (Q4 = 80%)

#                # Actualizar el diseño y mostrar cada gráfico de caja individual
#                fig.update_layout(title_text=f'Comparación de Clusters - {column}',
#                      xaxis_title="Clusters",
#                      yaxis_title=column,
#                      showlegend=False)

#                # Mostrar el gráfico en Streamlit
#                st.plotly_chart(fig)

######################

#            import pandas as pd
#            import matplotlib.pyplot as plt

#            # Calcular el percentil 60 global de 'Fuerza' en todo df_combined_hombres
#            percentile_40_IMME = df_combined_2['IMME'].quantile(0.40)

            # Crear un DataFrame vacío para almacenar las filas que cumplen la condición
#            df_filtered_2 = pd.DataFrame()
#            percentages_deleted = {}

#            for cluster in df_filtered['Cluster'].unique():
#                # Filtrar el DataFrame por cada cluster
#                cluster_data = df_filtered[df_filtered['Cluster'] == cluster]
#                # Mantener solo las filas con 'Fuerza' menor o igual al percentil 60 global
#                filtered_cluster_data = cluster_data[cluster_data['IMME'] <= percentile_40_IMME]
#                # Agregar las filas filtradas al nuevo DataFrame
#                df_filtered_2 = pd.concat([df_filtered_2, filtered_cluster_data])
#                # Calcular el porcentaje de filas eliminadas en cada cluster
#                percentage_deleted = 100 * (1 - len(filtered_cluster_data) / len(cluster_data))
#                percentages_deleted[cluster] = percentage_deleted

#            # Convertir los porcentajes a un DataFrame para visualizar
#            percentages_df = pd.DataFrame(list(percentages_deleted.items()), columns=['Cluster', 'Percentage Deleted'])

#            # Mostrar el DataFrame con los datos filtrados

#            # Crear el diagrama de barras para mostrar el porcentaje de filas eliminadas por cluster    
#            plt.figure(figsize=(10, 6))
#            plt.bar(percentages_df['Cluster'], percentages_df['Percentage Deleted'], color='purple', alpha=0.7)
#            plt.xlabel('Cluster')
#            plt.ylabel('Porcentaje de Filas Eliminadas')
#            plt.title('Porcentaje de Filas Eliminadas por Cluster (IMME > Quintil 40% Global)')
#            plt.show()





except Exception as e:
    st.error(f"Ocurrió un error al intentar cargar el archivo: {e}")

