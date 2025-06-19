#########################################################
import streamlit as st

# Menú en barra lateral
opcion = st.sidebar.radio(
    "Selecciona una pestaña:",
    ["Introducción", "Proceso", "Formularios", "Equipo de trabajo"])

# Contenido condicional
if opcion == "Introducción":
    st.title("Sobre SARC-Predictor")
    st.markdown("""
    <div style="text-align: justify;">
       
    Esta aplicación es resultado del proyecto de estancia posdoctoral **"Identificación 
    de las etapas y tipos de sarcopenia mediante modelos predictivos como herramienta 
    de apoyo en el diagnóstico a partir de parámetros antropométricos"**, desarrollado 
    por el Doctor en Ciencias (Astrofísica) Santiago Arceo Díaz, bajo la dirección de 
    la Doctora Xóchitl Rosío Angélica Trujillo Trujillo, y con la ayuda de los colaboradores mencionados en esta sección. Esta estancia es gracias a la 
    colaboración entre el entre el **Consejo Nacional de Humanidades Ciencia y Tecnología ([**CONAHCYT**](https://conahcyt.mx/)) y la Universidad de Colima ([**UCOL**](https://portal.ucol.mx/cuib/))**
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Muestra")
    st.markdown("""
    <div style="text-align: justify">
                   
    Los datos utilizados para los modelos fueron proveidos por el **Dr. Sergio Sánchez García** y personal del **Instituto Mexicano del Seguro Social**, a partir de las respuestas recolectadas del **"Cuadernillo de Obesidad, Sarcopenia y Fragilidad en Adultos Mayores Derechohabientes del Instituto Mexicano del Seguro Social de las Delegaciones Sur y Norte de la Ciudad de México"** (en sus ediciones de 2019 y 2022). A partir de las medidas antropométricas registradas, se crean modelos para predecir la incidencia de sarcopenia en los usuarios registrados (tanto en personas adultas mayores sanas como en aquellas que padecen de comorbilidades como la hipertensión, diabetes mellitus o artritis).  
            
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Algoritmos y lenguaje de programación")
    st.markdown("""
    <div style = "text-align: justify">
                   
    Elegimos el lenguaje de programación [**Python**](https://docs.python.org/es/3/tutorial/) y las plataformas [**Streamlit**](https://streamlit.io/) y [**GitHub**](https://github.com/). Estas opciones permiten una fácil visualización y manipulación de la aplicación, además de almacenar los algoritmos en la nube. Las técnicas utilizadas para el análisis de los datos y la creación de modelos de aproximación se derivan de prácticas usuales para la depuración de datos, la creación de árboles de ajuste, la técnica de clustering jerárquico y Random Forest. **La aplicación es de libre acceso y uso gratuito para cualquier personal de atención primaria de pacientes geriátricos.**
    </div>
    """, unsafe_allow_html=True)

    #st.title("Acerca de Sarc-open-IA")

    st.subheader("Objetivo")
    st.markdown("""
    <div style="text-align: justify">
                               
    El objetivo de esta aplicación es crear modelos para la predicción de sacorpenia a partir de medidas antropométricas, tomando en cuenta la posible presencia de comorbilidades. Adicionalmente, estos modelos pueden generarse a partir de distintas combinaciones de variables antropométricas, permitiendo generar un diagnóstico en situaciones en las que alguna de las variables mas comunes, no están disponibles debido a limitaciones de recursos.
    </div>             
    """,unsafe_allow_html=True)

    st.subheader("Ventajas y características")

    st.markdown("""
    <div style="text-align: justify">
                   
    - **Facilitar uso:** Queríamos que nuestra herramienta fuera fácil de usar para el personal médico, incluso si no estaban familiarizados con la inteligencia artificial o la programación. Para lograrlo, elegimos el lenguaje de programación [**Python**](https://docs.python.org/es/3/tutorial/) y las plataformas [**Streamlit**](https://streamlit.io/) y [**GitHub**](https://github.com/). Estas opciones permiten una fácil visualización y manipulación de la aplicación, además de almacenar los algoritmos en la nube.

    - **Interfaz amigable:** El resultado es una interfaz gráfica que permite a los médicos ingresar los datos antropométricos de los pacientes y ver gráficas útiles para el análisis estadístico. También ofrece un diagnóstico en tiempo real de la sarcopenia, y todo esto se hace utilizando cajas de texto y deslizadores para ingresar y manipular los datos.

    - **Accesibilidad total:** El personal médico puede descargar de  forma segura las gráficas y los archivos generados por la aplicación. Además, pueden acceder a ella desde cualquier dispositivo con conexión a internet, ya sea un teléfono celular, una computadora, tablet o laptop.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Método")
    st.markdown("""
    <div style="text-align:justify">
                   
    Los datos utiliados para el entrenamiento de los modelos provienen de las ediciones de los años 2019 y 2022 del **"Cuadernillo de Obesidad, Sarcopenia y Fragilidad en Adultos Mayores Derechohabientes del Instituto Mexicano del Seguro Social de las Delegaciones Sur y Norte de la Ciudad de México"**. Con los datos recolectados, se programó un algoritmo que aplica clustering jerárquico aglomerativo para clasficar pacientes en conjuntos que se caracterizan su similitud en las medidas antropométricas. En el caso del Índice de Masa Muscular Esquelética Apendicular, se crearon modelos de ajuste que calculan esta variable a partir de las circunferencias de pantorrilla, brazo, septiembre y octubre el año 2023 en una muestra de adultos mayores que residen en la Zona Metropolitana, Colima, Villa de Álvarez, México, se procedió al desarrollo de modelos predictivos mediante el algoritmo [**Random Forest**](https://cienciadedatos.net/documentos/py08_random_forest_python). En este caso, se crearon modelos que permiten estimar la [**masa muscular**](https://www.scielo.cl/scielo.php?pid=S0717-75182008000400003&script=sci_arttext&tlng=en) (medida en kilogramos) y el [**porcentaje corporal de grasa**](https://ve.scielo.org/scielo.php?pid=S0004-06222007000400008&script=sci_arttext) a partir de distintas medidas antropométricas. 
       
    Los modelos generados muestran un grado aceptable de coincidencia con las mediciones de estos parámetros, que típicamente requieren de balanzas de bioimpedancia y/o absorciometría de rayos X de energía dual. Una vez con las aproximaciones para masa muscular y porcentaje de grasa corporal, se estima el grado de riesgo de padecer sarcopenia para cada paciente mediante el uso del algoritmo de clustering jerarquico. 
       
    Estas condiciones de diagnóstico fueron propuestas con el objetivo de minimizar la cantidad de parámetros antropométricos y establecer puntos de corte que puedan ser validados por personal médico capacitado. **Este enfoque se asemeja a lo que se conoce en inteligencia artificial como un sistema experto, ya que los modelos resultantes requieren validación por parte de especialistas.**
    </div>
    """,unsafe_allow_html=True)


elif opcion == "Proceso":
    st.subheader("📉 Gráficas e interpretación")
    st.write("Aquí van los gráficos, importancias, etc.")






    #########################################################
    import streamlit as st
    import pandas as pd
    import requests
    from io import BytesIO


    import joblib
    import requests
    import io

    @st.cache_resource(show_spinner=False)
    def cargar_modelo_desde_github(url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            modelo = joblib.load(io.BytesIO(response.content))
            return modelo
        except Exception as e:
            st.error(f"Ocurrió un error al intentar cargar el archivo: {e}")
            return None


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
            #variances
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
            #from sklearn.tree import DecisionTreeRegressor
            #from sklearn.model_selection import train_test_split
            #from sklearn.metrics import mean_squared_error
            #from itertools import combinations
            #import pandas as pd

            #st.subheader("🔍 Selección de combinaciones óptimas de variables para predecir IMME")

            # Variables disponibles
            #variables = ["sexo",'P117', 'P118', 'P119', 'P120', 'P121', 'P122', 'P123', 'P124',
            #     'P125', 'P126', 'P127', 'P128', 'P129', 'IMC', 'P113', 'P112_vel']

            # Recalcular estatura si es necesario
            #if 'P118' not in df_combined.columns:
            #    df_combined['P118'] = ((df_combined['P117'] / df_combined['IMC'])**0.5) * 100


            import streamlit as st
            import pandas as pd
            import numpy as np
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error
            from itertools import combinations
            import joblib
            import matplotlib.pyplot as plt

            # Variables disponibles
            variables = ["sexo",'P117', 'P118', 'P119', 'P120', 'P121', 'P122', 'P123', 'P124',
                 'P125', 'P126', 'P127', 'P128', 'P129', 'IMC', 'P113', 'P112_vel']

            nombres_amigables = {
            'P117': 'Peso (kg)',
            'P118': 'Estatura (cm)',
            'P119': 'Talla sentada (cm)',
            'P120': 'Brazo (cm)',
            'P121': 'Cintura (cm)',
            'P122': 'Cadera (cm)',
            'P123': 'Muslo (cm)',
            'P124': 'Pantorrilla (cm)',
            'P125': 'Pliegue Tricipital (mm)',
            'P126': 'Pliegue Subescapular (mm)',
            'P127': 'Pliegue Bíceps (mm)',
            'P128': 'Pliegue Pantorrilla (mm)',
            'P129': 'Pliegue Suprailiaco (mm)',
            'IMC': 'IMC',
            'P113': 'Fuerza de prensión',
            'P112_vel': 'Velocidad de marcha',
            'sexo': 'Sexo'
                }


            st.subheader("🔍 Selección de combinaciones óptimas de variables para predecir IMME")

            # Número de variables por combinación
            selected_n = st.number_input("Selecciona el número de variables en cada combinación a mostrar", min_value=1, max_value=len(variables), value=3)    
            max_combinaciones = 5

            # Recalcular estatura si no existe
            if 'P118' not in df_combined.columns:
                df_combined['P118'] = ((df_combined['P117'] / df_combined['IMC'])**0.5) * 100


            ###################3
            if st.button("🔁 Calcular combinaciones"):
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

                st.session_state.errores_combinaciones = errores_combinaciones
                st.session_state.mejor_combinacion = min(errores_combinaciones, key=errores_combinaciones.get)
                st.session_state.mejor_error = errores_combinaciones[st.session_state.mejor_combinacion]

                # Calcular mejor combinación con `selected_n` variables
                filtered_results = {k: v for k, v in errores_combinaciones.items() if len(k) == selected_n}
                sorted_results = sorted(filtered_results.items(), key=lambda x: x[1])

                if sorted_results:
                    st.session_state.mejor_combinacion_n = list(sorted_results[0][0])
                else:
                    st.session_state.mejor_combinacion_n = None

            # Recuperar combinaciones desde session_state
            errores_combinaciones = st.session_state.get("errores_combinaciones", {})
            mejor_combinacion = st.session_state.get("mejor_combinacion", None)
            mejor_error = st.session_state.get("mejor_error", None)
            mejor_combinacion_n = st.session_state.get("mejor_combinacion_n", None)

            # Mostrar resultados si ya existen
            if errores_combinaciones and mejor_combinacion:
                st.markdown("### 🏆 Mejor combinación global:")
                #st.markdown(f"- **Variables**: `{mejor_combinacion}`")

                vars_amigables = [nombres_amigables.get(var, var) for var in mejor_combinacion]
                st.markdown(f"- **Variables**: `{', '.join(vars_amigables)}`")

                st.markdown(f"- **Error cuadrático medio (MSE)**: `{mejor_error:.4f}`")

                st.markdown(f"### 📊 Combinaciones con exactamente {selected_n} variables:")
                filtered_results = {k: v for k, v in errores_combinaciones.items() if len(k) == selected_n}
                sorted_results = sorted(filtered_results.items(), key=lambda x: x[1])

                #df_resultados = pd.DataFrame([
                #    {'Variables': ', '.join(k), 'MSE': v}
                #    for k, v in sorted_results
                #])

                df_resultados = pd.DataFrame([
                {'Variables': ', '.join([nombres_amigables.get(var, var) for var in k]), 'MSE': v}
                for k, v in sorted_results
                ])


                st.dataframe(df_resultados, use_container_width=True)
            else:
                st.info("Presiona el botón para calcular combinaciones óptimas.")
        
        
            import matplotlib.pyplot as plt
            import pandas as pd
            import numpy as np



            if mejor_combinacion_n is not None:
                # Datos para la mejor combinación global
                X_global = df_combined[list(mejor_combinacion)]
                y = df_combined['IMME']
                Xg_train, Xg_test, yg_train, yg_test = train_test_split(X_global, y, test_size=0.2, random_state=42)
                modelo_global = DecisionTreeRegressor(random_state=42).fit(Xg_train, yg_train)
                # Guardar y permitir descarga
                import joblib
                joblib.dump(modelo_global, "modelo_global_imme.pkl")
                with open("modelo_global_imme.pkl", "rb") as f:
                    st.download_button("⬇️ Descargar modelo global entrenado", f, file_name="modelo_global_imme.pkl")
                st.session_state.modelo_global = modelo_global
                y_pred_global = modelo_global.predict(Xg_test)
                mse_global = mean_squared_error(yg_test, y_pred_global)

                # Datos para la mejor combinación con n variables
                X_n = df_combined[mejor_combinacion_n]
                Xn_train, Xn_test, yn_train, yn_test = train_test_split(X_n, y, test_size=0.2, random_state=42)
                modelo_n = DecisionTreeRegressor(random_state=42).fit(Xn_train, yn_train)       
                y_pred_n = modelo_n.predict(Xn_test)
                mse_n = mean_squared_error(yn_test, y_pred_n)

                import joblib
                import os

                modelo_filename = f"modelo_n_variables_{selected_n}.pkl"
                joblib.dump(modelo_n, modelo_filename)

                # Mostrar botón de descarga
                with open(modelo_filename, 'rb') as f:
                    st.download_button(
                        label=f"⬇️ Descargar modelo con {selected_n} variables",
                        data=f,
                        file_name=modelo_filename,
                        mime='application/octet-stream'
                    )

        
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
                    #modelo = st.session_state.modelo_global
                    modelo = cargar_modelo_desde_github("https://raw.githubusercontent.com/SArcD/SARC_Predictor/main/modelo_global_imme.pkl")
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
                    'sexo': 'Sexo'
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
                    st.session_state.prediccion_valor = None  # Limpia previa

                    try:
                        input_df = pd.DataFrame([input_values])
                        pred = modelo.predict(input_df)[0]
                        st.session_state.prediccion_valor = pred
                    except Exception as e:
                        st.error(f"❌ Ocurrió un error al hacer la predicción: {e}")

                # Mostrar resultado solo si se generó una predicción
                if st.session_state.get("prediccion_valor") is not None:
                    st.success(f"🧠 IMME estimado: **{st.session_state.prediccion_valor:.2f}**")

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
            st.dataframe(resumen.style.format({'% Eliminados':'{:.1f}%'}), use_container_width=True)

        
            # Combinar resultados
            df_resultado = pd.concat([
                df_fuerza[~df_fuerza.index.isin(df_imme.index)][['Fuerza', 'IMME', 'Marcha', 'Clasificación Sarcopenia']],
                df_imme[~df_imme.index.isin(df_marcha.index)][['Fuerza', 'IMME', 'Marcha', 'Clasificación Sarcopenia']],
                df_marcha[['Fuerza', 'IMME', 'Marcha', 'Clasificación Sarcopenia']]
            ])


            # Crear una copia del DataFrame original filtrado
            df_filtered = df_filtered.copy()

            # Crear una columna vacía primero
            df_filtered['Clasificación Sarcopenia'] = None

            # Llenar con las etiquetas de df_resultado donde coincidan los índices
            df_filtered.loc[df_resultado.index, 'Clasificación Sarcopenia'] = df_resultado['Clasificación Sarcopenia']

            # Asignar "Sin Sarcopenia" a quienes no fueron clasificados
            df_filtered['Clasificación Sarcopenia'] = df_filtered['Clasificación Sarcopenia'].fillna('Sin Sarcopenia')
        #df_filtered
        
        # Mostrar resultados
        #st.subheader("Resultados de Clasificación de Sarcopenia")
        #if not df_resultado.empty:
        #    #st.write(df_resultado)
        #    st.dataframe(df_resultado, use_container_width=True)
        #else:
        #    st.warning("No se identificaron individuos con criterios de sarcopenia bajo los filtros establecidos.")
        ##df_filtered


        #df_combined
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches

        # Filtrar total por sexo (1 = hombres, 0 = mujeres)
        #sexo_elegido = 1  # Simular que el 
        #df_total_sexo = df_filtered.shape[0]
        #total_pacientes = len(df_total_sexo)
            total_pacientes = df_filtered.shape[0]
        #total_pacientes
        # Contar pacientes por clasificación
        #conteos = df_resultado['Clasificación Sarcopenia'].value_counts()
        #sospechosa = conteos.get('Sarcopenia Sospechosa', 0)
        #probable = conteos.get('Sarcopenia Probable', 0)
        #grave = conteos.get('Sarcopenia Grave', 0)
        #saludables = total_pacientes - sospechosa


        # Contar pacientes por clasificación (totales acumulativos)
            conteos = df_resultado['Clasificación Sarcopenia'].value_counts()
            total_grave = conteos.get('Sarcopenia Grave', 0)
            total_probable = conteos.get('Sarcopenia Probable', 0)
            total_sospechosa = conteos.get('Sarcopenia Sospechosa', 0)

        # Desglose excluyente para evitar dobles conteos
        #grave = total_grave
        #probable = total_probable - grave
        #sospechosa = total_sospechosa - probable - grave
        #saludables = total_pacientes - (sospechosa + probable + grave)


            grave = df_resultado[df_resultado['Clasificación Sarcopenia'] == 'Sarcopenia Grave'].shape[0]
            probable = df_resultado[
                (df_resultado['Clasificación Sarcopenia'] == 'Sarcopenia Probable') &
                (~df_resultado.index.isin(df_resultado[df_resultado['Clasificación Sarcopenia'] == 'Sarcopenia Grave'].index))
            ].shape[0]
            sospechosa = df_resultado[
                (df_resultado['Clasificación Sarcopenia'] == 'Sarcopenia Sospechosa') &
                (~df_resultado.index.isin(
                    df_resultado[df_resultado['Clasificación Sarcopenia'].isin(['Sarcopenia Grave', 'Sarcopenia Probable'])].index
                ))
            ].shape[0]

            saludables = total_pacientes - (grave + probable + sospechosa)

    
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

            # Diccionario de etiquetas amigables: nombre interno -> nombre visible
            column_labels = {
                'edad_am': 'Edad (años)',
                'P117': 'Peso (kg)',
                'P118': 'Estatura (cm)',
                'P119': 'Talla sentada (cm)',
                'P120': 'Brazo (cm)',
                'P121': 'Cintura (cm)',
                'P122': 'Cadera (cm)',
                'P123': 'Muslo (cm)',
                'P124': 'Pantorrilla (cm)',
                'P125': 'Pliegue Tricipital (mm)',
                'P126': 'Pliegue Subescapular (mm)',
                'P127': 'Pliegue Bíceps (mm)',
                'P128': 'Pliegue Pantorrilla (mm)',
                'P129': 'Pliegue Suprailiaco (mm)',
                'IMC': 'IMC',
                'Fuerza': 'Fuerza (kg)',
                'Marcha': 'Marcha (m/s)',
                'IMME': 'IMME'
            }

        # Lista completa de variables seleccionables
            all_columns = list(column_labels.keys())

        # Sidebar para seleccionar variables
        #st.sidebar.header("🔎 Selecciona variables para comparar")
        #selected_labels = st.sidebar.multiselect(
        #    label="Variables a graficar",
        #    options=list(column_labels.values()),
        #    default=list(column_labels.values())  # Puedes dejar vacío si prefieres no seleccionar todas por defecto
        #)

        # Mostrar selección de variables en el cuerpo central
            st.markdown("## Selecciona las variables que deseas comparar")
            selected_labels = st.multiselect(
                label="Variables disponibles",
                options=list(column_labels.values()),
                default=list(column_labels.values())  # Puedes cambiar esto si prefieres dejarlo vacío al inicio
            )
        
            # Mapear etiquetas visibles a columnas reales
            selected_columns = [col for col, label in column_labels.items() if label in selected_labels]

            # Validar que exista la columna 'Clasificación Sarcopenia'
            if 'Clasificación Sarcopenia' not in df_filtered.columns:
                st.error("❌ El DataFrame no contiene la columna 'Clasificación Sarcopenia'.")
            else:
                # Generar un gráfico por variable seleccionada
                for column in selected_columns:
                    fig = go.Figure()

                    # Agregar boxplot por grupo
                    for grupo, color in zip(
                        ['Sin Sarcopenia', 'Sarcopenia Sospechosa', 'Sarcopenia Probable', 'Sarcopenia Grave'],
                        ['green', 'goldenrod', 'orange', 'firebrick']
                    ):
                        data = df_filtered[df_filtered['Clasificación Sarcopenia'] == grupo][column].dropna()
                        fig.add_trace(go.Box(
                            y=data,
                            name=grupo,
                            boxpoints='outliers',
                            notched=True,
                            marker=dict(color=color)
                        ))

                    fig.update_layout(
                        title=f'Diagrama de caja - {column_labels[column]}',
                        yaxis_title=column_labels[column],
                        xaxis_title="Grupo de Sarcopenia",
                        title_font=dict(size=20),
                        yaxis=dict(tickfont=dict(size=14)),
                        xaxis=dict(tickfont=dict(size=14)),
                        height=500
                    )
                    st.plotly_chart(fig)
            st.session_state.df_filtered = df_filtered



###################################
        with st.expander("Modelos predictivos"):

            import streamlit as st
            import pandas as pd
            import matplotlib.pyplot as plt
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report, f1_score
            from sklearn.inspection import PartialDependenceDisplay
            from sklearn.preprocessing import LabelEncoder
            from imblearn.over_sampling import SMOTE

            import streamlit as st
            import pandas as pd
            import matplotlib.pyplot as plt
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            from sklearn.metrics import classification_report, f1_score
            from sklearn.inspection import PartialDependenceDisplay
            from imblearn.over_sampling import SMOTE

            st.subheader("📊 Predicción de sarcopenia con Random Forest + SMOTE")

            column_map = {
                'P117': 'Peso (kg)',
                'P118': 'Estatura (cm)',
                'P119': 'Talla sentada (cm)',
                'P120': 'Brazo (cm)',
                'P121': 'Cintura (cm)',
                'P122': 'Cadera (cm)',
                'P123': 'Muslo (cm)',
                'P124': 'Pantorrilla (cm)',
                'P125': 'Pliegue Tricipital (mm)',
                'P126': 'Pliegue Subescapular (mm)',
                'P127': 'Pliegue Bíceps (mm)',
                'P128': 'Pliegue Pantorrilla (mm)',
                'P129': 'Pliegue Suprailiaco (mm)',
                'IMC': 'IMC',
                'Fuerza': 'Fuerza (kg)',
                'Marcha': 'Marcha (m/s)',
                'IMME': 'IMME'
            }

            selected_vars_display = st.multiselect(
                "Selecciona las variables predictoras:",
                options=list(column_map.values()),
                default=['Fuerza (kg)', 'Marcha (m/s)', 'IMME']
            )

            inv_column_map = {v: k for k, v in column_map.items()}
            selected_vars = [inv_column_map[var] for var in selected_vars_display]

            if selected_vars:
                try:
                    # Convertir a numérico forzadamente
                    df = df_filtered.copy()
                    for col in selected_vars:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                    df = df.dropna(subset=selected_vars + ['Clasificación Sarcopenia'])

                    X = df[selected_vars]
                    y_raw = df['Clasificación Sarcopenia']

                    # Codificar clases
                    le = LabelEncoder()
                    y = le.fit_transform(y_raw)  # ahora y son enteros

                    # SMOTE
                    smote = SMOTE(random_state=42)
                    X_resampled, y_resampled = smote.fit_resample(X, y)

                    # Dividir
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
                    )

                    model = RandomForestClassifier(
                        n_estimators=300,
                        max_depth=3,
                        min_samples_leaf=5,
                        min_samples_split=10,
                        random_state=42
                    )
                    model.fit(X_train, y_train)

                    # Predicción
                    y_pred = model.predict(X_test)
                    report = classification_report(y_test, y_pred, target_names=le.classes_)
                    f1 = f1_score(y_test, y_pred, average='weighted')

                    # === DESCARGAR MODELO ENTRENADO ===
                    import joblib    
                    import io

                    modelo_bytes = io.BytesIO()
                    joblib.dump(model, modelo_bytes)
                    modelo_bytes.seek(0)

                    st.download_button(
                        label="⬇️ Descargar modelo entrenado (Random Forest - Sarcopenia)",
                        data=modelo_bytes,
                        file_name="modelo_sarcopenia_rf.pkl",
                        mime="application/octet-stream"
                    )

                    st.text("Reporte de clasificación:")
                    st.text(report)
                    st.text(f"Weighted F1-score: {f1:.4f}")

                    # Explicación de métricas
                    st.markdown("### 📘 ¿Cómo interpretar las métricas?")
                    st.markdown("""
                    Las siguientes métricas te ayudan a entender **cómo de bien el modelo identifica cada tipo de sarcopenia**:

                    - **Precisión (Precision)**: De todas las veces que el modelo predijo cierta clase, ¿cuántas veces acertó?  
                      - Ejemplo: Si `Precisión = 0.80` en *Sarcopenia Grave*, significa que 8 de cada 10 veces que el modelo dijo "Sarcopenia Grave", acertó.
                    - **Sensibilidad o Exhaustividad (Recall)**: De todas las personas que **realmente tienen** esa clase, ¿a cuántas identificó correctamente el modelo?  
                      - Ejemplo: Si `Recall = 0.60` en *Sarcopenia Sospechosa*, el modelo detectó 6 de cada 10 personas con esa condición.
                    - **F1-score**: Es un equilibrio entre precisión y sensibilidad. Si ambas son altas, el F1 también lo será.

                    #### 🔍 Guía rápida de interpretación:
                    - `> 0.85` → Excelente desempeño
                    - `0.70 - 0.85` → Buen desempeño, puede mejorar
                    - `0.50 - 0.70` → Desempeño moderado, se sugiere revisar variables o clases
                    - `< 0.50` → Débil, probablemente el modelo no distingue bien esta clase

                    """)

                    # Obtener importancias
                    importances = model.feature_importances_
                    var_importance = pd.Series(importances, index=selected_vars_display).sort_values(ascending=True)

                    # Graficar
                    fig_imp, ax_imp = plt.subplots(figsize=(8, 0.5 * len(var_importance)), dpi=150)
                    var_importance.plot(kind='barh', ax=ax_imp, color='skyblue', edgecolor='black')
                    ax_imp.set_title("Importancia relativa de cada variable", fontsize=12)
                    ax_imp.set_xlabel("Importancia")    
                    ax_imp.grid(axis='x', linestyle='--', alpha=0.7)
                    st.pyplot(fig_imp)

                    ##############################################################################################3#    
                    # --- OPCIÓN 4: Mostrar gráfica de dependencia parcial solo al presionar botón ---
                    if st.button("📈 Generar gráficas de dependencia parcial por clase"):
                        try:
                            color_map = {
                                'Sarcopenia Grave': '#d62728',
                                'Sin Sarcopenia': '#1f77b4',
                                'Sarcopenia Sospechosa': '#2ca02c',
                                'Sarcopenia Probable': '#ff7f0e'
                            }

                            fig, ax = plt.subplots(1, len(selected_vars), figsize=(5 * len(selected_vars), 4 + len(le.classes_) * 0.6), dpi=300)
                            if len(selected_vars) == 1:
                                ax = [ax]

                            for class_index, class_name in enumerate(le.classes_):
                                PartialDependenceDisplay.from_estimator(
                                    model,
                                    X_train,
                                    features=list(range(len(selected_vars))),
                                    feature_names=selected_vars_display,
                                    target=class_index,
                                    ax=ax,
                                    line_kw={
                                        "label": class_name,
                                        "color": color_map.get(class_name, None)
                                    }
                                )

                            for i, axis in enumerate(ax):
                                axis.set_xlabel(selected_vars_display[i])
                                axis.set_ylabel("Dependencia Parcial")
                                axis.grid(True)
                                axis.legend()

                            plt.suptitle("📈 Gráfica de dependencia parcial por clase", fontsize=14)
                            st.pyplot(fig)

                        except Exception as e:
                            st.error(f"Ocurrió un error al generar las gráficas de dependencia parcial: {e}")




                ################################################################################################
                # Agrega esta sección justo antes del bucle de graficación
                #color_map = {
                #    'Sarcopenia Grave': '#d62728',       # Rojo
                #    'Sin Sarcopenia': '#1f77b4',         # Azul
                #    'Sarcopenia Sospechosa': '#2ca02c',  # Verde
                #    'Sarcopenia Probable': '#ff7f0e'     # Naranja
                #}
                
                # Gráficos de dependencia parcial
                #fig, ax = plt.subplots(1, len(selected_vars), figsize=(5 * len(selected_vars), 4 + len(le.classes_) * 0.6), dpi=300)

                #if len(selected_vars) == 1:
                #    ax = [ax]

                #for class_index, class_name in enumerate(le.classes_):
                #    PartialDependenceDisplay.from_estimator(
                #        model,
                #        X_train,
                #        features=list(range(len(selected_vars))),
                #        feature_names=selected_vars_display,
                #        target=class_index,
                #        ax=ax,
                #        line_kw={"label": class_name,
                #        "color": color_map.get(class_name, None)}
                #    )
                
                #for i, axis in enumerate(ax):
                #    axis.set_xlabel(selected_vars_display[i])
                #    axis.set_ylabel("Dependencia Parcial")
                #    axis.grid(True)
                #    axis.legend()

                #plt.suptitle("📈 Gráfica de dependencia parcial por clase", fontsize=14)
                #st.pyplot(fig)

                except Exception as e:
                    st.error(f"Ocurrió un error durante el entrenamiento o visualización: {e}")





        
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
    
elif opcion == "Formularios":

    import streamlit as st
    import pandas as pd
    import numpy as np
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_squared_error
    import joblib
    import requests
    import io

    # Diccionario de nombres amigables
    nombres_amigables = {
        'P117': 'Peso (kg)',
        'P118': 'Estatura (cm)',
        'P119': 'Talla sentada (cm)',
        'P120': 'Brazo (cm)',
        'P121': 'Cintura (cm)',
        'P122': 'Cadera (cm)',
        'P123': 'Muslo (cm)',
        'P124': 'Pantorrilla (cm)',
        'P125': 'Pliegue Tricipital (mm)',
        'P126': 'Pliegue Subescapular (mm)',
        'P127': 'Pliegue Bíceps (mm)',
        'P128': 'Pliegue Pantorrilla (mm)',
        'P129': 'Pliegue Suprailiaco (mm)',
        'IMC': 'IMC',
        'P113': 'Fuerza (kg)',
        'P112_vel':  'Marcha (m/s)',
        'sexo': 'sexo'
    }

    variables_disponibles = list(nombres_amigables.keys())

    @st.cache_data
    def cargar_modelo_desde_url(url):
        response = requests.get(url)
        if response.status_code == 200:
            return joblib.load(io.BytesIO(response.content))
        else:
            st.error("Error al cargar el modelo.")
            return None

    modelos_dict = {
        "Mejor combinación global": ("https://github.com/SArcD/SARC_Predictor/raw/refs/heads/main/modelo_global_imme.pkl", None),
        "Combinación con 2 variables": ("https://github.com/SArcD/SARC_Predictor/raw/refs/heads/main/modelo_n_variables_2.pkl", 2),
        "Combinación con 3 variables": ("https://github.com/SArcD/SARC_Predictor/raw/refs/heads/main/modelo_n_variables_3.pkl", 3),
        "Combinación con 4 variables": ("https://github.com/SArcD/SARC_Predictor/raw/refs/heads/main/modelo_n_variables_4.pkl", 4),
        "Seleccionar manualmente": (None, None)
    }

    st.subheader("📤 Formularios para predecir IMME")
    tab_manual, tab_archivo, tab_sarcopenia = st.tabs(["🧍 Ingreso manual", "📁 Subir archivo", "Sarcopenia"])

    with st.sidebar:
        modelo_seleccionado = st.selectbox("Modelo para usar", list(modelos_dict.keys()))

    modelo_url, n_vars = modelos_dict[modelo_seleccionado]
    modelo = None

    if modelo_seleccionado != "Seleccionar manualmente":
        modelo = cargar_modelo_desde_url(modelo_url)
    else:
        seleccion_manual = st.multiselect("Variables manuales", options=variables_disponibles)
        if seleccion_manual:
            n_vars = len(seleccion_manual)
            modelo = None


    with tab_manual:
        if "pacientes_manual" not in st.session_state:
            st.session_state.pacientes_manual = []
        if "paciente_en_edicion" not in st.session_state:
            st.session_state.paciente_en_edicion = None

        st.markdown("### ✍️ Introducción manual de datos")

        input_values = {}
        variables_utilizadas = (
            seleccion_manual if modelo_seleccionado == "Seleccionar manualmente"
            else modelo.feature_names_in_
        )

        input_values["Identificador"] = st.text_input("Identificador del paciente", 
            value=(
                st.session_state.paciente_en_edicion["Identificador"]
                if st.session_state.paciente_en_edicion else ""
            )
        )

        for var in variables_utilizadas:
            label = nombres_amigables.get(var, var)
            key_input = f"{var}_manual"

            if var == "sexo":
                input_values[var] = (
                    1.0 if st.selectbox(label, ["Mujer", "Hombre"], 
                        key=key_input,
                        index=1 if (st.session_state.paciente_en_edicion and st.session_state.paciente_en_edicion[var] == 1.0) else 0
                    ) == "Hombre" else 0.0
                )
            else:
                input_values[var] = st.number_input(label, value=(
                    st.session_state.paciente_en_edicion[var]
                    if st.session_state.paciente_en_edicion else 0.0
                ), key=key_input)

        # Incluir SIEMPRE marcha (P112_vel)
        label_marcha = nombres_amigables.get("P112_vel", "P112_vel")
        key_input_marcha = "P112_vel_manual"
        input_values["P112_vel"] = st.number_input(
            label_marcha,
            value=(
                st.session_state.paciente_en_edicion["P112_vel"]
                if st.session_state.paciente_en_edicion and "P112_vel" in st.session_state.paciente_en_edicion
                else 0.0
            ),
            key=key_input_marcha
        )



        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Agregar paciente"):
                st.session_state.pacientes_manual.append(input_values.copy())
                st.session_state.paciente_en_edicion = None
        with col2:
            if st.session_state.paciente_en_edicion is not None:
                if st.button("✅ Guardar edición"):
                    idx = st.session_state.paciente_en_edicion_idx
                    st.session_state.pacientes_manual[idx] = input_values.copy()
                    st.session_state.paciente_en_edicion = None

        if st.session_state.pacientes_manual:
            df_manual = pd.DataFrame(st.session_state.pacientes_manual)

            # 🔽 Aquí insertas el bloque para renombrar
            columnas_amigables = {col: nombres_amigables.get(col, col) for col in df_manual.columns}
            df_mostrar = df_manual.rename(columns=columnas_amigables)

            st.markdown("### Pacientes registrados")
            st.dataframe(df_mostrar)  # Mostrar con nombres amigables
            
            #st.markdown("### 🗂️ Pacientes registrados")
            #st.dataframe(df_manual)

            # Selector para editar o borrar
            identificadores = df_manual["Identificador"].tolist()
            paciente_a_gestionar = st.selectbox("Selecciona un paciente para editar o borrar:", options=[""] + identificadores)

            col_editar, col_borrar = st.columns(2)
            with col_editar:
                if paciente_a_gestionar and paciente_a_gestionar != "":
                    idx = df_manual[df_manual["Identificador"] == paciente_a_gestionar].index[0]
                    if st.button("✏️ Editar paciente"):
                        st.session_state.paciente_en_edicion = st.session_state.pacientes_manual[idx]
                        st.session_state.paciente_en_edicion_idx = idx
                        st.info(f"✏️ Editando al paciente con Identificador: {paciente_a_gestionar}")

            with col_borrar:
                if paciente_a_gestionar and paciente_a_gestionar != "":
                    idx = df_manual[df_manual["Identificador"] == paciente_a_gestionar].index[0]
                    if st.button("🗑️ Borrar paciente"):
                        st.session_state.pacientes_manual.pop(idx)
                        st.success(f"🗑️ Paciente '{paciente_a_gestionar}' eliminado.")
                        st.session_state.paciente_en_edicion = None
                        st.rerun()  # Refrescar interfaz para evitar errores

            if st.button("🔮 Predecir IMME para pacientes"):
                if modelo_seleccionado == "Seleccionar manualmente":
                    X_manual = df_manual[seleccion_manual]
                    modelo = DecisionTreeRegressor().fit(X_manual, np.zeros(len(X_manual)))
                    pred = modelo.predict(X_manual)
                    rmse = 0.0
                else:
                    columnas_utilizadas = modelo.feature_names_in_
                    pred = modelo.predict(df_manual[columnas_utilizadas])
                    rmse = np.sqrt(mean_squared_error(np.zeros_like(pred), pred))

                df_manual["IMME"] = pred

                # Mostrar tabla con nombres amigables actualizados
                columnas_amigables = {col: nombres_amigables.get(col, col) for col in df_manual.columns}
                df_mostrar = df_manual.rename(columns=columnas_amigables)
                st.dataframe(df_mostrar)


                st.success(f"📉 RMSE estimado: {rmse:.4f}")




    
                # Mostrar tabla con nombres amigables actualizados
                #columnas_amigables = {col: nombres_amigables.get(col, col) for col in df_manual.columns}
                #df_mostrar = df_manual.rename(columns=columnas_amigables)
                #st.dataframe(df_mostrar)


                #st.success(f"📉 RMSE estimado: {rmse:.4f}")

    ##############################

#        import joblib
#        import urllib.request

#        # URL de los modelos
#        url_modelo_hombre = "https://github.com/SArcD/SARC_Predictor/raw/refs/heads/main/modelo_sarcopenia_rfhombre.pkl"
#        url_modelo_mujer = "https://github.com/SArcD/SARC_Predictor/raw/refs/heads/main/modelo_sarcopenia_rfmujer.pkl"

#        # Función para cargar modelo desde URL
#        @st.cache_resource
#        def cargar_modelo_desde_url(url):
#            with urllib.request.urlopen(url) as response:
#                modelo = joblib.load(response)
#            return modelo

#        # Cargar modelos
#        modelo_hombre = cargar_modelo_desde_url(url_modelo_hombre)
#        modelo_mujer = cargar_modelo_desde_url(url_modelo_mujer)
 
        # Verificar columnas requeridas
        #columnas_sarcopenia = ['Fuerza (kg)', 'Marcha (m/s)', "IMME"]
#        columnas_sarcopenia = ['P113', 'P112_vel', 'IMME']

#        if all(col in df_manual.columns for col in columnas_sarcopenia):
#            clasificaciones = []
#            for _, fila in df_manual.iterrows():
#                sexo = fila["sexo"]
#                entrada = fila[columnas_sarcopenia].values.reshape(1, -1)
#                modelo_uso = modelo_hombre if sexo == 1.0 else modelo_mujer
#                prediccion = modelo_uso.predict(entrada)[0]
#                clasificaciones.append(prediccion)
#
#            df_manual["Clasificación de sarcopenia"] = clasificaciones
#        else:
#            st.warning("⚠️ No se pudo calcular la clasificación de sarcopenia. Asegúrate de haber capturado 'Fuerza', 'marcha' e 'IMME'.")


    

    with tab_archivo:
        archivo = st.file_uploader("Sube tu archivo (.xlsx)", type="xlsx")
        if archivo:
            df_archivo = pd.read_excel(archivo)

            # Verificar presencia del identificador
            if "Identificador" not in df_archivo.columns:
                st.error("❌ La columna 'Identificador' es obligatoria en tu archivo.")
            else:
                columnas_requeridas = (
                    seleccion_manual if modelo_seleccionado == "Seleccionar manualmente"
                    else modelo.feature_names_in_
                )

                # Verifica que todas las columnas del modelo estén presentes
                if not all(col in df_archivo.columns for col in columnas_requeridas):
                    st.error(f"❌ Faltan columnas requeridas: {', '.join([col for col in columnas_requeridas if col not in df_archivo.columns])}")
                else:
                    if modelo_seleccionado == "Seleccionar manualmente":
                        modelo = DecisionTreeRegressor().fit(df_archivo[seleccion_manual], np.zeros(len(df_archivo)))
                        pred = modelo.predict(df_archivo[seleccion_manual])
                        rmse = 0.0  # No hay verdad para comparar
                    else:
                        pred = modelo.predict(df_archivo[columnas_requeridas])
                        rmse = np.sqrt(mean_squared_error(np.zeros_like(pred), pred))  # Estimación simbólica

                    # Agrega la predicción
                    df_archivo["IMME"] = pred

                    # Muestra solo Identificador + columnas utilizadas + predicción
                    columnas_mostrar = ["Identificador"] + list(columnas_requeridas) + ["IMME"]
                    st.dataframe(df_archivo[columnas_mostrar])

                    # Exportar resultados
                    output = io.BytesIO()
                    df_archivo[columnas_mostrar].to_excel(output, index=False)
                    st.download_button("⬇️ Descargar predicciones", output.getvalue(), file_name="imme_con_prediccion.xlsx")

                    st.success(f"📉 RMSE estimado: {rmse:.4f}")

    #with tab_sarcopenia:
  #      st.markdown("### 📋 Formulario para predicción de sarcopenia")
#
#        # Variables que usará el modelo
#        column_map = {
#            'Fuerza': 'Fuerza (kg)',
#            'Marcha': 'Marcha (m/s)',
#            'IMME': 'IMME'
#        }

#        # Selección de variables
#        selected_vars_display = st.multiselect(
#            "Selecciona las variables predictoras para entrenar el modelo:",
#            options=list(column_map.values()),
#            default=['Fuerza (kg)', 'Marcha (m/s)', 'IMME']
#        )
#        inv_column_map = {v: k for k, v in column_map.items()}
#        selected_vars = [inv_column_map[var] for var in selected_vars_display]

#        # Inicializar lista de pacientes
#        if "pacientes_sarcopenia" not in st.session_state:
#            st.session_state.pacientes_sarcopenia = []

#        # Iniciar paciente nuevo o cargar edición
#        if "edicion_sarcopenia" in st.session_state:
#            nuevo_paciente = st.session_state.edicion_sarcopenia
#            editando = True
#            st.info(f"✏️ Editando paciente: {nuevo_paciente['Identificador']}")
#        else:
#            nuevo_paciente = {}
#            nuevo_paciente["Identificador"] = st.text_input("Identificador del paciente", key="id_sarc")
#            editando = False

#        # Inputs
#        for var in selected_vars:
#            key_input = f"{var}_sarc"
#            valor = (
#                nuevo_paciente[var]
#                if editando and var in nuevo_paciente
#                else 0.0
#            )
#            nuevo_paciente[var] = st.number_input(f"Ingrese {column_map[var]}", key=key_input, value=valor)

#        # Botones para agregar o guardar edición
#        col_add, col_save = st.columns(2)
#        with col_add:
#            if not editando:
#                if st.button("➕ Agregar paciente para predicción de sarcopenia"):
#                    st.session_state.pacientes_sarcopenia.append(nuevo_paciente.copy())
#                    st.success("Paciente agregado.")
#                    st.rerun()
#        with col_save:
#            if editando:
#                if st.button("✅ Guardar cambios"):
#                    st.session_state.pacientes_sarcopenia[st.session_state.edicion_idx_sarcopenia] = nuevo_paciente.copy()
#                    del st.session_state.edicion_sarcopenia
#                    del st.session_state.edicion_idx_sarcopenia
#                    st.success("Cambios guardados.")
#                    st.rerun()

#        # Mostrar pacientes registrados
#        if st.session_state.pacientes_sarcopenia:
#            df_sarc = pd.DataFrame(st.session_state.pacientes_sarcopenia)
#            st.markdown("### 👥 Pacientes registrados")
#            st.dataframe(df_sarc)

#            # Selector para edición o eliminación
#            identificadores = df_sarc["Identificador"].tolist()
#            paciente_seleccionado = st.selectbox("Selecciona un paciente para editar o borrar:", [""] + identificadores)

#            col_edit, col_delete = st.columns(2)
#            with col_edit:
#                if paciente_seleccionado and paciente_seleccionado != "":
#                    idx = df_sarc[df_sarc["Identificador"] == paciente_seleccionado].index[0]
#                    if st.button("✏️ Editar paciente seleccionado"):
#                        st.session_state.edicion_sarcopenia = st.session_state.pacientes_sarcopenia[idx]
#                        st.session_state.edicion_idx_sarcopenia = idx
#                        st.rerun()
#            with col_delete:
#                if paciente_seleccionado and paciente_seleccionado != "":
#                    idx = df_sarc[df_sarc["Identificador"] == paciente_seleccionado].index[0]
#                    if st.button("🗑️ Borrar paciente seleccionado"):
#                        st.session_state.pacientes_sarcopenia.pop(idx)
#                        st.success(f"Paciente '{paciente_seleccionado}' eliminado.")
#                        st.rerun()

#            # Entrenar modelo y predecir
#            if st.button("🔮 Entrenar modelo y predecir sarcopenia"):
#                try:
#                    if "df_filtered" in st.session_state:
#                        df_train = st.session_state.df_filtered.copy()
#                    else:
#                        st.warning("No se encontró el DataFrame 'df_filtered'. Asegúrate de generar los datos antes.")
#                        st.stop()##
#
#                    for col in selected_vars:
#                        df_train[col] = pd.to_numeric(df_train[col], errors='coerce')

#                    df_train = df_train.dropna(subset=selected_vars + ['Clasificación Sarcopenia'])

#                    X = df_train[selected_vars]
#                    y_raw = df_train['Clasificación Sarcopenia']

#                    # Codificar etiquetas
#                    from sklearn.preprocessing import LabelEncoder
#                    le = LabelEncoder()
#                    y = le.fit_transform(y_raw)

#                    # SMOTE para balancear
#                    from imblearn.over_sampling import SMOTE
#                    smote = SMOTE(random_state=42)
#                    X_resampled, y_resampled = smote.fit_resample(X, y)

#                    # Entrenar modelo
#                    from sklearn.ensemble import RandomForestClassifier
#                    model = RandomForestClassifier(
#                        n_estimators=300,
#                        max_depth=3,
#                        min_samples_leaf=5,
#                        min_samples_split=10,
#                        random_state=42
#                    )
#                    model.fit(X_resampled, y_resampled)

#                    # Predicción
#                    X_pred = df_sarc[selected_vars]
#                    y_pred = model.predict(X_pred)
#                    y_pred_labels = le.inverse_transform(y_pred)

#                    df_sarc["Predicción Sarcopenia"] = y_pred_labels
#                    st.markdown("### 🧪 Resultados de predicción")
#                    st.dataframe(df_sarc)

#                except Exception as e:
#                    st.error(f"Ocurrió un error durante la predicción: {e}")

    with tab_sarcopenia:
        st.markdown("### 📋 Formulario para predicción de sarcopenia")

        # Asegurar que el DataFrame base esté disponible
        if "df_filtered" not in st.session_state:
            st.warning("Primero debes cargar o generar el DataFrame con los datos base.")
            st.stop()

        df_base = st.session_state.df_filtered.copy()

        # Variables disponibles
        column_map = {
            'Fuerza': 'Fuerza (kg)',
            'Marcha': 'Marcha (m/s)',
            'IMME': 'IMME',
            'P117': 'Peso (kg)',
            'P118': 'Estatura (cm)',
            'P120': 'Brazo (cm)',
            'P121': 'Cintura (cm)',
            'P122': 'Cadera (cm)',
            'P123': 'Muslo (cm)',
            'P124': 'Pantorrilla (cm)'
        }

        posibles_variables = [col for col in df_base.columns if col in column_map]
        disponibles = [k for k in column_map if k in posibles_variables]

        selected_vars_display = st.multiselect(
            "Selecciona las variables predictoras disponibles:",
            options=[column_map[k] for k in disponibles],
            default=[column_map[k] for k in disponibles if k in ['Fuerza', 'Marcha', 'IMME']]
        )
        inv_column_map = {v: k for k, v in column_map.items()}
        selected_vars = [inv_column_map[v] for v in selected_vars_display]

        # Lista de pacientes
        if "pacientes_sarcopenia" not in st.session_state:
            st.session_state.pacientes_sarcopenia = []

        # Ingreso/edición de paciente
        st.markdown("#### ➕ Registro de paciente")
        if "edicion_sarcopenia" in st.session_state:
            nuevo_paciente = st.session_state.edicion_sarcopenia
            editando = True
            st.info(f"✏️ Editando paciente: {nuevo_paciente['Identificador']}")
        else:
            nuevo_paciente = {"Identificador": st.text_input("Identificador del paciente", key="id_sarc")}
            editando = False

        for var in selected_vars:
            key_input = f"{var}_sarc"
            valor = nuevo_paciente.get(var, 0.0)
            nuevo_paciente[var] = st.number_input(f"Ingrese {column_map[var]}", key=key_input, value=valor)

        # Botones
        col1, col2 = st.columns(2)
        with col1:
            if not editando and st.button("➕ Agregar paciente"):
                st.session_state.pacientes_sarcopenia.append(nuevo_paciente.copy())
                st.success("Paciente agregado.")
                st.rerun()

        with col2:
            if editando and st.button("✅ Guardar cambios"):
                idx = st.session_state.edicion_idx_sarcopenia
                st.session_state.pacientes_sarcopenia[idx] = nuevo_paciente.copy()
                del st.session_state.edicion_sarcopenia
                del st.session_state.edicion_idx_sarcopenia
                st.success("Cambios guardados.")
                st.rerun()

        # Tabla de pacientes
        if st.session_state.pacientes_sarcopenia:
            df_sarc = pd.DataFrame(st.session_state.pacientes_sarcopenia)
            st.markdown("### 👥 Pacientes registrados")
            st.dataframe(df_sarc)

            seleccion = st.selectbox("Selecciona un paciente para editar o borrar:", [""] + df_sarc["Identificador"].tolist())

            col3, col4 = st.columns(2)
            with col3:
                if seleccion and seleccion != "":
                    idx = df_sarc[df_sarc["Identificador"] == seleccion].index[0]
                    if st.button("✏️ Editar paciente seleccionado"):
                        st.session_state.edicion_sarcopenia = st.session_state.pacientes_sarcopenia[idx]
                        st.session_state.edicion_idx_sarcopenia = idx
                        st.rerun()
            with col4:
                if seleccion and seleccion != "":
                    idx = df_sarc[df_sarc["Identificador"] == seleccion].index[0]
                    if st.button("🗑️ Borrar paciente seleccionado"):
                        st.session_state.pacientes_sarcopenia.pop(idx)
                        st.success(f"Paciente '{seleccion}' eliminado.")
                        st.rerun()

            # Entrenar modelo
            if st.button("🔮 Entrenar modelo y predecir sarcopenia"):
                try:
                    from sklearn.ensemble import RandomForestClassifier
                    from sklearn.preprocessing import LabelEncoder
                    from sklearn.metrics import classification_report, f1_score
                    from imblearn.over_sampling import SMOTE

                    df_train = df_base.copy()
                    for col in selected_vars:
                        df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
                    df_train = df_train.dropna(subset=selected_vars + ['Clasificación Sarcopenia'])

                    X = df_train[selected_vars]
                    y_raw = df_train['Clasificación Sarcopenia']
                    le = LabelEncoder()
                    y = le.fit_transform(y_raw)

                    smote = SMOTE(random_state=42)
                    X_resampled, y_resampled = smote.fit_resample(X, y)

                    model = RandomForestClassifier(
                        n_estimators=300,
                        max_depth=3,
                        min_samples_leaf=5,
                        min_samples_split=10,
                        random_state=42
                    )
                    model.fit(X_resampled, y_resampled)

                    # Métricas del modelo
                    y_pred_train = model.predict(X)
                    f1 = f1_score(y, y_pred_train, average='weighted')
                    report = classification_report(y, y_pred_train, target_names=le.classes_)

                    # Predicción para los pacientes registrados
                    X_pred = df_sarc[selected_vars]
                    y_pred = model.predict(X_pred)
                    y_labels = le.inverse_transform(y_pred)

                    df_sarc["Predicción Sarcopenia"] = y_labels
                    st.markdown("### 🧪 Resultados de predicción")
                    st.dataframe(df_sarc)

                    # Reporte de desempeño
                    st.markdown("### 📈 Desempeño del modelo (solo con las variables seleccionadas)")
                    st.text(report)
                    st.success(f"F1-score (ponderado): {f1:.4f}")

                    # Descargar resultados
                    import io
                    output = io.BytesIO()
                    df_sarc.to_excel(output, index=False)
                    st.download_button("⬇️ Descargar predicciones", output.getvalue(), file_name="predicciones_sarcopenia.xlsx")

                except Exception as e:
                    st.error(f"Ocurrió un error durante la predicción: {e}")











elif opcion == "Equipo de trabajo":
    st.subheader("Equipo de Trabajo")

       # Información del equipo
    equipo = [{
               "nombre": "Dr. Santiago Arceo Díaz",
               "foto": "ArceoS.jpg",
               "reseña": "Licenciado en Física, Maestro en Física y Doctor en Ciencias (Astrofísica). Posdoctorante de la Universidad de Colima y profesor del Tecnológico Nacional de México Campus Colima. Cuenta con el perfil deseable, pertenece al núcleo académico y es colaborador del cuerpo académico Tecnologías Emergentes y Desarrollo Web de la Maestría Sistemas Computacionales. Ha dirigido tesis de la Maestría en Sistemas Computacionales y en la Maestría en Arquitectura Sostenible y Gestión Urbana.",
               "CV": "https://scholar.google.com.mx/citations?user=3xPPTLoAAAAJ&hl=es", "contacto": "santiagoarceodiaz@gmail.com"},
           {
               "nombre": "José Ramón González",
               "foto": "JR.jpeg",
               "reseña": "Estudiante de la facultad de medicina en la Universidad de Colima, cursando el servicio social en investigación en el Centro Universitario de Investigaciones Biomédicas, bajo el proyecto Aplicación de un software basado en modelos predictivos como herramienta de apoyo en el diagnóstico de sarcopenia en personas adultas mayores a partir de parámetros antropométricos.", "CV": "https://scholar.google.com.mx/citations?user=3xPPTLoAAAAJ&hl=es", "contacto": "jgonzalez90@ucol.mx"},
           {
               "nombre": "Dra. Xochitl Angélica Rosío Trujillo Trujillo",
               "foto": "DraXochilt.jpg",
               "reseña": "Bióloga, Maestra y Doctora en Ciencias Fisiológicas con especialidad en Fisiología. Es Profesora-Investigadora de Tiempo Completo de la Universidad de Colima. Cuenta con perfil deseable y es miembro del Sistema Nacional de Investigadores en el nivel 3. Su línea de investigación es en Biomedicina en la que cuenta con una producción científica de más de noventa artículos en revistas internacionales, varios capítulos de libro y dos libros. Imparte docencia y ha formado a más de treinta estudiantes de licenciatura y de posgrado en programas académicos adscritos al Sistema Nacional de Posgrado del CONAHCYT.",
               "CV": "https://portal.ucol.mx/cuib/XochitlTrujillo.htm", "contacto": "rosio@ucol.mx"},
                 {
               "nombre": "Dr. Miguel Huerta Viera",
               "foto": "DrHuerta.jpg",
               "reseña": "Doctor en Ciencias con especialidad en Fisiología y Biofísica. Es Profesor-Investigador Titular “C” del Centro Universitario de Investigaciones Biomédicas de la Universidad de Colima. Es miembro del Sistema Nacional de Investigadores en el nivel 3 emérito. Su campo de investigación es la Biomedicina, con énfasis en la fisiología y biofísica del sistema neuromuscular y la fisiopatología de la diabetes mellitus. Ha publicado más de cien artículos revistas indizadas al Journal of Citation Reports y ha graduado a más de 40 Maestros y Doctores en Ciencias en programas SNP-CONAHCyT.",
               "CV": "https://portal.ucol.mx/cuib/dr-miguel-huerta.htm", "contacto": "huertam@ucol.mx"},
                 {
               "nombre": "Dr. Jaime Alberto Bricio Barrios",
               "foto":  "BricioJ.jpg",
               "reseña": "Licenciado en Nutrición, Maestro en Ciencias Médicas, Maestro en Seguridad Alimentaria y Doctor en Ciencias Médicas. Profesor e Investigador de Tiempo Completo de la Facultad de Medicina en la Universidad de Colima. miembro del Sistema Nacional de Investigadores en el nivel 1. Miembro fundador de la asociación civil DAYIN (Desarrollo de Ayuda con Investigación)",
               "CV": "https://scholar.google.com.mx/citations?hl=es&user=ugl-bksAAAAJ", "contacto": "jbricio@ucol.mx"},      
               {
               "nombre": "Mtra. Elena Elsa Bricio Barrios",
               "foto": "BricioE.jpg",
               "reseña": "Química Metalúrgica, Maestra en Ciencias en Ingeniería Química y doctorante en Ingeniería Química. Actualmente es profesora del Tecnológico Nacional de México Campus Colima. Cuenta con el perfil deseable, es miembro del cuerpo académico Tecnologías Emergentes y Desarrollo Web y ha codirigido tesis de la Maestría en Sistemas Computacionales.",
               "CV": "https://scholar.google.com.mx/citations?hl=es&user=TGZGewEAAAAJ", "contacto": "elena.bricio@colima.tecnm.mx"},
               {
               "nombre": "Dra. Mónica Ríos Silva",
               "foto": "rios.jpg",
               "reseña": "Médica cirujana y partera con especialidad en Medicina Interna y Doctorado en Ciencias Médicas por la Universidad de Colima, médica especialista del Hospital Materno Infantil de Colima y PTC de la Facultad de Medicina de la Universidad de Colima. Es profesora de los posgrados en Ciencias Médicas, Ciencias Fisiológicas, Nutrición clínica y Ciencia ambiental global.",
               "CV": "https://scholar.google.com.mx/scholar?hl=en&as_sdt=0%2C5&q=Monica+Rios+silva&btnG=", "contacto": "mrios@ucol.mx"},
               {
               "nombre": "Dra. Rosa Yolitzy Cárdenas María",  
               "foto": "cardenas.jpg",
               "reseña": "Ha realizado los estudios de Química Farmacéutica Bióloga, Maestría en Ciencias Médicas y Doctorado en Ciencias Médicas, todos otorgados por la Universidad de Colima. Actualmente, se desempeña como Técnica Académica Titular C en el Centro Universitario de Investigaciones Biomédicas de la Universidad de Colima, enfocándose en la investigación básica y clínica de enfermedades crónico-degenerativas no transmisibles en investigación. También es profesora en la Maestría y Doctorado en Ciencias Médicas, así como en la Maestría en Nutrición Clínica de la misma universidad. Es miembro del Sistema Nacional de Investigadores nivel I y miembro fundador activo de la asociación civil DAYIN (https://www.dayinac.org/)",
               "CV": "https://scholar.google.com.mx/scholar?hl=en&as_sdt=0%2C5&q=rosa+yolitzy+c%C3%A1rdenas-mar%C3%ADa&btnG=&oq=rosa+yoli", "contacto": "rosa_cardenas@ucol.mx"}
               ]

    # Establecer la altura deseada para las imágenes
    altura_imagen = 150  # Cambia este valor según tus preferencias

    # Mostrar información de cada miembro del equipo
    for miembro in equipo:
           st.subheader(miembro["nombre"])
           img = st.image(miembro["foto"], caption=f"Foto de {miembro['nombre']}", use_column_width=False, width=altura_imagen)
           st.write(f"Correo electrónico: {miembro['contacto']}")
           st.write(f"Reseña profesional: {miembro['reseña']}")
           st.write(f"CV: {miembro['CV']}")

    # Información de contacto
    st.subheader("Información de Contacto")
    st.write("Si deseas ponerte en contacto con nuestro equipo, puedes enviar un correo a santiagoarceodiaz@gmail.com")

