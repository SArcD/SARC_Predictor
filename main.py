import pandas as pd
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


############################


# Configura la barra lateral para seleccionar secciones
#seccion = st.sidebar.selectbox(
#    "Selecciona una sección",
#    ["Introducción", "Análisis de datos", "Visualización de resultados", "Conclusiones"]
#)

## Muestra el contenido basado en la selección
#if seccion == "Introducción":
#    st.title("Introducción")
#    st.write("Aquí va el contenido de la introducción...")
    
#elif seccion == "Análisis de datos":
#    st.title("Análisis de datos")
#    st.write("Aquí va el contenido sobre el análisis de datos...")
#    # Aquí puedes incluir código y gráficos específicos del análisis

#elif seccion == "Visualización de resultados":
#    st.title("Visualización de resultados")
#    st.write("Aquí va el contenido de visualización...")
#    # Agrega gráficos y resultados visualizados

#elif seccion == "Conclusiones":
#    st.title("Conclusiones")
#    st.write("Aquí van las conclusiones del análisis...")




############################

pestañas = st.sidebar.radio("Selecciona una pestaña:", ("Presentación", "Predicción de Sarcopenia", "Registro de datos",  "Equipo de trabajo"))
if pestañas == "Presentación":
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
                   
    Los datos utiliados para el entrenamiento de los modelos provienen de las ediciones de los años 2019 y 2022 del **"Cuadernillo de Obesidad, Sarcopenia y Fragilidad en Adultos Mayores Derechohabientes del Instituto Mexicano del Seguro Social de las Delegaciones Sur y Norte de la Ciudad de México"**. Con los datos recolectados, se programó un algoritmo que aplica clustering jerárquico aglomerativo para clasficar pacientes en conjuntos que se caracterizan su similitud en las medidas antropométricas. En el caso del Índice de Masa Muscular Esquelética Apendicular, se crearon modelos de ajuste que calculan esta variable a partir de las circunferencias de pantorrilla, brazo, 
                
                 septiembre y octubre el año 2023 en una muestra de adultos mayores que residen en la Zona Metropolitana, Colima, Villa de Álvarez, México, se procedió al desarrollo de modelos predictivos mediante el algoritmo [**Random Forest**](https://cienciadedatos.net/documentos/py08_random_forest_python). En este caso, se crearon modelos que permiten estimar la [**masa muscular**](https://www.scielo.cl/scielo.php?pid=S0717-75182008000400003&script=sci_arttext&tlng=en) (medida en kilogramos) y el [**porcentaje corporal de grasa**](https://ve.scielo.org/scielo.php?pid=S0004-06222007000400008&script=sci_arttext) a partir de distintas medidas antropométricas. 
       
    Los modelos generados muestran un grado aceptable de coincidencia con las mediciones de estos parámetros, que típicamente requieren de balanzas de bioimpedancia y/o absorciometría de rayos X de energía dual. Una vez con las aproximaciones para masa muscular y porcentaje de grasa corporal, se estima el grado de riesgo de padecer sarcopenia para cada paciente mediante el uso del algoritmo de clustering jerarquico. 
       
    Estas condiciones de diagnóstico fueron propuestas con el objetivo de minimizar la cantidad de parámetros antropométricos y establecer puntos de corte que puedan ser validados por personal médico capacitado. **Este enfoque se asemeja a lo que se conoce en inteligencia artificial como un sistema experto, ya que los modelos resultantes requieren validación por parte de especialistas.**
    </div>
    """,unsafe_allow_html=True)





#######
if pestañas == "Predicción de Sarcopenia":
    st.subheader("Carga de los datos")
    
    st.markdown("""
    En el siguiente menú puede elegir entre las bases de datos disponibles
    
    """)
    # Seleccionar el año de la base de datos
    selected_year = st.selectbox("Por favor, seleccione la base de datos:", ["2019", "2022"])

    # Definir la ruta del archivo en función de la selección
    if selected_year == "2022":
        file_path = "Base 2022 Santiago Arceo.xls"
    else:
        file_path = "Base 2019 Santiago Arceo.xls"

    # Intento de cargar el archivo de Excel usando `xlrd` para archivos `.xls`
    try:
        #df = pd.read_excel("archivo.xlsx", engine="openpyxl")  # Para archivos .xlsx

        datos = pd.read_excel(file_path)  # Rellenar NaN con espacios
        st.write(f"Datos de la base {selected_year} cargados con éxito:")
        st.dataframe(datos)

        with st.expander("**Diccionario de variables**"):
            # Diccionario de significados completo basado en el cuadernillo
            significados = {
            'folio_paciente': 'Folio asignado a cada paciente',
            'edad_am': 'Años cumplidos al momento de la entrevista',
            'sexo': 'Sexo del paciente (1: Masculino, 2: Femenino)',
            'nacio_en_mexico': 'Indicador de si nació en México (0: No, 1: Sí)',
            'P20': 'Realiza alguna actividad laboral (0: No, 1: Sí)',
            'P21': 'Ocupación actual del paciente',
            'P35_3': '¿Fuma cigarros actualmente? (0: No, 1: Sí)',
            'P35_4': 'Frecuencia de fumar (1: Diario, 2: No todos los días)',
            'P35_5_1': 'Cantidad de cigarros que fuma por día',
            'P36': '¿Actualmente toma bebidas alcohólicas? (0: No, 1: Sí, 2: Nunca ha tomado)',
            'P36_1': 'Frecuencia semanal de consumo de bebidas alcohólicas en los últimos tres meses',
            'P44_3': 'Diagnóstico de VIH (0: No, 1: Sí)',
            'P44_3_1': 'Años desde el diagnóstico de VIH',
            'P44_3_2': 'Recibió tratamiento para VIH (0: No, 1: Sí)',
            'P44_5': 'Diagnóstico de anemia por deficiencia (0: No, 1: Sí)',
            'P44_7': 'Diagnóstico de arritmia cardiaca (0: No, 1: Sí)',
            'P44_8': 'Diagnóstico de artritis reumatoide (0: No, 1: Sí)',
            'P44_9': 'Diagnóstico de cáncer metastásico (0: No, 1: Sí)',
            'P44_11': 'Diagnóstico de depresión (0: No, 1: Sí)',
            'P44_12': 'Diagnóstico de diabetes complicada (0: No, 1: Sí)',
            'P44_13': 'Diagnóstico de diabetes sin complicación (0: No, 1: Sí)',
            'P44_14': 'Diagnóstico de enfermedad cerebrovascular (0: No, 1: Sí)',
            'P44_20': 'Diagnóstico de hipertensión complicada (0: No, 1: Sí)',
            'P44_21': 'Diagnóstico de hipertensión sin complicación (0: No, 1: Sí)',
            'P44_24': 'Diagnóstico de insuficiencia renal (0: No, 1: Sí)',
            'P44_27': 'Diagnóstico de obesidad (0: No, 1: Sí)',
            'P44_31': 'Pérdida de peso (0: No, 1: Sí)',
            'P55_4_1': 'Capacidad para caminar en terreno plano (1: Sin dificultad, 5: No puede por otras causas)',
            'P55_4_2': 'Capacidad para levantarse de una silla sin apoyo de las manos (1: Sin dificultad, 5: No puede por otras causas)',
            'P55_4_3': 'Capacidad para sentarse y levantarse del sanitario (1: Sin dificultad, 5: No puede por otras causas)',
            'P55_4_4': 'Capacidad para sacar ropa del ropero y cajones (1: Sin dificultad, 5: No puede por otras causas)',
            'TOTAL_RUIS': 'Total de síntomas reportados en el índice de comorbilidad',
            'NIVELES_INCONTINENCIA': 'Nivel de incontinencia del paciente',
            'P57': '¿Ha sufrido alguna caída en el último año? (0: No, 1: Sí)',
            'P57_1': 'Número de caídas en el último año (1: 1 vez, 6: Más de 5 veces)',
            'Evaluacion_Global_30pto_Mini_nutritional': 'Evaluación global de estado nutricional (puntaje sobre 30)',
            'Katz_suma': 'Suma de la escala Katz para la limitación funcional',
            'Barthel_sum2': 'Puntuación total en la escala de Barthel',
            'P112_1': 'Evaluación del equilibrio en el SPPB (1: Excelente, 5: Deficiente)',
            'Observaciones_SPPB': 'Observaciones generales sobre el equilibrio en el SPPB',
            'P113': 'Fuerza de presión medida (kg)',
            'P113_1': 'Fuerza de presión medida en la mano dominante',
            'P117_1': 'Medida de la circunferencia de la pantorrilla - 1',
            'P117_2': 'Medida de la circunferencia de la pantorrilla - 2',
            'P117_3': 'Medida de la circunferencia de la pantorrilla - 3',
            'P118_1': 'Medida de la circunferencia de muslo - 1',
            'P118_2': 'Medida de la circunferencia de muslo - 2',
            'P118_3': 'Medida de la circunferencia de muslo - 3',
            'P119_1': 'Medida de la circunferencia de brazo - 1',
            'P119_2': 'Medida de la circunferencia de brazo - 2',
            'P119_3': 'Medida de la circunferencia de brazo - 3',
            'P120_1': 'Medida de la circunferencia de cadera - 1',
            'P120_2': 'Medida de la circunferencia de cadera - 2',
            'P120_3': 'Medida de la circunferencia de cadera - 3',
            'P121_1': 'Medida de la circunferencia de cintura - 1',
            'P121_2': 'Medida de la circunferencia de cintura - 2',
            'P121_3': 'Medida de la circunferencia de cintura - 3',
            'P122_1': 'Medida de la circunferencia de tórax - 1',
            'P122_2': 'Medida de la circunferencia de tórax - 2',
            'P122_3': 'Medida de la circunferencia de tórax - 3',
            'P123_1': 'Medida de la circunferencia de cuello - 1',
            'P123_2': 'Medida de la circunferencia de cuello - 2',
            'P123_3': 'Medida de la circunferencia de cuello - 3',
            'P124_1': 'Medida de la circunferencia de muñeca - 1',
            'P124_2': 'Medida de la circunferencia de muñeca - 2',
            'P124_3': 'Medida de la circunferencia de muñeca - 3',
            'INICIA_Circunferencias': 'Indicador de inicio de mediciones de circunferencias',
            'INICIA_Pliegues': 'Indicador de inicio de mediciones de pliegues cutáneos',
            'P125_1': 'Medida del pliegue cutáneo bicipital - 1',
            'P125_2': 'Medida del pliegue cutáneo bicipital - 2',
            'P125_3': 'Medida del pliegue cutáneo bicipital - 3',
            'P126_1': 'Medida del pliegue cutáneo tricipital - 1',
            'P126_2': 'Medida del pliegue cutáneo tricipital - 2',
            'P126_3': 'Medida del pliegue cutáneo tricipital - 3',
            'P127_1': 'Medida del pliegue subescapular - 1',
            'P127_2': 'Medida del pliegue subescapular - 2',
            'P127_3': 'Medida del pliegue subescapular - 3',
            'P128_1': 'Medida del pliegue de la pantorrilla - 1',
            'P128_2': 'Medida del pliegue de la pantorrilla - 2',
            'P128_3': 'Medida del pliegue de la pantorrilla - 3',
            'P129_1': 'Medida de la longitud talón-rodilla - 1',
            'P129_2': 'Medida de la longitud talón-rodilla - 2',
            'P129_3': 'Medida de la longitud talón-rodilla - 3',
            'P130_1': 'Medida de la longitud de brazada - 1',
            'P130_2': 'Medida de la longitud de brazada - 2',
            'P130_3': 'Medida de la longitud de brazada - 3',
            'obsevaciones': 'Observaciones adicionales'
            }

        # Mostrar el diccionario completo en un recuadro con barra de desplazamiento
       # with st.expander("Ver diccionario completo"):
       #     for col, desc in significados.items():
       #         st.write(f"**{col}**: {desc}")

        # Título del diccionario
        #title = "Diccionario de variables"
        #st.write(f"## {title}")


            # Crear el menú desplegable
            variable_seleccionada = st.selectbox("Seleccione una variable para ver su significado:", list(significados.keys()))

            # Mostrar la descripción de la variable seleccionada
            st.write(f"**{variable_seleccionada}:** {significados[variable_seleccionada]}")



        
            # Caja de entrada para búsqueda
            #st.write("### Buscador de variables")
            buscar = st.text_input("Escribe una o más variables separadas por comas, las variables se mostrarán junto a su significado en la tabla de abajo")

            # Estado persistente del DataFrame de resultados
            if 'resultados' not in st.session_state:
                st.session_state['resultados'] = pd.DataFrame(columns=['Variable', 'Significado'])

            # Procesar y mostrar resultados de la búsqueda
            if buscar:
                # Separar las variables ingresadas por comas y quitar espacios
                variables = [var.strip() for var in buscar.split(',')]
    
                # Agregar cada variable buscada y su significado al DataFrame de resultados
                nuevos_resultados = []
                for var in variables:
                    significado = significados.get(var, "Variable no encontrada")
                    nuevos_resultados.append({'Variable': var, 'Significado': significado})
    
                # Actualizar el DataFrame en el estado persistente
                st.session_state['resultados'] = pd.concat([st.session_state['resultados'], pd.DataFrame(nuevos_resultados)], ignore_index=True).drop_duplicates()

        # Mostrar el diccionario completo en un recuadro con barra de desplazamiento
        #with st.expander("Ver diccionario completo"):
        #for col, desc in significados.items():
        #    st.write(f"**{col}**: {desc}")
    
        
            # Mostrar el DataFrame de resultados acumulados
            st.write("Resultados de búsqueda")
            st.dataframe(st.session_state['resultados'], use_container_width=True)


        #######################################################################################################################
        with st.expander("Descripción de los datos"):
            st.write("Distribución de pacientes por sexo en la edición de 2019")
            st.write("El siguiente grafico muestra la distribución de participantes de acuerdo al sexo.")
            # Reemplazar valores numéricos en la columna 'sexo' con etiquetas 'Hombre' y 'Mujer'
            datos['sexo'] = datos['sexo'].replace({1.0: 'Hombre', 2.0: 'Mujer'})

            # Generar gráfico de pastel para la columna 'sexo'
            sexo_counts = datos['sexo'].value_counts()

            # Crear la figura para el gráfico de pastel
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(sexo_counts, labels=sexo_counts.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightpink'])
            ax.set_title('Proporción de Hombres vs Mujeres')
            ax.axis('equal')  # Asegura que el gráfico sea un círculo

            # Mostrar el gráfico de pastel en Streamlit
            st.pyplot(fig)

            # Menú desplegable para seleccionar la visualización del gráfico de comorbilidades
            seleccion = st.selectbox("Selecciona la muestra para visualizar el gráfico de comorbilidades", 
                             options=["Muestra completa", "Hombres", "Mujeres"])

            # Filtrar el DataFrame según la selección del usuario
            if seleccion == "Hombres":
                datos_filtrados = datos[datos['sexo'] == 'Hombre']
            elif seleccion == "Mujeres":
                datos_filtrados = datos[datos['sexo'] == 'Mujer']
            else:
                datos_filtrados = datos

            # Total de pacientes en la muestra seleccionada
            total_pacientes = len(datos_filtrados)

        ###################import pandas as pd
            columns_to_check = ['P44_3', 'P44_5', 'P44_7', 'P44_8', 'P44_9', 'P44_11', 'P44_12',
                            'P44_13', 'P44_14', 'P44_20', 'P44_21', 'P44_24', 'P44_27', 'P44_31']
            comorbidities_labels = ['VIH', 'Anemia', 'Arritmia', 'Artritis Reumatoide', 'Cáncer', 'Depresión', 'Diabetes Complicada',
                                'Diabetes Leve', 'Enfermedad Cerebro Vascular', 'Hipertensión Complicada', 'Hipertensión Sin Complicación',
                                'Insuficiencia Renal', 'Obesidad', 'Pérdida de Peso']

            # Crear un diccionario de mapeo entre las columnas y las etiquetas
            comorbidity_mapping = dict(zip(columns_to_check, comorbidities_labels))

            # Validar que las columnas existan en el DataFrame
            columns_to_check = [col for col in columns_to_check if col in datos_filtrados.columns]

            # Convertir las columnas de comorbilidades a valores numéricos (0 o 1) y llenar valores faltantes con 0
            datos_filtrados[columns_to_check] = datos_filtrados[columns_to_check].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

            # Barra 1: Pacientes con y sin comorbilidades
            datos_filtrados['Sin Comorbilidades'] = (datos_filtrados[columns_to_check].sum(axis=1) == 0).astype(int)
            pacientes_sin_comorbilidades = datos_filtrados['Sin Comorbilidades'].sum()
            pacientes_con_comorbilidades = len(datos_filtrados) - pacientes_sin_comorbilidades
            total_barra1 = len(datos_filtrados)

            # Normalizar las proporciones para la barra 1
            sin_comorbilidades_percent = pacientes_sin_comorbilidades / total_barra1
            con_comorbilidades_percent = pacientes_con_comorbilidades / total_barra1

            # Barra 2 y 3: Comorbilidades mayores y menores al 1%
            comorbidities_counts = datos_filtrados[columns_to_check].sum()
            comorbidities_percentages = (comorbidities_counts / len(datos_filtrados)) * 100

            major_comorbidities = comorbidities_percentages[comorbidities_percentages > 1].sort_values(ascending=False)
            minor_comorbidities = comorbidities_percentages[comorbidities_percentages <= 1].sort_values(ascending=False)

            # Normalizar las alturas de las barras (a nivel interno, para mostrar apiladas)
            major_normalized = major_comorbidities / major_comorbidities.sum()
            minor_normalized = minor_comorbidities / minor_comorbidities.sum()

            # Pacientes en cada barra (barra 2 y 3)
            total_major = datos_filtrados[major_comorbidities.index].sum(axis=1).astype(bool).sum()
            total_minor = datos_filtrados[minor_comorbidities.index].sum(axis=1).astype(bool).sum()

            # Etiquetas y valores normalizados
            major_labels = [comorbidity_mapping.get(col, col) for col in major_comorbidities.index]
            minor_labels = [comorbidity_mapping.get(col, col) for col in minor_comorbidities.index]

            # Crear gráfico de barras apiladas
            fig, ax = plt.subplots(figsize=(12, 8))
            x = np.arange(3)  # Tres barras: Pacientes, Mayores al 1%, Menores al 1%
            width = 0.6  # Ancho de las barras

            # Barra 1: Pacientes con y sin comorbilidades (Normalizado)
            ax.bar(x[0], sin_comorbilidades_percent, width, label='Sin Comorbilidades', color='lightblue')
            ax.text(x[0], sin_comorbilidades_percent / 2, f'{sin_comorbilidades_percent * 100:.1f}%', ha='center', va='center', fontsize=10)
            ax.bar(x[0], con_comorbilidades_percent, width, bottom=sin_comorbilidades_percent, label='Con Comorbilidades', color='steelblue')
            ax.text(x[0], sin_comorbilidades_percent + con_comorbilidades_percent / 2, f'{con_comorbilidades_percent * 100:.1f}%', ha='center', va='center', fontsize=10)
            ax.text(x[0], 1.05, f'Total: {total_barra1}', ha='center', va='bottom', fontsize=10, fontweight='bold')

            # Barra 2: Comorbilidades mayores al 1% (Ordenado y Relativo al Total)
            bottom_major = 0
            for label, value, actual_percent in zip(major_labels, major_normalized.values, major_comorbidities.values):
                ax.bar(x[1], value, width, bottom=bottom_major, label=label)
                ax.text(x[1], bottom_major + value / 2, f'{actual_percent:.1f}%', ha='center', va='center', fontsize=10)
                bottom_major += value
            ax.text(x[1], 1.05, f'Total: {total_major}', ha='center', va='bottom', fontsize=10, fontweight='bold')

            # Barra 3: Comorbilidades menores al 1% (Ordenado y Relativo al Total)
            bottom_minor = 0
            for label, value, actual_percent in zip(minor_labels, minor_normalized.values, minor_comorbidities.values):
                ax.bar(x[2], value, width, bottom=bottom_minor, label=label)
                ax.text(x[2], bottom_minor + value / 2, f'{actual_percent:.1f}%', ha='center', va='center', fontsize=10)
                bottom_minor += value
            ax.text(x[2], 1.05, f'Total: {total_minor}', ha='center', va='bottom', fontsize=10, fontweight='bold')

            # Configurar las etiquetas del gráfico
            ax.set_xticks(x)
            ax.set_xticklabels(["Pacientes", "Comorbilidades > 1%", "Comorbilidades <= 1%"])
            ax.set_ylabel('Altura Normalizada')
            ax.set_title('Distribución Normalizada de Pacientes y Comorbilidades', pad=20)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # Mostrar gráfico en Streamlit
            st.pyplot(fig)


###############3

        # Mostrar el total de pacientes para cada gráfico
        #st.write(f"Total de pacientes en la muestra seleccionada ({seleccion}): {total_pacientes}")
        #st.write(f"Total de pacientes con al menos una comorbilidad menor al 1%: {pacientes_con_menor_comorbilidad}")

    ##############################################################################

            df=datos[['P117_1','P117_2','P117_3','P118_1','P118_2','P118_3','P119_1','P119_2','P119_3','P120_1','P120_2','P120_3','P121_1','P121_2','P121_3','P122_1','P122_2','P122_3','P123_1','P123_2','P123_3','P124_1','P124_2','P124_3','P125_1','P125_2','P125_3','P126_1','P126_2','P126_3','P127_1','P127_2','P127_3','P128_1','P128_2','P128_3','P129_1','P129_2','P129_3','P130_1','P130_2','P130_3']]

            # Corrigiendo la advertencia al agrupar columnas
            df_grouped = df.T.groupby(lambda x: x.split('_')[0]).mean().T
            # Calculando el IMC: Peso / (Altura^2)
            df_grouped['IMC'] = df_grouped['P117'] / ((df_grouped['P118']*0.01) ** 2)

            # Promediar los valores de las columnas P113_1, P113_3, y P113_5
            df_2=datos[['P113_1', 'P113_3', 'P113_5']]
            df_2['P113_iz'] = datos[['P113_1', 'P113_3', 'P113_5']].mean(axis=1)
            # Promediar los valores de las columnas P113_1, P113_3, y P113_5
            df_2 = df_2.drop(columns=['P113_1', 'P113_3', 'P113_5'])

            # Promediar los valores de las columnas P113_1, P113_3, y P113_5
            df_3=datos[['P113_2', 'P113_4', 'P113_6']]
            df_3['P113_der'] = datos[['P113_2', 'P113_4', 'P113_6']].mean(axis=1)
            # Promediar los valores de las columnas P113_1, P113_3, y P113_5
            df_3 = df_3.drop(columns=['P113_2', 'P113_4', 'P113_6'])

            df_3b = pd.concat([df_2,df_3], axis=1)
            df_3b['P113']=(df_2['P113_iz']+df_3['P113_der'])/2
            df_3b = df_3b.drop(columns=['P113_iz', 'P113_der'])

            # Seleccionar las columnas y eliminar los valores que sean 0 antes de calcular el promedio
            df_4 = datos[['P112_4_1', 'P112_4_2']].replace(0, np.nan).dropna()
            # Calcular el promedio
            df_4['P112'] = df_4.mean(axis=1)
            # Verificar los valores únicos en P112 para asegurarse de que no sean todos iguales
            unique_values = df_4['P112'].unique()

            df_4['P112_vel'] = 4 / df_4['P112']
            df_4 = df_4.drop(columns=['P112_4_1', 'P112_4_2', 'P112'])

            df_datos=datos[['folio_paciente','edad_am','sexo','nacio_en_mexico', 'P44_3', 'P44_5', 'P44_7', 'P44_8', 'P44_9', 'P44_11', 'P44_12', 'P44_13', 'P44_14', 'P44_20', 'P44_21', 'P44_24', 'P44_27', 'P44_31']]

            # Concatenating df_grouped with df_r to create a single DataFrame
            df_combined = pd.concat([df_datos, df_grouped, df_3b, df_4], axis=1)
            #df_combined = st.dataframe(df_combined, use_container_width=True)
            df_combined.describe()


            # Filtrar las filas con NaN
            df_filtered = df_combined.dropna()

            # Mostrar un resumen del DataFrame resultante
            df_summary = df_filtered.describe()
            df_summary = st.dataframe(df_summary, use_container_width=True)


        ####################$$$$$$$$$$$$$$$$$$$$$########################
        with st.expander("Simplificar variables"):
            # Definir las columnas de comorbilidades y etiquetas
            columns_to_check = [
            'P44_3', 'P44_5', 'P44_7', 'P44_8', 'P44_9', 'P44_11', 'P44_12',
            'P44_13', 'P44_14', 'P44_20', 'P44_21', 'P44_24', 'P44_27', 'P44_31'
            ]
            comorbidities_labels = [
            'VIH', 'Anemia', 'Arritmia', 'Artritis Reumatoide', 'Cáncer', 'Depresión',
            'Diabetes Complicada', 'Diabetes Leve', 'Enfermedad Cerebro Vascular',
            'Hipertensión Complicada', 'Hipertensión Sin Complicación',
            'Insuficiencia Renal', 'Obesidad', 'Pérdida de Peso'
            ]

            # Crear un diccionario para enlazar columnas con etiquetas
            comorbidities_dict = dict(zip(columns_to_check, comorbidities_labels))

            # Crear menú desplegable para seleccionar comorbilidades
            selected_comorbidities = st.multiselect(
            "Selecciona cuáles comorbilidades considerar dentro de la muestra:",
            options=["Ninguna"] + comorbidities_labels,
            default="Ninguna"
            )

            # Filtrar el DataFrame
            if "Ninguna" in selected_comorbidities:
                df_combined_sc = df_combined[(df_combined[columns_to_check] == 0).all(axis=1)]
            else:
                # Obtener las columnas seleccionadas de acuerdo con las etiquetas
                selected_columns = [col for col, label in comorbidities_dict.items() if label in selected_comorbidities]
                df_combined_sc = df_combined[(df_combined[selected_columns] == 1).any(axis=1)]

        
            ########################$$$$$$$$$$#################################3
    
            # Mostrar el DataFrame resultante
            df_combined_sc

            df_combined=df_combined_sc[['folio_paciente', 'edad_am', 'sexo', 'nacio_en_mexico', 'P117',
            'P118', 'P119', 'P120', 'P121', 'P122', 'P123', 'P124', 'P125', 'P126',
            'P127', 'P128', 'P129', 'P130', 'IMC', 'P113', 'P112_vel']]
    

            # Medicion de varianza
            df_combined_2 = df_combined[df_combined['sexo'] == "Hombre"]

        #####
            # Suponiendo que 'df_combined' ya está definido

            # Crear un menú desplegable para seleccionar el filtro
            opcion = st.selectbox("Selecciona el género a mostrar", ["Muestra completa", "Hombre", "Mujer"])

            # Filtrar el DataFrame según la selección
            if opcion == "Hombre":
                df_combined_2 = df_combined[df_combined['sexo'] == "Hombre"]
            elif opcion == "Mujer":
                df_combined_2 = df_combined[df_combined['sexo'] == "Mujer"]
            else:
                df_combined_2 = df_combined  # Muestra completa sin filtrar
            ###

            # Caja de input para definir el rango de edad
            edad_min = st.number_input("Edad mínima", min_value=0, max_value=120, value=60, step=1)
            edad_max = st.number_input("Edad máxima", min_value=0, max_value=120, value=80, step=1)

            # Filtrar por rango de edad
            df_combined_2 = df_combined_2[(df_combined_2['edad_am'] >= edad_min) & (df_combined_2['edad_am'] <= edad_max)]

            # Mostrar el DataFrame filtrado final
            st.write("Filas filtradas del DataFrame según comorbilidades, rango de edad y género seleccionados:")
            st.dataframe(df_combined_2)

        ###

            # Mostrar el DataFrame filtrado
            #st.write("### DataFrame Filtrado")
            #st.dataframe(df_combined_2, use_container_width=True)

        #####
            columns_to_standardize = df_combined_2.columns[4:]  # Selecting columns from the 4th column onwards

            # Diccionario para renombrar las columnas
            column_names = {
            'P112_vel': 'Marcha',
            'P113': 'Fuerza',
            'P125': 'P. Tricipital',
            'P128': 'P. Pantorrilla',
            'P127': 'P. Biceps',
            'P126': 'P. Subescapular',
            'IMC': 'IMC',
            'P121': 'Cintura',
            'P123': 'Muslo',
            'P120': 'Brazo',
            'P124': 'Pantorrilla',
            'P117': 'Peso'
            }

            # Calculating variance using the provided method: dividing by the mean and then calculating the variance
            features = df_combined_2[list(column_names.keys())].rename(columns=column_names)

            variances = (features / features.mean()).dropna().var()

            variances=variances.sort_values(ascending=False)
            # Graficar las varianzas como gráfico de barras
            fig, ax = plt.subplots(figsize=(10, 6))
            variances.plot(kind='bar', ax=ax)
            ax.set_title("Varianzas de las Características Estandarizadas")
            ax.set_xlabel("Características")
            ax.set_ylabel("Varianza")
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Mostrar el gráfico en Streamlit
            st.pyplot(fig)


        ################# Redes de correlación  ################################

            from sklearn.feature_selection import VarianceThreshold

            # Standardizing the columns from the 4th column onwards in df_combined
            columns_to_standardize = df_combined_2.columns[4:]  # Selecting columns from the 4th column onwards

            # Calculating variance using the provided method: dividing by the mean and then calculating the variance
            features = df_combined_2[columns_to_standardize]  # Selecting the features to be standardized
            variances = (features / features.mean()).dropna().var()

            # Sorting variances in descending order
            variances = variances.sort_values(ascending=False)

            # Applying the variance threshold mask
            sel = VarianceThreshold(threshold=0.005)
            sel.fit(features / features.mean())

            # Creating a boolean mask based on the variance
            mask = variances >= 0.005

            # Applying the mask to create a reduced DataFrame
            df_reduced_2 = features.loc[:, mask]
            df_reduced_2.to_excel('reduced_df_2.xlsx', index=False)

            import networkx as nx

            # Selección de variables y cálculo de la matriz de correlación
            selected_vars = ['P112_vel', 'P113', 'P117', 'P120', 'P121', 'P122', 'P123', 'P124', 'P125', 'P128', 'P127', 'P126', 'IMC']
            correlation_matrix = df_reduced_2[selected_vars].corr()

            # Crear el grafo basado en la matriz de correlación
            G = nx.Graph()

            # Agregar nodos y aristas al grafo en función de un umbral de correlación
            threshold = 0.3  # Umbral para mostrar correlaciones
            for i, var1 in enumerate(selected_vars):
                for j, var2 in enumerate(selected_vars):
                    if i < j:  # Evitar duplicar aristas
                        corr_value = correlation_matrix.iloc[i, j]
                        if abs(corr_value) > threshold:
                            G.add_edge(var1, var2, weight=corr_value)

            # Dibujar el grafo de correlación
            fig, ax = plt.subplots(figsize=(10, 8))
            pos = nx.spring_layout(G, seed=42)
            edges = G.edges(data=True)

            # Dibujar nodos
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=700, node_color="lightblue")

            # Dibujar aristas con ancho variable basado en la fuerza de correlación
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                edgelist=[(u, v) for u, v, w in edges if w["weight"] > 0],
                width=[w["weight"] * 5 for u, v, w in edges if w["weight"] > 0],
                edge_color="blue", alpha=0.7)
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                edgelist=[(u, v) for u, v, w in edges if w["weight"] < 0],
                width=[-w["weight"] * 5 for u, v, w in edges if w["weight"] < 0],
                edge_color="red", alpha=0.7
            )

            # Dibujar etiquetas de nodos
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_family="sans-serif")

            # Configurar título
            ax.set_title("Red de Correlación entre Variables Seleccionadas")

            # Mostrar el gráfico en Streamlit
            st.pyplot(fig)

        ######### mapa de correlación ### #

            import seaborn as sns
            from sklearn.preprocessing import MinMaxScaler


            # Seleccionar las columnas para normalización
            selected_columns = ['P112_vel', 'P113', 'P125', 'P128', 'P127', 'P126', 'IMC', 'P121', 'P123', 'P120', 'P124']
            numeric_df = df_reduced_2[selected_columns]

            # Normalizar los datos utilizando Min-Max
            scaler = MinMaxScaler()
            normalized_df = scaler.fit_transform(numeric_df)

            # Crear un DataFrame con los datos normalizados
            normalized_df = pd.DataFrame(normalized_df, columns=selected_columns)

            # Calcular la matriz de correlación
            corr = normalized_df.corr(method='pearson')

            # Crear una máscara triangular superior para el mapa de calor
            mask = np.triu(np.ones_like(corr, dtype=bool))

            # Configurar y mostrar el gráfico de mapa de calor en Streamlit
            fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
            cmap = sns.diverging_palette(h_neg=10, h_pos=240, as_cmap=True)
            sns.heatmap(corr, mask=mask, center=0, cmap=cmap, linewidths=1, annot=True, fmt='.2f', square=True, ax=ax)

            # Mostrar el gráfico en Streamlit
            st.pyplot(fig)

        ######################################################
        with st.expander("**Clasificación de participantes usando clustering**"):
            df_combined_2['sexo'] = df_combined_2['sexo'].replace({'Hombre': 1.0, 'Mujer': 0.0})

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
            df_combined_2['IMME'] = df_combined_2.apply(calcular_IMME, axis=1)
            df_combined_2.describe()


    #################################### Clustering EWGSOP2 ##################################3

            import pandas as pd
            import numpy as np
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            from sklearn.cluster import AgglomerativeClustering
            from scipy.spatial.distance import pdist, squareform
            from sklearn.metrics import silhouette_score
            import matplotlib.pyplot as plt

            # Suponiendo que df_combined_2 ya está definido
            selected_columns = ['P113']
            numeric_data_2 = df_combined_2[selected_columns].dropna()

            # Verificar el número de muestras
            n_samples = len(numeric_data_2)

            if n_samples < 2:
                st.warning("Número insuficiente de muestras para realizar clustering. Se necesita al menos 2 muestras.")
            else:
                # Normalizar los datos
                scaler = StandardScaler()
                normalized_data_2 = scaler.fit_transform(numeric_data_2)

                # Aplicar PCA para reducir la dimensionalidad
                pca = PCA(n_components=1)  # Ajustar según sea necesario
                pca_data = pca.fit_transform(normalized_data_2)

                # Calcular la matriz de distancias
                distance_matrix = squareform(pdist(pca_data))

                # Ajustar el rango de K según el número de muestras
                max_clusters = min(15, n_samples)  # Máximo es 15 o el número de muestras
                avg_distances = []
                silhouettes = []
                K = range(2, max_clusters)

                for k in K:
                    clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
                    labels = clustering.fit_predict(pca_data)

                    # Calcular la distancia intra-cluster
                    intra_cluster_distances = []
                    for cluster in range(k):
                        cluster_points = distance_matrix[np.ix_(labels == cluster, labels == cluster)]
                        intra_cluster_distances.append(np.mean(cluster_points))

                    avg_distances.append(np.mean(intra_cluster_distances))

                    # Calcular el Silhouette Score
                    silhouette_avg = silhouette_score(pca_data, labels)
                    silhouettes.append(silhouette_avg)

                # Gráfica del método del codo
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                ax1.plot(K, avg_distances, 'bo-')
                ax1.set_xlabel('Número de clusters (k)')
                ax1.set_ylabel('Distancia intra-cluster promedio')
                ax1.set_title('Método del codo para Agglomerative Clustering')
                st.pyplot(fig1)

                # Gráfica del Silhouette Score
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                ax2.plot(K, silhouettes, 'go-')
                ax2.set_xlabel('Número de clusters (k)')
                ax2.set_ylabel('Silhouette Score')
                ax2.set_title('Silhouette Score para Agglomerative Clustering')
                st.pyplot(fig2)

            # Asegurarse de que el DataFrame final no contenga nulos
            df_combined_2 = df_combined_2.dropna()

            #########################################

            import pandas as pd
            import numpy as np
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import AgglomerativeClustering

            # Suponiendo que df_combined_2 ya está filtrado según los criterios anteriores
            selected_columns = ['P113']
            df_combined_copy=df_combined_2.copy()
            # Filtrar el DataFrame para incluir solo las columnas seleccionadas
            numeric_data_2 = df_combined_2[selected_columns]

            # Eliminar valores no numéricos y valores faltantes
            numeric_data_2 = numeric_data_2.dropna()

            # Verificar si hay suficientes muestras para el número deseado de clusters
            n_samples = len(numeric_data_2)
            n_clusters = min(4, n_samples)  # Ajustar el número de clusters a un máximo del número de muestras

            # Normalizar los datos
            scaler = StandardScaler()
            normalized_data_2 = scaler.fit_transform(numeric_data_2)

            # Aplicar Agglomerative Clustering si hay más de un cluster posible
            if n_clusters > 1:
                clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
                labels_2019 = clustering.fit_predict(normalized_data_2)
            else:
                labels_2019 = np.zeros(n_samples)  # Asignar todos a un solo cluster si no hay suficientes muestras

            # Agregar las etiquetas al DataFrame original filtrado
            df_combined_2['Cluster'] = labels_2019

            df_combined_2  # Mostrar las primeras filas del DataFrame resultante


        ##################### Graficos de caja ###########################
            import plotly.graph_objects as go


            # Renombrar las columnas
            df_combined_2 = df_combined_2.rename(columns={
                'P112_vel': 'Marcha',
                'P113': 'Fuerza',
                'P125': 'P. Tricipital',
                'P128': 'P. Pantorrilla',
                'IMC': 'IMC',
                'P127': 'Biceps',
                'P126': 'P. subescapular',
                'P121': 'Cintura',
                'P123': 'Muslo',
                'P120': 'Brazo',
                'P122': 'Cadera',
                'P124': 'Pantorrilla',
                'P117': 'Peso'
            })

            # Seleccionar las columnas específicas
            selected_columns_renamed = [
                'Marcha', 'Fuerza', 'P. Tricipital', 'P. Pantorrilla',
                'IMC', 'Biceps', 'P. subescapular', 'Cintura', 'Muslo', 'Brazo', 'Cadera', 'Pantorrilla', 'Peso', 'IMME'
            ]
            numeric_columns = df_combined_2[selected_columns_renamed + ['Cluster']]

            # Crear un gráfico de caja para cada parámetro y comparar los clusters
            for column in selected_columns_renamed:
                # Obtener los datos de cada cluster para el parámetro actual
                cluster_data = [df_combined_2[df_combined_2['Cluster'] == cluster][column] for cluster in df_combined_2['Cluster'].unique()]

                # Calcular los quintiles
                quintile_1 = df_combined_2[column].quantile(0.20)
                quintile_2 = df_combined_2[column].quantile(0.40)
                quintile_3 = df_combined_2[column].quantile(0.60)
                quintile_4 = df_combined_2[column].quantile(0.80)

                # Crear la figura para el gráfico de caja
                fig = go.Figure()

                # Agregar el gráfico de caja para cada cluster
                for j, cluster_series in enumerate(cluster_data):
                    fig.add_trace(go.Box(y=cluster_series, boxpoints='all', notched=True, name=f'Cluster {j}'))

                # Agregar líneas horizontales para los quintiles
                fig.add_shape(type="line",
                          x0=0, x1=1, y0=quintile_1, y1=quintile_1,
                          line=dict(color="blue", width=2, dash="dash"),
                            xref="paper", yref="y")
                fig.add_shape(type="line",
                          x0=0, x1=1, y0=quintile_2, y1=quintile_2,
                          line=dict(color="green", width=2, dash="dash"),
                          xref="paper", yref="y")
                fig.add_shape(type="line",
                          x0=0, x1=1, y0=quintile_3, y1=quintile_3,
                          line=dict(color="orange", width=2, dash="dash"),
                          xref="paper", yref="y")
                fig.add_shape(type="line",
                          x0=0, x1=1, y0=quintile_4, y1=quintile_4,
                          line=dict(color="red", width=2, dash="dash"),
                          xref="paper", yref="y")

                # Actualizar el diseño
                fig.update_layout(title_text=f'Comparación de Clusters - {column}',
                              xaxis_title="Clusters",
                              yaxis_title=column,
                              showlegend=False)

                # Mostrar el gráfico en Streamlit
                st.plotly_chart(fig)
        with st.expander("**Filtrar variables**"):
            ####### eliminar fuerza alta ###

            # Calcular el percentil 40 global de 'Fuerza'
            percentile_40_fuerza = df_combined_2['Fuerza'].quantile(0.40)

            # Crear un DataFrame vacío para almacenar las filas que cumplen la condición
            df_filtered = pd.DataFrame()
            percentages_deleted = {}

            for cluster in df_combined_2['Cluster'].unique():
                # Filtrar el DataFrame por cada cluster
                cluster_data = df_combined_2[df_combined_2['Cluster'] == cluster]
                # Mantener solo las filas con 'Fuerza' menor o igual al percentil 40 global
                filtered_cluster_data = cluster_data[cluster_data['Fuerza'] <= percentile_40_fuerza]
                # Agregar las filas filtradas al nuevo DataFrame
                df_filtered = pd.concat([df_filtered, filtered_cluster_data])
                # Calcular el porcentaje de filas eliminadas en cada cluster
                percentage_deleted = 100 * (1 - len(filtered_cluster_data) / len(cluster_data))
                percentages_deleted[cluster] = percentage_deleted

            # Convertir los porcentajes a un DataFrame para visualizar
            percentages_df = pd.DataFrame(list(percentages_deleted.items()), columns=['Cluster', 'Percentage Deleted'])

            # Mostrar el DataFrame con los datos filtrados
            st.write("### DataFrame Filtrado por Cluster con 'Fuerza' <= Percentil 40")
            st.dataframe(df_filtered)

            # Crear el diagrama de barras para mostrar el porcentaje de filas eliminadas por cluster
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(percentages_df['Cluster'], percentages_df['Percentage Deleted'], color='purple', alpha=0.7)
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Porcentaje de Filas Eliminadas')
            ax.set_title('Porcentaje de Filas Eliminadas por Cluster (Fuerza > Percentil 40% Global)')

            # Mostrar el gráfico de barras en Streamlit
            st.pyplot(fig)

        ################### IMME #######################################

            import pandas as pd
            import numpy as np
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            from sklearn.cluster import AgglomerativeClustering
            from scipy.spatial.distance import pdist, squareform
            from sklearn.metrics import silhouette_score
            import matplotlib.pyplot as plt

            # Suponiendo que df_filtered ya está definido
            selected_columns = ['IMME']
            numeric_data_2 = df_filtered[selected_columns].dropna()

            # Verificar el número de muestras
            n_samples = len(numeric_data_2)

            if n_samples < 2:
                st.warning("Número insuficiente de muestras para realizar clustering. Se necesita al menos 2 muestras.")
            else:
                # Normalizar los datos
                scaler = StandardScaler()
                normalized_data_2 = scaler.fit_transform(numeric_data_2)

                # Aplicar PCA para reducir la dimensionalidad
                pca = PCA(n_components=1)  # Ajusta el número de componentes si es necesario
                pca_data = pca.fit_transform(normalized_data_2)

                # Calcular la matriz de distancias
                distance_matrix = squareform(pdist(pca_data))

                # Ajustar el rango de K según el número de muestras
                max_clusters = min(15, n_samples)  # Máximo es 15 o el número de muestras
                avg_distances = []
                silhouettes = []
                K = range(2, max_clusters)

                for k in K:
                    clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
                    labels = clustering.fit_predict(pca_data)

                    # Calcular la distancia intra-cluster
                    intra_cluster_distances = []
                    for cluster in range(k):
                        cluster_points = distance_matrix[np.ix_(labels == cluster, labels == cluster)]
                        intra_cluster_distances.append(np.mean(cluster_points))

                    avg_distances.append(np.mean(intra_cluster_distances))

                    # Calcular el Silhouette Score
                    silhouette_avg = silhouette_score(pca_data, labels)
                    silhouettes.append(silhouette_avg)

                # Graficar el método del codo
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                ax1.plot(K, avg_distances, 'bo-')
                ax1.set_xlabel('Número de clusters (k)')
                ax1.set_ylabel('Distancia intra-cluster promedio')
                ax1.set_title('Método del codo para Agglomerative Clustering')
                st.pyplot(fig1)

                # Graficar el Silhouette Score
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                ax2.plot(K, silhouettes, 'go-')
                ax2.set_xlabel('Número de clusters (k)')
                ax2.set_ylabel('Silhouette Score')
                ax2.set_title('Silhouette Score para Agglomerative Clustering')
                st.pyplot(fig2)

        ### clustering ###
        ###

            import pandas as pd
            import numpy as np
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import AgglomerativeClustering

            # Supongamos que df_filtered ya está definido y contiene la columna 'IMME'
            selected_columns = ['IMME']

            # Filtrar el DataFrame para incluir solo las columnas seleccionadas y eliminar valores faltantes
            numeric_data_2 = df_filtered[selected_columns].dropna()

            # Verificar el número de muestras
            n_samples = len(numeric_data_2)

            # Si no hay suficientes muestras, asignar un solo cluster a todos los datos
            if n_samples < 2:
                st.warning("Número insuficiente de muestras para clustering. Todos los datos se asignarán a un solo cluster.")
                df_filtered['Cluster'] = 0  # Asignar a todos la misma etiqueta de cluster
            else:
                # Definir el número de clusters adecuado, respetando el rango permitido
                n_clusters = min(4, max(2, n_samples - 1))

                # Normalizar los datos
                scaler = StandardScaler()
                normalized_data_2 = scaler.fit_transform(numeric_data_2)

                # Aplicar Agglomerative Clustering
                clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
                labels_2019 = clustering.fit_predict(normalized_data_2)

                # Agregar las etiquetas al DataFrame original filtrado
                df_filtered = df_filtered.loc[numeric_data_2.index]  # Alinear índices
                df_filtered['Cluster'] = labels_2019

            # Filtrar el DataFrame para incluir solo las columnas seleccionadas y el cluster
            numeric_columns_2 = df_filtered[selected_columns + ['Cluster']]

            numeric_columns_2  # Mostrar el DataFrame resultante con las columnas seleccionadas y el cluster

        ####

            # Crear un gráfico de caja individual para cada parámetro y comparar los clusters
            for column in selected_columns_renamed:
                # Obtener los datos de cada cluster para el parámetro actual
                cluster_data = [df_filtered[df_filtered['Cluster'] == cluster][column] for cluster in df_filtered['Cluster'].unique()]

                # Calcular los quintiles
                quintile_1 = df_filtered[column].quantile(0.20)
                quintile_2 = df_filtered[column].quantile(0.40)
                quintile_3 = df_filtered[column].quantile(0.60)
                quintile_4 = df_filtered[column].quantile(0.80)

                # Crear la figura para el gráfico de caja
                fig = go.Figure()

                # Agregar el gráfico de caja para cada cluster
                for j, cluster_series in enumerate(cluster_data):
                    fig.add_trace(go.Box(y=cluster_series, boxpoints='all', notched=True, name=f'Cluster {j}'))

                # Agregar líneas horizontales para los quintiles
                fig.add_shape(type="line",
                      x0=0, x1=1, y0=quintile_1, y1=quintile_1,
                      line=dict(color="blue", width=2, dash="dash"),
                      xref="paper", yref="y")
                fig.add_shape(type="line",
                      x0=0, x1=1, y0=quintile_2, y1=quintile_2,
                      line=dict(color="green", width=2, dash="dash"),
                      xref="paper", yref="y")
                fig.add_shape(type="line",
                      x0=0, x1=1, y0=quintile_3, y1=quintile_3,
                      line=dict(color="orange", width=2, dash="dash"),
                      xref="paper", yref="y")
                fig.add_shape(type="line",
                      x0=0, x1=1, y0=quintile_4, y1=quintile_4,
                      line=dict(color="red", width=2, dash="dash"),
                      xref="paper", yref="y")

                # Actualizar el diseño
                fig.update_layout(title_text=f'Comparación de Clusters - {column}',
                          xaxis_title="Clusters",
                          yaxis_title=column,
                          showlegend=False)

                # Mostrar el gráfico en Streamlit
                st.plotly_chart(fig)

            # Calcular el percentil 40 global de 'IMME'
            percentile_40_IMME = df_combined_2['IMME'].quantile(0.40)

            # Crear un DataFrame vacío para almacenar las filas que cumplen la condición
            df_filtered_2 = pd.DataFrame()
            percentages_deleted = {}

            for cluster in df_filtered['Cluster'].unique():
                # Filtrar el DataFrame por cada cluster
                cluster_data = df_filtered[df_filtered['Cluster'] == cluster]
                # Mantener solo las filas con 'IMME' menor o igual al percentil 40 global
                filtered_cluster_data = cluster_data[cluster_data['IMME'] <= percentile_40_IMME]
                # Agregar las filas filtradas al nuevo DataFrame
                df_filtered_2 = pd.concat([df_filtered_2, filtered_cluster_data])
                # Calcular el porcentaje de filas eliminadas en cada cluster
                percentage_deleted = 100 * (1 - len(filtered_cluster_data) / len(cluster_data))
                percentages_deleted[cluster] = percentage_deleted

            # Convertir los porcentajes a un DataFrame para visualizar
            percentages_df = pd.DataFrame(list(percentages_deleted.items()), columns=['Cluster', 'Percentage Deleted'])

            # Mostrar el DataFrame con los datos filtrados
            st.write("### DataFrame Filtrado por Cluster con 'IMME' <= Percentil 40")
            st.dataframe(df_filtered_2)

            # Crear el diagrama de barras para mostrar el porcentaje de filas eliminadas por cluster
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(percentages_df['Cluster'], percentages_df['Percentage Deleted'], color='purple', alpha=0.7)
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Porcentaje de Filas Eliminadas')
            ax.set_title('Porcentaje de Filas Eliminadas por Cluster (IMME <= Percentil 40% Global)')

            # Mostrar el gráfico de barras en Streamlit
            st.pyplot(fig)


            # Filtrar el DataFrame combinado para obtener pacientes no en df_filtered_2
            df_combined_2_filtered = df_combined_2[~df_combined_2['folio_paciente'].isin(df_filtered_2['folio_paciente'])]

            # Filtrar las columnas numéricas seleccionadas en ambos DataFrames
            numeric_columns_filtered_2 = df_filtered_2[selected_columns_renamed]
            numeric_columns_combined_2 = df_combined_2[selected_columns_renamed]

            # Crear un gráfico de caja para cada parámetro comparando df_filtered_2 y df_combined_2_filtered
            for column in numeric_columns_filtered_2.columns:
                # Calcular los quintiles en df_combined_2
                quintile_1 = numeric_columns_combined_2[column].quantile(0.20)
                quintile_2 = numeric_columns_combined_2[column].quantile(0.40)
                quintile_3 = numeric_columns_combined_2[column].quantile(0.60)
                quintile_4 = numeric_columns_combined_2[column].quantile(0.80)

                # Crear la figura para el gráfico de caja
                fig = go.Figure()

                # Agregar gráfico de caja para los datos de df_filtered_2
                fig.add_trace(go.Box(
                    y=numeric_columns_filtered_2[column],
                    boxpoints='all',
                    notched=True,
                    name='Sarcopenia',
                    marker=dict(color='blue')
                ))

                # Agregar gráfico de caja para los datos de df_combined_2_filtered
                fig.add_trace(go.Box(
                    y=df_combined_2_filtered[column],
                    boxpoints='all',
                    notched=True,
                    name='Resto de los datos',
                    marker=dict(color='green')
                ))

                # Agregar líneas horizontales para los quintiles calculados en df_combined_2
                fig.add_shape(type="line",
                      x0=0, x1=1, y0=quintile_1, y1=quintile_1,
                      line=dict(color="blue", width=2, dash="dash"),
                      xref="paper", yref="y")

                fig.add_shape(type="line",
                      x0=0, x1=1, y0=quintile_2, y1=quintile_2,
                      line=dict(color="green", width=2, dash="dash"),
                      xref="paper", yref="y")

                fig.add_shape(type="line",
                      x0=0, x1=1, y0=quintile_3, y1=quintile_3,
                      line=dict(color="orange", width=2, dash="dash"),
                      xref="paper", yref="y")

                fig.add_shape(type="line",
                      x0=0, x1=1, y0=quintile_4, y1=quintile_4,
                      line=dict(color="red", width=2, dash="dash"),
                      xref="paper", yref="y")

                # Actualizar el diseño del gráfico
                fig.update_layout(
                title_text=f'Comparación entre Sarcopenia y Resto - {column}',
                xaxis_title="DataFrames",
                yaxis_title=column,
                showlegend=False
                )

                # Mostrar el gráfico en Streamlit
                st.plotly_chart(fig)

            ##############################  Marcha  ##############################################33

            import pandas as pd
            import numpy as np
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import AgglomerativeClustering

            # Suponiendo que df_filtered_2 ya está definido
            selected_columns = ['Marcha']

            # Filtrar el DataFrame para incluir solo las columnas seleccionadas y eliminar valores faltantes
            numeric_data_2 = df_filtered_2[selected_columns].dropna()

            # Verificar el número de muestras
            n_samples = len(numeric_data_2)

            if n_samples < 2:
                st.warning("Número insuficiente de muestras para realizar clustering. Se asignará un único cluster a todos los datos.")
                df_filtered_2['Cluster'] = 0  # Asignar a todos la misma etiqueta de cluster
            else:
                # Definir el número de clusters adecuado, respetando el rango permitido
                n_clusters = min(4, max(2, n_samples - 1))

                # Normalizar los datos
                scaler = StandardScaler()
                normalized_data_2 = scaler.fit_transform(numeric_data_2)

                # Aplicar Agglomerative Clustering
                clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
                labels_2019 = clustering.fit_predict(normalized_data_2)

                # Agregar las etiquetas al DataFrame original filtrado
                df_filtered_2 = df_filtered_2.loc[numeric_data_2.index]  # Alinear índices
                df_filtered_2['Cluster'] = labels_2019

            df_filtered_2  # Mostrar las primeras filas del DataFrame resultante


            from matplotlib import pyplot as plt
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            # Filtrar el DataFrame para incluir solo las columnas seleccionadas
            numeric_columns_2 = df_filtered_2[selected_columns_renamed + ['Cluster']]

            # Crear un gráfico de caja individual para cada parámetro y comparar los clusters
            for column in selected_columns_renamed:
                # Obtener los datos de cada cluster para el parámetro actual
                cluster_data = [df_filtered_2[df_filtered_2['Cluster'] == cluster][column] for cluster in df_filtered_2['Cluster'].unique()]

                # Calcular los quintiles (Q1=20%, Q2=40%, mediana=Q3=60%, Q4=80%)
                quintile_1 = df_filtered_2[column].quantile(0.20)
                quintile_2 = df_filtered_2[column].quantile(0.40)
                quintile_3 = df_filtered_2[column].quantile(0.60)
                quintile_4 = df_filtered_2[column].quantile(0.80)

                # Crear una nueva figura para el gráfico de caja
                fig = go.Figure()

                # Agregar el gráfico de caja para cada cluster
                for j, cluster_series in enumerate(cluster_data):
                    fig.add_trace(go.Box(y=cluster_series, boxpoints='all', notched=True, name=f'Cluster {j}'))

                # Agregar líneas horizontales para los quintiles
                fig.add_shape(type="line",
                      x0=0, x1=1, y0=quintile_1, y1=quintile_1,
                      line=dict(color="blue", width=2, dash="dash"),
                      xref="paper", yref="y")  # Línea del primer quintil (Q1 = 20%)

                fig.add_shape(type="line",
                      x0=0, x1=1, y0=quintile_2, y1=quintile_2,
                      line=dict(color="green", width=2, dash="dash"),
                      xref="paper", yref="y")  # Línea del segundo quintil (Q2 = 40%)

                fig.add_shape(type="line",
                      x0=0, x1=1, y0=quintile_3, y1=quintile_3,
                      line=dict(color="orange", width=2, dash="dash"),
                      xref="paper", yref="y")  # Línea de la mediana (Q3 = 60%)

                fig.add_shape(type="line",
                      x0=0, x1=1, y0=quintile_4, y1=quintile_4,
                      line=dict(color="red", width=2, dash="dash"),
                      xref="paper", yref="y")  # Línea del cuarto quintil (Q4 = 80%)

                # Actualizar el diseño del gráfico
                fig.update_layout(
                    title_text=f'Comparación de Clusters - {column}',
                    xaxis_title="Clusters",
                    yaxis_title=column,
                    showlegend=False
                )

                # Mostrar el gráfico en Streamlit
                st.plotly_chart(fig)


            # Crear un DataFrame vacío para almacenar las filas que cumplen la condición
            df_filtered_3 = pd.DataFrame()
            percentages_deleted = {}

            for cluster in df_filtered_2['Cluster'].unique():
                # Filtrar el DataFrame por cada cluster
                cluster_data = df_filtered_2[df_filtered_2['Cluster'] == cluster]
                # Mantener solo las filas con 'Marcha' menor a 0.8 m/s
                filtered_cluster_data = cluster_data[cluster_data['Marcha'] < 0.8]

                # Agregar las filas filtradas al nuevo DataFrame
                df_filtered_3 = pd.concat([df_filtered_3, filtered_cluster_data])
                # Calcular el porcentaje de filas eliminadas en cada cluster
                percentage_deleted = 100 * (1 - len(filtered_cluster_data) / len(cluster_data))
                percentages_deleted[cluster] = percentage_deleted

            # Convertir los porcentajes a un DataFrame para visualizar
            percentages_df = pd.DataFrame(list(percentages_deleted.items()), columns=['Cluster', 'Percentage Deleted'])

            # Mostrar el DataFrame con los datos filtrados
            st.write("### DataFrame Filtrado por Cluster con 'Marcha' < 0.8 m/s")
            st.dataframe(df_filtered_3)

            # Crear el diagrama de barras para mostrar el porcentaje de filas eliminadas por cluster
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(percentages_df['Cluster'], percentages_df['Percentage Deleted'], color='purple', alpha=0.7)
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Porcentaje de Filas Eliminadas')
            ax.set_title('Porcentaje de Filas Eliminadas por Cluster (Marcha < 0.8 m/s)')

            # Mostrar el gráfico de barras en Streamlit
            st.pyplot(fig)

            ###

            # Filtrar las columnas numéricas seleccionadas en ambos DataFrames
            numeric_columns_filtered_3 = df_filtered_3[selected_columns_renamed]
            numeric_columns_combined_2 = df_combined_2[selected_columns_renamed]

            # Crear un gráfico de caja para cada parámetro comparando df_filtered_3 y df_combined_2_filtered
            for column in selected_columns_renamed:
                # Calcular los quintiles en df_combined_2
                quintile_1 = numeric_columns_combined_2[column].quantile(0.20)
                quintile_2 = numeric_columns_combined_2[column].quantile(0.40)
                quintile_3 = numeric_columns_combined_2[column].quantile(0.60)
                quintile_4 = numeric_columns_combined_2[column].quantile(0.80)

                # Crear la figura para el gráfico de caja
                fig = go.Figure()

                # Agregar gráfico de caja para los datos de df_filtered_3
                fig.add_trace(go.Box(
                    y=numeric_columns_filtered_3[column],
                    boxpoints='all',
                    notched=True,
                    name='Sarcopenia',
                    marker=dict(color='blue')
                ))

                # Agregar gráfico de caja para los datos de df_combined_2_filtered
                fig.add_trace(go.Box(
                    y=df_combined_2_filtered[column],
                    boxpoints='all',
                    notched=True,
                    name='Resto de los datos',
                    marker=dict(color='green')
                ))

                # Agregar líneas horizontales para los quintiles calculados en df_combined_2
                fig.add_shape(type="line",
                      x0=0, x1=1, y0=quintile_1, y1=quintile_1,
                      line=dict(color="blue", width=2, dash="dash"),
                      xref="paper", yref="y")

                fig.add_shape(type="line",
                      x0=0, x1=1, y0=quintile_2, y1=quintile_2,
                      line=dict(color="green", width=2, dash="dash"),
                      xref="paper", yref="y")

                fig.add_shape(type="line",
                      x0=0, x1=1, y0=quintile_3, y1=quintile_3,
                      line=dict(color="orange", width=2, dash="dash"),
                      xref="paper", yref="y")

                fig.add_shape(type="line",
                      x0=0, x1=1, y0=quintile_4, y1=quintile_4,
                      line=dict(color="red", width=2, dash="dash"),
                      xref="paper", yref="y")

                # Actualizar el diseño para el tamaño y estilo de las fuentes
                fig.update_layout(
                    title_text=f'Comparación entre cluster de "Sarcopenia Grave" y el resto de los pacientes - {column}',
                    title_font=dict(size=20, family="Arial"),  # Título en negritas y tamaño ajustado
                    xaxis_title="DataFrames",
                    yaxis_title=column,
                    showlegend=False,
                    xaxis=dict(
                        tickfont=dict(
                            size=14,
                            family="Arial"
                        )
                    ),
                    yaxis=dict(
                        title_font=dict(size=18, family="Arial"),
                        tickfont=dict(
                            size=12,
                            family="Arial"
                        )
                    )
                )

                # Mostrar el gráfico en Streamlit
                st.plotly_chart(fig)


            from matplotlib_venn import venn3
            import matplotlib.pyplot as plt

            # Crear conjuntos de folios de cada DataFrame
            folios_df_filtered_3 = set(df_filtered_3['folio_paciente'].tolist())  # Sarcopenia Severa
            folios_df_filtered_2 = set(df_filtered_2['folio_paciente'].tolist())  # Sarcopenia
            folios_df_combined_2 = set(df_combined_2['folio_paciente'].tolist())  # Muestra Completa

            # Crear el diagrama de Venn para los tres conjuntos con colores personalizados
            plt.figure(figsize=(10, 8))
            venn = venn3([folios_df_filtered_3, folios_df_filtered_2, folios_df_combined_2],
                     set_labels=('Sarcopenia Severa', 'Sarcopenia', 'Muestra Completa'))

            # Aplicar los colores personalizados asegurando que "Sarcopenia Severa" sea rojo
            if venn.get_patch_by_id('100'):
                venn.get_patch_by_id('100').set_color('red')      # Solo Sarcopenia Severa
            if venn.get_patch_by_id('010'):
                venn.get_patch_by_id('010').set_color('orange')   # Solo Sarcopenia
            if venn.get_patch_by_id('001'):
                venn.get_patch_by_id('001').set_color('green')    # Solo Muestra Completa

            # Asignar colores a las intersecciones
            if venn.get_patch_by_id('110'):
                venn.get_patch_by_id('110').set_color('red')      # Sarcopenia Severa dentro de Sarcopenia
            if venn.get_patch_by_id('011'):
                venn.get_patch_by_id('011').set_color('orange')   # Sarcopenia dentro de Muestra Completa
            if venn.get_patch_by_id('111'):
                venn.get_patch_by_id('111').set_color('red')      # Sarcopenia Severa dentro de Sarcopenia y Muestra Completa

            # Ajustar transparencia para todas las áreas
            for patch_id in ['100', '010', '001', '110', '011', '111']:
                if venn.get_patch_by_id(patch_id):
                    venn.get_patch_by_id(patch_id).set_alpha(0.6)

            # Título del gráfico
            plt.title('Proporción de pacientes: Sarcopenia Severa y Sarcopenia (Muestra: "Hombres", IMMS-2019)')

            # Mostrar el gráfico de Venn en Streamlit
            st.pyplot(plt)
        ####################################

        with st.expander("Clusters"):
            import matplotlib.pyplot as plt
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            from sklearn.cluster import AgglomerativeClustering
            from scipy.spatial.distance import pdist, squareform
            from sklearn.metrics import silhouette_score


            # Seleccionar las columnas y normalizar los datos
            selected_columns = ['Fuerza', 'Marcha', 'IMME']
            numeric_data_2 = df_combined_2[selected_columns].dropna()
            scaler = StandardScaler()
            normalized_data_2 = scaler.fit_transform(numeric_data_2)

            # Aplicar PCA para reducir la dimensionalidad
            pca = PCA(n_components=3)  # Puedes ajustar el número de componentes según sea necesario
            pca_data = pca.fit_transform(normalized_data_2)

            # Calcular la matriz de distancias
            distance_matrix = squareform(pdist(pca_data))

            # Aplicar Agglomerative Clustering
            avg_distances = []
            silhouettes = []
            K = range(2, 15)
            for k in K:
                clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
                labels = clustering.fit_predict(pca_data)

                # Calcular la distancia intra-cluster
                intra_cluster_distances = []
                for cluster in range(k):
                    cluster_points = distance_matrix[np.ix_(labels == cluster, labels == cluster)]
                    intra_cluster_distances.append(np.mean(cluster_points))

                avg_distances.append(np.mean(intra_cluster_distances))

                # Calcular el Silhouette Score
                silhouette_avg = silhouette_score(pca_data, labels)
                silhouettes.append(silhouette_avg)

            #  Graficar el método del codo
            st.header("Método del codo para Clustering jerárquico aglomerativo")
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            ax1.plot(K, avg_distances, 'bo-')
            ax1.set_xlabel('Número de clusters (k)')
            ax1.set_ylabel('Distancia intra-cluster promedio')
            ax1.set_title('Método del codo para Clustering jerárquico aglomerativo')
            st.pyplot(fig1)

            # Graficar el Silhouette Score
            st.header("Silhouette Score para Clustering jerárquico aglomerativo")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.plot(K, silhouettes, 'go-')
            ax2.set_xlabel('Número de clusters (k)')
            ax2.set_ylabel('Score de silueta')
            ax2.set_title('Silhouette Score para Clustering jerárquico aglomerativo')
            st.pyplot(fig2)

            #df_combined_2=df_combined_2.dropna()

            import streamlit as st
            import pandas as pd
            import numpy as np
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import AgglomerativeClustering

            df_combined_2=df_combined_copy

            selected_columns=['P113','P112_vel','IMME']

            # Seleccionar las columnas específicas para clustering
            numeric_data_2 = df_combined_2[selected_columns]

            # Título y descripción en Streamlit
            st.title("Clustering jerárquico aglomerativo")
            st.write("Esta aplicación permite aplicar clustering jerárquico aglomerativo a datos seleccionados y definir el número de clusters.")

            # Caja para definir el número de clusters
            n_clusters = st.sidebar.number_input("Número de clusters", min_value=2, max_value=10, value=4, step=1)

            # Normalizar los datos
            scaler = StandardScaler()
            normalized_data_2 = scaler.fit_transform(numeric_data_2)

            # Aplicar Agglomerative Clustering
            clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            labels_2019 = clustering.fit_predict(normalized_data_2)

            # Agregar las etiquetas al DataFrame original filtrado
            df_combined_2['Cluster'] = labels_2019

            # Mostrar las primeras filas del DataFrame resultante
            st.write("Resultados del clustering:")
            st.dataframe(df_combined_2)

            # Agregar un mensaje sobre el número de clusters usados
            st.write(f"Clustering realizado con **{n_clusters}** clusters.")

            ####################################

            import streamlit as st
            import plotly.graph_objects as go

            # Renombrar las columnas
            df_combined_2 = df_combined_2.rename(columns={
                'P112_vel': 'Marcha',
                'P113': 'Fuerza',
                'P125': 'P. Tricipital',
                'P128': 'P. Pantorrilla',
                'IMC': 'IMC',
                'P127': 'Biceps',
                'P126': 'P. subescapular',
                'P121': 'Cintura',
                'P123': 'Muslo',
                'P120': 'Brazo',
                'P122': 'Cadera',
                'P124': 'Pantorrilla',
                'P117': 'Peso'
            })

            # Seleccionar las columnas específicas con los nuevos nombres
            selected_columns_renamed = [
                'Marcha', 'Fuerza', 'P. Tricipital', 'P. Pantorrilla',
                'IMC', 'Biceps', 'P. subescapular', 'Cintura', 'Muslo', 'Brazo', 'Cadera', 'Pantorrilla', 'Peso', 'IMME'
            ]

            # Filtrar el DataFrame para incluir solo las columnas seleccionadas
            numeric_columns_2 = df_combined_2[selected_columns_renamed]

            st.title("Comparación de Clusters por Parámetro")

            # Crear un gráfico de caja individual para cada parámetro y comparar los clusters
            for column in numeric_columns_2.columns:
                # Obtener los datos de cada cluster para el parámetro actual
                cluster_data = [df_combined_2[df_combined_2['Cluster'] == cluster][column] for cluster in range(8)]

                # Calcular los quintiles (Q1=20%, Q2=40%, mediana=Q3=60%, Q4=80%)
                quintile_1 = df_combined_2[column].quantile(0.20)
                quintile_2 = df_combined_2[column].quantile(0.40)
                quintile_3 = df_combined_2[column].quantile(0.60)
                quintile_4 = df_combined_2[column].quantile(0.80)

                # Crear una nueva figura para el gráfico de caja
                fig = go.Figure()

                # Agregar el gráfico de caja para cada cluster
                for j in range(6):
                    fig.add_trace(go.Box(y=cluster_data[j], boxpoints='all', notched=True, name=f'Cluster {j}'))

                # Agregar líneas horizontales para los quintiles
                fig.add_shape(type="line",
                        x0=0, x1=1, y0=quintile_1, y1=quintile_1,
                        line=dict(color="blue", width=2, dash="dash"),
                        xref="paper", yref="y")

                fig.add_shape(type="line",
                        x0=0, x1=1, y0=quintile_2, y1=quintile_2,
                        line=dict(color="green", width=2, dash="dash"),
                        xref="paper", yref="y")

                fig.add_shape(type="line",
                        x0=0, x1=1, y0=quintile_3, y1=quintile_3,
                        line=dict(color="orange", width=2, dash="dash"),
                        xref="paper", yref="y")

                fig.add_shape(type="line",
                        x0=0, x1=1, y0=quintile_4, y1=quintile_4,
                        line=dict(color="red", width=2, dash="dash"),
                        xref="paper", yref="y")

                # Actualizar el diseño para hacer las etiquetas más grandes y en negritas
                fig.update_layout(
                    title_text=f'Comparación de Clusters - {column}',
                    title_font=dict(size=18, family="Arial"),
                    xaxis_title=" ",
                    yaxis_title=column,
                    xaxis=dict(
                        title_font=dict(size=16, family="Arial"),
                        tickfont=dict(size=14, family="Arial")
                    ),
                    yaxis=dict(
                        title_font=dict(size=16, family="Arial"),
                        tickfont=dict(size=14, family="Arial")
                    ),
                    showlegend=True
                )

                # Mostrar el gráfico en Streamlit
                st.plotly_chart(fig)



        import streamlit as st
        import pandas as pd
        from scipy.stats import ttest_ind
        import itertools

        # Definir las columnas de interés y los clusters
        variables_de_interes = selected_columns_renamed
        clusters = df_combined_2['Cluster'].unique()

        # Realizar pruebas t para cada par de clusters y cada variable
        resultados_pruebas_t = {}

        for variable in variables_de_interes:
            resultados_pruebas_t[variable] = {}
            for cluster1, cluster2 in itertools.combinations(clusters, 2):
                grupo1 = df_combined_2[df_combined_2['Cluster'] == cluster1][variable]
                grupo2 = df_combined_2[df_combined_2['Cluster'] == cluster2][variable]

                # Realizar la prueba t
                t_stat, p_valor = ttest_ind(grupo1, grupo2)

                # Guardar los resultados
                resultados_pruebas_t[variable][f'Cluster {cluster1} vs Cluster {cluster2}'] = {'t_stat': t_stat, 'p_valor': p_valor}


        # Mostrar los resultados sin anidar expanders
        with st.expander('Pruebas de hipótesis'):
            for variable, resultados in resultados_pruebas_t.items():
                st.write(f'**Resultados para la variable: {variable}**')
                for comparacion, valores in resultados.items():
                    st.write(f'{comparacion}: t_stat = {valores["t_stat"]:.4f}, p_valor = {valores["p_valor"]:.4f}')


        with st.expander("**Análisis de los clusters**"):
        
            import streamlit as st
            from sklearn.metrics import silhouette_score
            from sklearn.manifold import TSNE
            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt
            import numpy as np
            from scipy.stats import gaussian_kde

            # Ajustar el número de componentes de PCA para reducir la dimensionalidad inicial
            st.title("Análisis de Clustering con t-SNE y PCA")
            st.write("Este análisis aplica PCA y t-SNE para visualizar clusters en datos reducidos dimensionalmente.")

            # Paso 1: Reducir la dimensionalidad con PCA
            pca = PCA(n_components=3)  # Reducir a 3 componentes principales
            pca_data = pca.fit_transform(normalized_data_2)

            # Paso 2: Aplicar t-SNE sobre los datos de PCA
            tsne = TSNE(n_components=2, perplexity=30, learning_rate=40, n_iter=300, early_exaggeration=10)
            tsne_data_2 = tsne.fit_transform(pca_data)

            # Paso 3: Crear el gráfico de t-SNE ajustado con curvas de densidad de kernel
            fig, ax = plt.subplots()
            labels = df_combined_2['Cluster']
            for cluster in np.unique(labels):
                indices = np.where(labels == cluster)
                ax.scatter(tsne_data_2[indices, 0], tsne_data_2[indices, 1], label=f'Cluster {cluster}')
    
                # Densidad de kernel
                kde = gaussian_kde(tsne_data_2[indices].T)
                x_range = np.linspace(np.min(tsne_data_2[:, 0]) - 1, np.max(tsne_data_2[:, 0]) + 1, 100)
                y_range = np.linspace(np.min(tsne_data_2[:, 1]) - 1, np.max(tsne_data_2[:, 1]) + 1, 100)
                xx, yy = np.meshgrid(x_range, y_range)
                positions = np.vstack([xx.ravel(), yy.ravel()])
                zz = np.reshape(kde(positions).T, xx.shape)
                ax.contour(xx, yy, zz, colors='k', alpha=0.5)

            # Añadir detalles al gráfico
            ax.legend()
            ax.set_title('Gráfico de Dispersión de t-SNE con Curvas de Densidad de Kernel')

            # Mostrar el gráfico en Streamlit
            st.pyplot(fig)
        ################################################

            import streamlit as st
            import matplotlib.pyplot as plt
            from matplotlib_venn import venn2
            import pandas as pd

            # Crear conjuntos de folios del DataFrame filtrado
            set_folios_df_filtered_2 = set(df_filtered_2['folio_paciente'])

            # Crear una caja de entrada para el número de clusters
#            num_clusters = st.number_input("Número de clusters", min_value=1, max_value=10, value=4, step=1)

            st.title("Diagramas de Venn por Cluster")
            st.write("Diagrama de Venn entre el grupo de pacientes con sarcopenia grave y cada cluster.")

            # Crear diagramas de Venn para cada cluster
            for cluster_num in range(n_clusters):
                # Crear conjunto de folios para el cluster actual
                cluster_data = df_combined_2[df_combined_2['Cluster'] == cluster_num]
                set_folios_df_cluster = set(cluster_data['folio_paciente'])
    
                # Crear el diagrama de Venn
                fig, ax = plt.subplots(figsize=(8, 6))
                venn2([set_folios_df_filtered_2, set_folios_df_cluster],
                set_labels=('Sarcopenia', f'Cluster {cluster_num}'))
                ax.set_title(f"Diagrama de Venn entre el grupo de pacientes con sarcopenia grave y el Cluster {cluster_num}")
    
                # Mostrar el gráfico en Streamlit
                st.pyplot(fig)



            # Variable para almacenar el cluster con la mayor intersección
            max_similarity = 0
            best_cluster = None

            # Calcular la intersección para cada cluster y encontrar el de mayor similitud
            for cluster_num in range(n_clusters):
                # Crear conjunto de folios para el cluster actual
                cluster_data = df_combined_2[df_combined_2['Cluster'] == cluster_num]
                set_folios_df_cluster = set(cluster_data['folio_paciente'])
    
                # Calcular el grado de intersección
                intersection_size = len(set_folios_df_filtered_2.intersection(set_folios_df_cluster))
    
                # Actualizar el cluster con la mayor intersección
                if intersection_size > max_similarity:
                    max_similarity = intersection_size
                    best_cluster = cluster_num

            # Mostrar el cluster con mayor similitud
            if best_cluster is not None:
                st.write(f"El cluster con mayor similitud con el grupo de sarcopenia grave es el Cluster {best_cluster} con {max_similarity} elementos en común.")
    
                # Crear el diagrama de Venn para el cluster con mayor intersección
                fig, ax = plt.subplots(figsize=(8, 6))
                set_folios_df_best_cluster = set(df_combined_2[df_combined_2['Cluster'] == best_cluster]['folio_paciente'])
                venn2([set_folios_df_filtered_2, set_folios_df_best_cluster],
                    set_labels=('Sarcopenia', f'Cluster {best_cluster}'))
                ax.set_title(f"Diagrama de Venn entre Sarcopenia grave y el Cluster {best_cluster}")
    
                # Mostrar el gráfico en Streamlit
                st.pyplot(fig)
            else:
                st.write("No se encontraron intersecciones entre los clusters y el grupo de sarcopenia grave.")




            ################################################
            import streamlit as st
            import matplotlib.pyplot as plt
            from matplotlib_venn import venn2
            import pandas as pd

            # Crear conjuntos de folios del DataFrame filtrado
            set_folios_df_filtered_3 = set(df_filtered_3['folio_paciente'])

            # Crear una caja de entrada para el número de clusters
#            num_clusters = st.number_input("Número de clusters", min_value=1, max_value=10, value=4, step=1)

            st.title("Diagramas de Venn por Cluster")
            st.write("Diagrama de Venn entre el grupo de pacientes con sarcopenia grave y cada cluster.")

            # Crear diagramas de Venn para cada cluster
            for cluster_num in range(n_clusters):
                # Crear conjunto de folios para el cluster actual
                cluster_data = df_combined_2[df_combined_2['Cluster'] == cluster_num]
                set_folios_df_cluster = set(cluster_data['folio_paciente'])
    
                # Crear el diagrama de Venn
                fig, ax = plt.subplots(figsize=(8, 6))
                venn2([set_folios_df_filtered_3, set_folios_df_cluster],
                    set_labels=('Sarcopenia grave', f'Cluster {cluster_num}'))
                ax.set_title(f"Diagrama de Venn entre el grupo de pacientes con sarcopenia grave y el Cluster {cluster_num}")
    
                # Mostrar el gráfico en Streamlit
                st.pyplot(fig)



            # Variable para almacenar el cluster con la mayor intersección
            max_similarity = 0
            best_cluster = None

            # Calcular la intersección para cada cluster y encontrar el de mayor similitud
            for cluster_num in range(n_clusters):
                # Crear conjunto de folios para el cluster actual
                cluster_data = df_combined_2[df_combined_2['Cluster'] == cluster_num]
                set_folios_df_cluster = set(cluster_data['folio_paciente'])
    
                # Calcular el grado de intersección
                intersection_size = len(set_folios_df_filtered_3.intersection(set_folios_df_cluster))
    
                # Actualizar el cluster con la mayor intersección
                if intersection_size > max_similarity:
                    max_similarity = intersection_size
                    best_cluster = cluster_num

            # Mostrar el cluster con mayor similitud
            if best_cluster is not None:
                st.write(f"El cluster con mayor similitud con el grupo de sarcopenia grave es el Cluster {best_cluster} con {max_similarity} elementos en común.")
    
                # Crear el diagrama de Venn para el cluster con mayor intersección
                fig, ax = plt.subplots(figsize=(8, 6))
                set_folios_df_best_cluster = set(df_combined_2[df_combined_2['Cluster'] == best_cluster]['folio_paciente'])
                venn2([set_folios_df_filtered_3, set_folios_df_best_cluster],
                    set_labels=('Sarcopenia grave', f'Cluster {best_cluster}'))
                ax.set_title(f"Diagrama de Venn entre Sarcopenia grave y el Cluster {best_cluster}")
    
                # Mostrar el gráfico en Streamlit
                st.pyplot(fig)
            else:
                st.write("No se encontraron intersecciones entre los clusters y el grupo de sarcopenia grave.")


            import streamlit as st
            import plotly.graph_objects as go
            import pandas as pd

            # Definir las columnas seleccionadas para los gráficos de caja
            selected_columns_renamed = [
                'Marcha', 'Fuerza', 'P. Tricipital', 'P. Pantorrilla',
                'IMC', 'Biceps', 'P. subescapular', 'Cintura', 'Muslo', 'Brazo', 'Cadera', 'Pantorrilla', 'Peso', 'IMME'
            ]

            # Crear conjuntos de folios de df_filtered_2
            set_folios_df_filtered_2 = set(df_filtered_2['folio_paciente'])

            # Calcular el cluster con la mayor intersección
            num_clusters = df_combined_2['Cluster'].nunique()
            max_similarity = 0
            best_cluster = None

            for cluster_num in range(num_clusters):
                cluster_data = df_combined_2[df_combined_2['Cluster'] == cluster_num]
                set_folios_df_cluster = set(cluster_data['folio_paciente'])
                intersection_size = len(set_folios_df_filtered_2.intersection(set_folios_df_cluster))
    
                if intersection_size > max_similarity:
                    max_similarity = intersection_size
                    best_cluster = cluster_num

            # Obtener los datos del cluster con mayor intersección
            if best_cluster is not None:
                st.title("Comparación de Gráficos de Caja")
                st.write(f"Comparación entre la intersección de pacientes con sarcopenia y el Cluster {best_cluster}")

                df_best_cluster = df_combined_2[df_combined_2['Cluster'] == best_cluster]
                set_folios_df_best_cluster = set(df_best_cluster['folio_paciente'])
                interseccion_folios = set_folios_df_filtered_2.intersection(set_folios_df_best_cluster)
                solo_df_best_cluster = set_folios_df_best_cluster - interseccion_folios

                # Filtrar los DataFrames para la intersección y el conjunto sin intersección
                df_best_cluster_interseccion = df_best_cluster[df_best_cluster['folio_paciente'].isin(interseccion_folios)]
                df_best_cluster_solo = df_best_cluster[df_best_cluster['folio_paciente'].isin(solo_df_best_cluster)]

                # Crear gráficos de caja individuales para cada columna
                for column in selected_columns_renamed:
                    fig = go.Figure()

                    # Caja de datos en intersección
                    box_intersection = go.Box(y=df_best_cluster_interseccion[column], name='Intersección con Sarcopenia', boxpoints='all', notched=True, marker=dict(color='blue'))
                    fig.add_trace(box_intersection)

                    # Caja de datos sin intersección
                    box_non_intersection = go.Box(y=df_best_cluster_solo[column], name=f'Solo en Cluster {best_cluster}', boxpoints='all', notched=True, marker=dict(color='green'))
                    fig.add_trace(box_non_intersection)

                    # Calcular los quintiles (Q1=20%, Q2=40%, mediana=Q3=60%, Q4=80%) en df_best_cluster
                    quintile_1_2 = df_best_cluster[column].quantile(0.20)
                    quintile_2_2 = df_best_cluster[column].quantile(0.40)
                    quintile_3_2 = df_best_cluster[column].quantile(0.60)
                    quintile_4_2 = df_best_cluster[column].quantile(0.80)

                    # Agregar líneas horizontales para los quintiles
                    fig.add_shape(type="line",
                          x0=0, x1=1, y0=quintile_1_2, y1=quintile_1_2,
                          line=dict(color="blue", width=2, dash="dash"),
                          xref="paper", yref="y")

                    fig.add_shape(type="line",
                          x0=0, x1=1, y0=quintile_2_2, y1=quintile_2_2,
                          line=dict(color="green", width=2, dash="dash"),
                          xref="paper", yref="y")

                    fig.add_shape(type="line",
                          x0=0, x1=1, y0=quintile_3_2, y1=quintile_3_2,
                          line=dict(color="orange", width=2, dash="dash"),
                          xref="paper", yref="y")

                    fig.add_shape(type="line",
                          x0=0, x1=1, y0=quintile_4_2, y1=quintile_4_2,
                          line=dict(color="red", width=2, dash="dash"),
                          xref="paper", yref="y")

                    # Actualizar el diseño del gráfico
                    fig.update_layout(
                        title_text=f'Comparación de Gráfico de Caja - {column}',
                        xaxis_title="Conjuntos",
                        yaxis_title=column,
                        showlegend=True,
                        height=600,
                        width=800,
                        margin=dict(t=50, b=50, l=50, r=50)
                    )

                    # Mostrar el gráfico individual en Streamlit
                    st.plotly_chart(fig)


            # Crear conjuntos de folios de df_filtered_3
            set_folios_df_filtered_3 = set(df_filtered_3['folio_paciente'])

            # Calcular el cluster con la mayor intersección
            num_clusters = df_combined_2['Cluster'].nunique()
            max_similarity = 0
            best_cluster = None

            for cluster_num in range(num_clusters):
                cluster_data = df_combined_2[df_combined_2['Cluster'] == cluster_num]
                set_folios_df_cluster = set(cluster_data['folio_paciente'])
                intersection_size = len(set_folios_df_filtered_3.intersection(set_folios_df_cluster))
    
                if intersection_size > max_similarity:
                    max_similarity = intersection_size
                    best_cluster = cluster_num

            # Obtener los datos del cluster con mayor intersección
            if best_cluster is not None:
                st.title("Comparación de Gráficos de Caja")
                st.write(f"Comparación entre la intersección de pacientes con sarcopenia grave y el Cluster {best_cluster}")

                df_best_cluster = df_combined_2[df_combined_2['Cluster'] == best_cluster]
                set_folios_df_best_cluster = set(df_best_cluster['folio_paciente'])
                interseccion_folios = set_folios_df_filtered_3.intersection(set_folios_df_best_cluster)
                solo_df_best_cluster = set_folios_df_best_cluster - interseccion_folios

                # Filtrar los DataFrames para la intersección y el conjunto sin intersección
                df_best_cluster_interseccion = df_best_cluster[df_best_cluster['folio_paciente'].isin(interseccion_folios)]
                df_best_cluster_solo = df_best_cluster[df_best_cluster['folio_paciente'].isin(solo_df_best_cluster)]

                # Crear gráficos de caja individuales para cada columna
                for column in selected_columns_renamed:
                    fig = go.Figure()

                    # Caja de datos en intersección
                    box_intersection = go.Box(y=df_best_cluster_interseccion[column], name='Intersección con Sarcopenia grave', boxpoints='all', notched=True, marker=dict(color='blue'))
                    fig.add_trace(box_intersection)

                    # Caja de datos sin intersección
                    box_non_intersection = go.Box(y=df_best_cluster_solo[column], name=f'Solo en Cluster {best_cluster}', boxpoints='all', notched=True, marker=dict(color='green'))
                    fig.add_trace(box_non_intersection)

                    # Calcular los quintiles (Q1=20%, Q2=40%, mediana=Q3=60%, Q4=80%) en df_best_cluster
                    quintile_1_2 = df_best_cluster[column].quantile(0.20)
                    quintile_2_2 = df_best_cluster[column].quantile(0.40)
                    quintile_3_2 = df_best_cluster[column].quantile(0.60)
                    quintile_4_2 = df_best_cluster[column].quantile(0.80)

                    # Agregar líneas horizontales para los quintiles
                    fig.add_shape(type="line",
                          x0=0, x1=1, y0=quintile_1_2, y1=quintile_1_2,
                          line=dict(color="blue", width=2, dash="dash"),
                          xref="paper", yref="y")

                    fig.add_shape(type="line",
                          x0=0, x1=1, y0=quintile_2_2, y1=quintile_2_2,
                          line=dict(color="green", width=2, dash="dash"),
                          xref="paper", yref="y")

                    fig.add_shape(type="line",
                          x0=0, x1=1, y0=quintile_3_2, y1=quintile_3_2,
                          line=dict(color="orange", width=2, dash="dash"),
                          xref="paper", yref="y")

                    fig.add_shape(type="line",
                          x0=0, x1=1, y0=quintile_4_2, y1=quintile_4_2,
                          line=dict(color="red", width=2, dash="dash"),
                          xref="paper", yref="y")

                    # Actualizar el diseño del gráfico
                    fig.update_layout(
                        title_text=f'Comparación de Gráfico de Caja - {column}',
                        xaxis_title="Conjuntos",
                        yaxis_title=column,
                        showlegend=True,
                        height=600,
                        width=800,
                        margin=dict(t=50, b=50, l=50, r=50)
                    )

                    # Mostrar el gráfico individual en Streamlit
                    st.plotly_chart(fig)
            else:
                st.write("No se encontraron clusters con intersecciones significativas con el grupo de sarcopenia grave.")


##################################
        with st.expander("**Puntos de corte**"):
            import streamlit as st
            import pandas as pd
            import numpy as np
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.model_selection import train_test_split

            # Definir las columnas predictoras y la variable objetivo
            selected_columns = ['IMME', 'Marcha', 'Fuerza']
            X = df_combined_2[selected_columns]
            y = df_combined_2['Cluster']

            # Calcular el cluster con la mayor intersección con df_filtered_3
            set_folios_df_filtered_3 = set(df_filtered_3['folio_paciente'])
            num_clusters = df_combined_2['Cluster'].nunique()
            max_similarity = 0
            best_cluster = None

            for cluster_num in range(num_clusters):
                cluster_data = df_combined_2[df_combined_2['Cluster'] == cluster_num]
                set_folios_df_cluster = set(cluster_data['folio_paciente'])
                intersection_size = len(set_folios_df_filtered_3.intersection(set_folios_df_cluster))
    
                if intersection_size > max_similarity:
                    max_similarity = intersection_size
                    best_cluster = cluster_num

            # Mostrar el cluster con mayor similitud
            st.title("Puntos de Corte para el Cluster con Mayor Intersección")
            if best_cluster is not None:
                st.write(f"El cluster con la mayor intersección con el grupo de sarcopenia grave es el Cluster {best_cluster}.")

                # Definir el número de iteraciones para entrenar múltiples modelos
                n_iterations = st.number_input("Número de iteraciones", min_value=100, max_value=2000, value=1000, step=100)

                # Crear listas para almacenar los puntos de corte de cada modelo
                split_points_best_cluster = {col: [] for col in selected_columns}

                # Entrenar el modelo n_iterations veces y recopilar puntos de corte para clasificar en el cluster con mayor intersección
                for i in range(n_iterations):
                    # Dividir el dataset en conjunto de entrenamiento y prueba
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

                    # Crear y entrenar el modelo de árbol de decisión
                    decision_tree = DecisionTreeClassifier(random_state=i, min_samples_split=5, min_samples_leaf=20)
                    decision_tree.fit(X_train, y_train)

                    # Extraer puntos de corte de los nodos que conducen a la clasificación del cluster con mayor intersección
                    tree = decision_tree.tree_
                    for idx, feature in enumerate(tree.feature):
                        if feature != -2:  # -2 indica que no es un nodo de decisión
                            variable = selected_columns[feature]
                            threshold = tree.threshold[idx]
                            # Considerar solo umbrales que conducen a la clasificación del cluster seleccionado
                            left_child, right_child = tree.children_left[idx], tree.children_right[idx]
                            if (tree.value[left_child][0, best_cluster] > 0 or tree.value[right_child][0, best_cluster] > 0):
                                split_points_best_cluster[variable].append(threshold)

                # Calcular el promedio de los puntos de corte para clasificar en el cluster con mayor intersección
                average_split_points = {var: np.mean(points) for var, points in split_points_best_cluster.items() if points}
    
                # Mostrar los puntos de corte promedio en Streamlit
                st.write("Puntos de corte promedio para clasificar en el Cluster con la mayor intersección:")
                for variable, threshold in average_split_points.items():
                    st.write(f"{variable}: {threshold:.4f}")

            # Mostrar el gráfico de importancia de características
                st.write("Importancia de características del último modelo de árbol de decisión:")
                feature_importances = decision_tree.feature_importances_
                plt.figure(figsize=(8, 6))
                plt.barh(selected_columns, feature_importances, color='skyblue')
                plt.xlabel("Importancia de la característica")
                plt.title("Importancia de características del modelo de árbol de decisión")
                st.pyplot(plt)

            else:
                st.write("No se encontró un cluster con intersecciones significativas con el grupo de sarcopenia grave.")
##########################################################

            import streamlit as st
            import pandas as pd
            import numpy as np
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.model_selection import train_test_split

            # Definir las columnas predictoras y la variable objetivo
            #selected_columns = ['Pantorrilla', 'IMC', 'Brazo', 'Cintura']
            selected_columns = st.multiselect("Selecciona las variables predictoras", selected_columns_renamed, default=['Pantorrilla', 'Brazo', 'Cintura', 'IMC'])

        
            X = df_combined_2[selected_columns]
            y = df_combined_2['Cluster']

            # Calcular el cluster con la mayor intersección con df_filtered_3
            set_folios_df_filtered_3 = set(df_filtered_3['folio_paciente'])
            num_clusters = df_combined_2['Cluster'].nunique()
            max_similarity = 0
            best_cluster = None

            for cluster_num in range(num_clusters):
                cluster_data = df_combined_2[df_combined_2['Cluster'] == cluster_num]
                set_folios_df_cluster = set(cluster_data['folio_paciente'])
                intersection_size = len(set_folios_df_filtered_3.intersection(set_folios_df_cluster))
    
                if intersection_size > max_similarity:
                    max_similarity = intersection_size
                    best_cluster = cluster_num

            # Mostrar el cluster con mayor similitud
            st.title("Puntos de Corte para el Cluster con Mayor Intersección")
            if best_cluster is not None:
                st.write(f"El cluster con la mayor intersección con el grupo de sarcopenia grave es el Cluster {best_cluster}.")

                # Definir el número de iteraciones para entrenar múltiples modelos
                n_iterations = st.number_input("Númro de iteraciones", min_value=100, max_value=2000, value=1000, step=100)

                # Crear listas para almacenar los puntos de corte de cada modelo
                split_points_best_cluster = {col: [] for col in selected_columns}

                # Entrenar el modelo n_iterations veces y recopilar puntos de corte para clasificar en el cluster con mayor intersección
                for i in range(n_iterations):
                    # Dividir el dataset en conjunto de entrenamiento y prueba
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

                    # Crear y entrenar el modelo de árbol de decisión
                    decision_tree = DecisionTreeClassifier(random_state=i, min_samples_split=5, min_samples_leaf=20)
                    decision_tree.fit(X_train, y_train)

                    # Extraer puntos de corte de los nodos que conducen a la clasificación del cluster con mayor intersección
                    tree = decision_tree.tree_
                    for idx, feature in enumerate(tree.feature):
                        if feature != -2:  # -2 indica que no es un nodo de decisión
                            variable = selected_columns[feature]
                            threshold = tree.threshold[idx]
                            # Considerar solo umbrales que conducen a la clasificación del cluster seleccionado
                            left_child, right_child = tree.children_left[idx], tree.children_right[idx]
                            if (tree.value[left_child][0, best_cluster] > 0 or tree.value[right_child][0, best_cluster] > 0):
                                split_points_best_cluster[variable].append(threshold)

                # Calcular el promedio de los puntos de corte para clasificar en el cluster con mayor intersección
                average_split_points = {var: np.mean(points) for var, points in split_points_best_cluster.items() if points}
    
                # Mostrar los puntos de corte promedio en Streamlit
                st.write("Puntos de corte promedio para clasificar en el Cluster con la mayor intersección:")
                for variable, threshold in average_split_points.items():
                    st.write(f"{variable}: {threshold:.4f}")

            # Mostrar el gráfico de importancia de características
                st.write("Importancia de características del último modelo de árbol de decisión:")
                feature_importances = decision_tree.feature_importances_
                plt.figure(figsize=(8, 6))
                plt.barh(selected_columns, feature_importances, color='skyblue')
                plt.xlabel("Importancia de la característica")
                plt.title("Importancia de características del modelo de árbol de decisión")
                st.pyplot(plt)

            else:
                st.write("No se encontró un cluster con intersecciones significativas con el grupo de sarcopenia grave.")

###########

            # Crear subconjuntos con bootstrap y entrenar árboles de decisión
            from sklearn.utils import resample

            # Crear listas para almacenar los puntos de corte de cada modelo
            split_points_best_cluster = {col: [] for col in selected_columns}

            # Número de iteraciones definido por el usuario
            n_iterations = st.number_input("Número de iteraciones (bootstrap)", min_value=100, max_value=2000, value=1000, step=100)

            for i in range(n_iterations):
                # Crear un subconjunto con bootstrap
                X_bootstrap, y_bootstrap = resample(X, y, replace=True, random_state=i)

                # Dividir el dataset en conjunto de entrenamiento y prueba
                X_train, X_test, y_train, y_test = train_test_split(X_bootstrap, y_bootstrap, test_size=0.3, random_state=i)

                # Crear y entrenar el modelo de árbol de decisión
                decision_tree = DecisionTreeClassifier(random_state=i, min_samples_split=5, min_samples_leaf=20)
                decision_tree.fit(X_train, y_train)

                # Extraer puntos de corte de los nodos que conducen a la clasificación del cluster con mayor intersección
                tree = decision_tree.tree_
                for idx, feature in enumerate(tree.feature):
                    if feature != -2:  # -2 indica que no es un nodo de decisión
                        variable = selected_columns[feature]
                        threshold = tree.threshold[idx]
                        # Considerar solo umbrales que conducen a la clasificación del cluster seleccionado
                        left_child, right_child = tree.children_left[idx], tree.children_right[idx]
                        if (tree.value[left_child][0, best_cluster] > 0 or tree.value[right_child][0, best_cluster] > 0):
                            split_points_best_cluster[variable].append(threshold)

            # Calcular el promedio de los puntos de corte para clasificar en el cluster con mayor intersección
            average_split_points = {var: np.mean(points) for var, points in split_points_best_cluster.items() if points}

            # Mostrar los puntos de corte promedio en Streamlit
            st.write("Puntos de corte promedio para clasificar en el Cluster con la mayor intersección (con bootstrap):")
            for variable, threshold in average_split_points.items():
                st.write(f"{variable}: {threshold:.4f}")

            # Mostrar el gráfico de importancia de características
            st.write("Importancia de características del último modelo de árbol de decisión:")
            feature_importances = decision_tree.feature_importances_
            plt.figure(figsize=(8, 6))
            plt.barh(selected_columns, feature_importances, color='skyblue')
            plt.xlabel("Importancia de la característica")
            plt.title("Importancia de características del modelo de árbol de decisión")
            st.pyplot(plt)




########################################################33

#        import streamlit as st
#        import pandas as pd
#        import numpy as np
#        from sklearn.ensemble import RandomForestRegressor
#        from sklearn.metrics import r2_score
#        from sklearn.tree import plot_tree
#        import matplotlib.pyplot as plt
#        from sklearn.model_selection import train_test_split
#        import seaborn as sns

#        # Calcular el cluster con la mayor intersección con df_filtered_3
#        set_folios_df_filtered_3 = set(df_filtered_3['folio_paciente'])
#        num_clusters = df_combined_2['Cluster'].nunique()
#        max_similarity = 0
#        best_cluster = None

#        for cluster_num in range(num_clusters):
#            cluster_data = df_combined_2[df_combined_2['Cluster'] == cluster_num]
#            set_folios_df_cluster = set(cluster_data['folio_paciente'])
#            intersection_size = len(set_folios_df_filtered_3.intersection(set_folios_df_cluster))
#    
#            if intersection_size > max_similarity:
#                max_similarity = intersection_size
#                best_cluster = cluster_num

#        st.title("Análisis de Random Forest para el Cluster con Mayor Intersección")

#        if best_cluster is not None:
#            st.write(f"El cluster con la mayor intersección con el grupo de sarcopenia grave es el Cluster {best_cluster}.")

#            # Filtrar el DataFrame para el cluster de mayor intersección
#            df_best_cluster = df_combined_2[df_combined_2['Cluster'] == best_cluster]

#            # Seleccionar variables predictoras mediante un multiselect
#            all_columns = df_best_cluster.columns.tolist()
#            selected_columns = st.multiselect("Selecciona las variables predictoras", all_columns, default=['Pantorrilla', 'Brazo', 'Cintura', 'IMC'])
    
#            if selected_columns:
#                # Definir las variables predictoras y la variable objetivo
#                X = df_best_cluster[selected_columns]
#                y = df_best_cluster['IMME']

#                # Dividir el dataset en conjunto de entrenamiento y prueba
#                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

#                # Crear y entrenar el modelo Random Forest
#                random_forest = RandomForestRegressor(n_estimators=100, random_state=2, min_samples_split=5, min_samples_leaf=20)
#                random_forest.fit(X_train, y_train)

#                # Evaluar cada árbol individual en el conjunto de prueba
#                best_r2 = -np.inf
#                best_tree_index = 0

#                for i, tree in enumerate(random_forest.estimators_):
#                    y_pred = tree.predict(X_test)
#                    r2 = r2_score(y_test, y_pred)

#                    if r2 > best_r2:
#                        best_r2 = r2
#                        best_tree_index = i

#                # Seleccionar el mejor árbol según R²
#                best_tree = random_forest.estimators_[best_tree_index]

#                # Mostrar el valor de R² del modelo completo en Streamlit
#                overall_r2 = r2_score(y_test, random_forest.predict(X_test))
#                st.write(f"R² del modelo Random Forest: {overall_r2:.2f}")

#                # Graficar el árbol de decisión seleccionado
#                st.write(f"Mejor Árbol Seleccionado del Random Forest (R²: {best_r2:.2f})")
#                fig, ax = plt.subplots(figsize=(40, 20))
#                plot_tree(best_tree, feature_names=selected_columns, filled=True, fontsize=14, ax=ax)
#                st.pyplot(fig)

#                # Mostrar la importancia de las características en el Random Forest
#                feature_importances_rf = pd.DataFrame({
#                    'Feature': selected_columns,
#                    'Importance': random_forest.feature_importances_
#                }).sort_values(by='Importance', ascending=False)

#                # Mostrar gráfico de importancia de características
#                st.write("Importancia de las Características (Random Forest)")
#                fig, ax = plt.subplots(figsize=(10, 6))
#                sns.barplot(x='Importance', y='Feature', data=feature_importances_rf, ax=ax)
#                plt.title("Importancia de las Características (Random Forest)")
#                st.pyplot(fig)

#                # Mostrar la tabla de importancias en Streamlit
#                st.write(feature_importances_rf)
#        else:
#            st.write("No se encontró un cluster con intersecciones significativas con el grupo de sarcopenia grave.")














####################################
    except FileNotFoundError:
        st.error("No se encontró el archivo en la ruta especificada. Verifica la ruta e inténtalo de nuevo.")
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")


else:
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
