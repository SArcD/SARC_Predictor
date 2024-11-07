import pandas as pd

# Ruta al archivo xlsx
archivo_xlsx = "/content/Base 2019 Santiago Arceo.xlsx"

datos = pd.read_excel(archivo_xlsx)

datos
