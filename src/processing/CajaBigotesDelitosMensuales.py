from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataframe_image as dfi # Importar para poder exportar dataframes a imagenes para el propósito de plasmar tablas

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# Se  cargan datos de tanto delitos que cuentan con municipio especificado como aquellos que no, no es requerida esta información para obtener el total de 
# delitos mensuales estos delitos para el estado en general.
df_validos = pd.read_csv(INTERIM_DIR / 'delitos_jalisco_sin_match_geo.csv')


df_no_validos = pd.read_csv(INTERIM_DIR / 'delitos_jalisco_validos.csv')

# Se agregan las filas de los delitos con area geografica no valida a aquellos con municipios válidos, posible por compartir la misma disposición de las columnas
df_delitos = pd.concat([df_validos,df_no_validos], ignore_index=True)

df_delitos = df_delitos.sort_values(by='Año')

years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
months = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
amount_crimes = []

# Se calcula el total de delitos para cada mes listado en el arreglo, recolectando todos los delitos que pertenecen a un año
for year in years:
    df_year = df_delitos[df_delitos['Año'] == year]
    for month in months:
        s_month = df_year[month]
        total = s_month.sum()
        # print(total)
        amount_crimes.append(total.item())
print(amount_crimes)
print(len(amount_crimes))

# Se crea un dataframe que pueda contener el total de meses, 12 por la cantidad de años del histórico. 
data = {
    'Año': [2015]*12 + [2016]*12 + [2017]*12 + [2018]*12 + [2019]*12 + [2020]*12 + [2021]*12 + [2022]*12 + [2023]*12 + [2024]*12 + [2025]*12,
    'Mes': list(range(1,13)) * 11,
    'Cantidad_Delitos': amount_crimes
}
df_delitos_mens = pd.DataFrame(data)

df_delitos_mens.to_csv(PROCESSED_DIR / 'q5_totales_mensuales.csv')

plt.figure(figsize=(12, 6)) # Tamaño de la figura

# Se traza el diagrama de caja y bigotes a partir de la columna Cantidad_Delitos que contiene el total de delitos de un mes y se establecen las categorías como los años
df_delitos_mens.boxplot(column='Cantidad_Delitos', by='Año')

plt.suptitle("")
plt.title('Variación en total de delitos mensuales por año en el estado de Jalisco (2015-2025)', fontsize=15)
plt.xlabel('Año', fontsize=12)
plt.ylabel('Cantidad de Delitos Mensuales', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig(OUTPUTS_DIR / 'q5_var_del_mens.png', dpi=300, bbox_inches='tight')
plt.show()