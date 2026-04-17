from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataframe_image as dfi # Necesario para exportar los dataframes en formato de tabla a imagenes

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

delitos = pd.read_csv(INTERIM_DIR / 'IIEG_Seguridad.csv') # Se carga el dataset  con los datos necesarios

# Primero se agrupan las filas por año, delito y bien jurídico afectado,
# size() permite obtener cantidad de filas contenidas por grupo formado, sirve para conocer la cantidad de incidentes de un tipo de delito en un año específico
# La cuenta del número de delitos que cumplen con los criterios del grupo se almacenan en una nueva columna num_casos
# Se limpia el índice para restaurar el dataframe a partir del groupby
casosAñoTipoBien = delitos.groupby(['year', 'delito', 'bien_afectado']).size().reset_index(name='num_casos')
print(casosAñoTipoBien.head(16))

# Se agrupa el nuevo dataframe por año y bien jurídico afectado
# Contando ya con el número de casos para cada delito, simplemente se suman este para todos afectan a un bien jurídico en particular para el año en 
# cuestión y se renombra la columna a total
anual_bien = casosAñoTipoBien.groupby(['year','bien_afectado'])['num_casos'].sum().reset_index(name='total')
print(anual_bien.head(10))

# Se obtiene un dataframe que contiene las filas del máximo bien jurídico afectado, al rescatar aquella fila con el valor más grande en la columna total para cada grupo (año)
max_bien = anual_bien.loc[anual_bien.groupby('year')['total'].idxmax()]
print(max_bien[['year','bien_afectado','total']])

# Para calcular aquellos delitos que perjudican con un mayor número de casos a cada bien del patrimonio por año
idx = casosAñoTipoBien.groupby(['year', 'bien_afectado'])['num_casos'].idxmax()
resultado = casosAñoTipoBien.loc[idx]
resultado = resultado.sort_values(by=['year', 'bien_afectado'])
print(resultado[['year', 'bien_afectado', 'delito', 'num_casos']])

# Se filtra para obtener los delitos que atentaban contra el patrimonio tenían más frecuencia en cada año
tablaDelitoPatrimonio = resultado[['year', 'bien_afectado', 'delito', 'num_casos']]
tablaDelitoPatrimonio.to_csv(PROCESSED_DIR / 'q4_mayores_delitos_por_bien.csv')
tablaDelitoPatrimonio = tablaDelitoPatrimonio[tablaDelitoPatrimonio['bien_afectado'] == 'El patrimonio']
tablaDelitoPatrimonio.reset_index(drop=True, inplace=True)
print(tablaDelitoPatrimonio.head(10))

# Se exporta la tabla con el delito más frecuente en cada año que daña al patrimonio
imagenTabla = tablaDelitoPatrimonio.copy()
imagenTabla = imagenTabla.drop(columns=['num_casos'])
imagenTabla = imagenTabla.rename(columns={'year': 'Año', 'bien_afectado':'Bien Jurídico Afectado', 'delito':'Delito'})
print(imagenTabla.head(10))
dfi.export(imagenTabla, OUTPUTS_DIR / 'q4_mayor_delito_patrimonio-anio.png', table_conversion='chrome')

# Comienzo de cálculo de los municipios que registran la mayor cantidad de incidentes para cada delito por año
casosAñoTipoMunicipio = delitos.groupby(['year', 'delito', 'municipio']).size().reset_index(name='num_casos')
print(casosAñoTipoMunicipio.head(16))

# Se agrupan las filas por año y delito y se transforma a un dataframe que incluye exclusivamente las filas con el máximo núm de casos registrados para el delito
# en dicho año
idx = casosAñoTipoMunicipio.groupby(['year','delito'])['num_casos'].idxmax()
resultado = casosAñoTipoMunicipio.loc[idx]

print(resultado[['year', 'delito', 'municipio', 'num_casos']])

# Se filtra paara obtener sólo las filas que apliquen para el robo a vehículos particulares
tablaRoboMunicipio = resultado[['year', 'delito', 'municipio', 'num_casos']]
tablaRoboMunicipio.to_csv(PROCESSED_DIR / 'q4_mayores_mun_por_delito.csv')
tablaRoboMunicipio = tablaRoboMunicipio[tablaRoboMunicipio['delito'] == 'ROBO A VEHICULOS PARTICULARES']
tablaRoboMunicipio.reset_index(drop=True, inplace=True)
print(tablaRoboMunicipio.head(10))

# Se exporta el dataframe filtrado en forma de tabla a una imagen
imagenTabla = tablaRoboMunicipio.copy()
imagenTabla = imagenTabla.drop(columns=['num_casos'])
imagenTabla = imagenTabla.rename(columns={'year': 'Año', 'delito':'Delito', 'municipio':'Municipio'})
print(imagenTabla.head(10))
dfi.export(imagenTabla, OUTPUTS_DIR / 'q4_mayor_municipio_robovehc-anio.png', table_conversion='chrome')

years = delitos['year'].unique()

anual_bien.to_csv(PROCESSED_DIR / 'q4_totales_segun_bien_por_anio.csv')

# Elaboración de gráfico de líneas para demostrar la evolución en la incidencia de delitos categorizados por bien jurídico que perjudican 
plt.figure(figsize=(10,5))
plt.plot(years, anual_bien.loc[anual_bien['bien_afectado'] == 'El patrimonio', 'total'], label='El patrimonio', marker='o')
plt.plot(years, anual_bien.loc[anual_bien['bien_afectado'] == 'La familia', 'total'], label='La familia', marker='s')
plt.plot(years, anual_bien.loc[anual_bien['bien_afectado'] == 'La libertad y la seguridad sexual', 'total'], label='La libertad y la seguridad sexual', marker='^')
plt.plot(years, anual_bien.loc[anual_bien['bien_afectado'] == 'La vida y la integridad corporal', 'total'], label='La vida y la integridad corporal', marker='*')
plt.xlabel("Año")
plt.ylabel("Número de incidentes")
plt.title('Tendencia de Incidencia Delictiva por Bien Jurídico Afectado en el estado de Jalisco (2017 - 2026)')
plt.xticks(rotation=45)
plt.legend()
plt.savefig(OUTPUTS_DIR / 'q4_tendencia_bien_jurídico_afectado.png')
plt.show()
