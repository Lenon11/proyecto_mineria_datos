from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import chi2_contingency


# =========================
# RUTAS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RUTA_DELITOS = INTERIM_DIR / "delitos_jalisco_validos.csv"
RUTA_IIEG = INTERIM_DIR / "IIEG_Seguridad_jalisco.csv"
RUTA_GEO = RAW_DIR / "mun_jal.geojson"
RUTA_IND_ANUAL = INTERIM_DIR / "indicadores_anuales_jalisco.csv"
RUTA_IND_BIENAL = INTERIM_DIR / "indicadores_bienales_jalisco.csv"


# =========================
# UTILIDADES
# =========================
def normalizar_texto(texto: str) -> str:
    """
    Normaliza strings para hacer merges robustos:
    - mayúsculas
    - sin acentos
    - sin signos raros
    - espacios limpios
    """
    if pd.isna(texto):
        return np.nan

    texto = str(texto).strip().upper()
    texto = "".join(
        c for c in unicodedata.normalize("NFKD", texto)
        if not unicodedata.combining(c)
    )
    texto = re.sub(r"[^A-Z0-9 ]+", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def normalizar_municipio(texto: str) -> str:
    """
    Normalización especial para municipios.
    Además de limpiar acentos y caracteres,
    homogeniza variaciones comunes.
    """
    texto = normalizar_texto(texto)

    if pd.isna(texto):
        return np.nan

    reemplazos = {
        "SAN PEDRO TLAQUEPAQUE": "SAN PEDRO TLAQUEPAQUE",
        "TLAJOMULCO": "TLAJOMULCO DE ZUNIGA",
        "TLAJOMULCO DE ZUNIGA": "TLAJOMULCO DE ZUNIGA",
        "IXTLAHUACAN DE LOS MEMBRILLOS": "IXTLAHUACAN DE LOS MEMBRILLOS",
        "ZAPOTLANEJO": "ZAPOTLANEJO",
    }

    for patron, destino in reemplazos.items():
        if patron in texto:
            return destino

    return texto


def municipios_amg() -> set[str]:
    """
    Lista usual de municipios del Área Metropolitana de Guadalajara.
    """
    return {
        "GUADALAJARA",
        "ZAPOPAN",
        "SAN PEDRO TLAQUEPAQUE",
        "TLAJOMULCO DE ZUNIGA",
        "TONALA",
        "EL SALTO",
        "JUANACATLAN",
        "IXTLAHUACAN DE LOS MEMBRILLOS",
        "ZAPOTLANEJO",
    }


def detectar_columna(df: pd.DataFrame, opciones: list[str], nombre_logico: str) -> str:
    """
    Detecta una columna válida dentro de varias opciones posibles.
    """
    for col in opciones:
        if col in df.columns:
            return col
    raise KeyError(f"No encontré la columna para '{nombre_logico}'. Opciones esperadas: {opciones}")


def clasificar_delito_general(texto: str) -> str:
    """
    Agrupa subtipos en categorías analíticas.
    Esto evita duplicidad conceptual como:
    VIOLACION / VIOLACION SIMPLE / VIOLACION EQUIPARADA.
    """
    if pd.isna(texto):
        return "OTROS"

    texto = normalizar_texto(texto)

    if (
        "VIOLACION" in texto
        or "ABUSO SEXUAL" in texto
        or "INCESTO" in texto
        or "HOSTIGAMIENTO SEXUAL" in texto
        or "ACOSO SEXUAL" in texto
    ):
        return "VIOLENCIA_SEXUAL"

    if "HOMICIDIO" in texto:
        return "HOMICIDIO"

    if "SECUESTRO" in texto:
        return "SECUESTRO"

    if "EXTORSION" in texto:
        return "EXTORSION"

    if "FEMINICIDIO" in texto:
        return "FEMINICIDIO"

    if "NARCOMENUDEO" in texto:
        return "NARCOMENUDEO"

    if "ROBO" in texto:
        return "ROBO"

    if "DESAPARICION" in texto:
        return "DESAPARICION"

    if "VIOLENCIA FAMILIAR" in texto or "VIOLENCIA INTRAFAMILIAR" in texto:
        return "VIOLENCIA_FAMILIAR"

    return "OTROS"


# =========================
# CARGA Y PREPARACIÓN
# =========================
def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    delitos = pd.read_csv(RUTA_DELITOS)
    iieg = pd.read_csv(RUTA_IIEG)
    geo = gpd.read_file(RUTA_GEO)
    ind_anual = pd.read_csv(RUTA_IND_ANUAL)
    ind_bienal = pd.read_csv(RUTA_IND_BIENAL)
    return delitos, iieg, geo, ind_anual, ind_bienal


def preparar_base_delitos() -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Genera una base consolidada de delitos:
    - agrega zona geográfica (AMG / Interior) desde IIEG
    - agrega población y geometría por municipio
    - crea métricas per cápita
    """
    delitos, iieg, geo, _, _ = cargar_datos()

    col_del_municipio = detectar_columna(delitos, ["Municipio", "municipio"], "municipio en delitos")
    col_del_anio = detectar_columna(delitos, ["Año", "anio", "year"], "año en delitos")
    col_del_tipo = detectar_columna(delitos, ["Tipo de delito", "tipo_delito"], "tipo de delito")
    col_del_subtipo = detectar_columna(delitos, ["Subtipo de delito", "subtipo_delito"], "subtipo de delito")
    col_del_total = detectar_columna(delitos, ["Total_Anual", "total_delitos", "Total"], "total anual")

    col_iieg_municipio = detectar_columna(iieg, ["municipio", "Municipio"], "municipio en IIEG")
    col_iieg_zona = detectar_columna(iieg, ["zona_geografica", "Zona geográfica", "zona"], "zona geográfica")
    col_geo_nom = detectar_columna(geo, ["nomgeo", "municipio", "Municipio"], "nombre municipio geo")
    col_geo_pob = detectar_columna(geo, ["pob_total", "POBTOT", "poblacion"], "población geo")
    col_geo_cve = detectar_columna(geo, ["cve_mun", "CVE_MUN", "id"], "clave municipal geo")

    delitos = delitos.copy()
    iieg = iieg.copy()
    geo = geo.copy()

    delitos["mun_norm"] = delitos[col_del_municipio].apply(normalizar_municipio)
    iieg["mun_norm"] = iieg[col_iieg_municipio].apply(normalizar_municipio)
    geo["mun_norm"] = geo[col_geo_nom].apply(normalizar_municipio)

    delitos[col_del_anio] = pd.to_numeric(delitos[col_del_anio], errors="coerce")
    delitos[col_del_total] = pd.to_numeric(delitos[col_del_total], errors="coerce").fillna(0)
    geo[col_geo_pob] = pd.to_numeric(geo[col_geo_pob], errors="coerce")

    mapa_zona = (
        iieg[["mun_norm", col_iieg_zona]]
        .dropna()
        .groupby("mun_norm")[col_iieg_zona]
        .agg(lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0])
        .reset_index()
        .rename(columns={col_iieg_zona: "zona_geografica"})
    )

    geo_small = geo[[col_geo_cve, col_geo_nom, "mun_norm", col_geo_pob, "geometry"]].copy()
    geo_small = geo_small.rename(
        columns={
            col_geo_cve: "cve_mun",
            col_geo_nom: "nomgeo",
            col_geo_pob: "pob_total",
        }
    )

    delitos = delitos.merge(mapa_zona, on="mun_norm", how="left")
    delitos = delitos.merge(
        geo_small[["mun_norm", "pob_total"]],
        on="mun_norm",
        how="left",
        suffixes=("", "_geo"),
    )

    amg_set = municipios_amg()
    delitos["zona_geografica"] = np.where(
        delitos["zona_geografica"].isna() & delitos["mun_norm"].isin(amg_set),
        "AMG",
        delitos["zona_geografica"],
    )
    delitos["zona_geografica"] = delitos["zona_geografica"].fillna("Interior")

    delitos["es_amg"] = delitos["zona_geografica"].str.upper().eq("AMG")

    delitos["categoria_delito"] = delitos[col_del_subtipo].apply(clasificar_delito_general)

    delitos_mun = (
        delitos.groupby([col_del_municipio, "mun_norm", col_del_anio], as_index=False)[col_del_total]
        .sum()
        .rename(
            columns={
                col_del_municipio: "Municipio",
                col_del_anio: "Año",
                col_del_total: "Total_Anual",
            }
        )
    )

    delitos_geo = delitos_mun.merge(geo_small, on="mun_norm", how="left")

    delitos_geo["tasa_100k"] = np.where(
        delitos_geo["pob_total"].fillna(0) > 0,
        delitos_geo["Total_Anual"] / delitos_geo["pob_total"] * 100_000,
        np.nan,
    )

    delitos_geo = gpd.GeoDataFrame(delitos_geo, geometry="geometry", crs=geo.crs)

    delitos = delitos.rename(
        columns={
            col_del_municipio: "Municipio",
            col_del_anio: "Año",
            col_del_tipo: "Tipo de delito",
            col_del_subtipo: "Subtipo de delito",
            col_del_total: "Total_Anual",
        }
    )

    return delitos, delitos_geo


# =========================
# PREGUNTA 1
# ¿Existe relación entre tipo de delito y región geográfica en el tiempo?
# =========================
def tabla_relacion_tipo_region_tiempo(delitos: pd.DataFrame) -> pd.DataFrame:
    tabla = (
        delitos.groupby(["Año", "zona_geografica", "Tipo de delito"], as_index=False)["Total_Anual"]
        .sum()
        .sort_values(["Año", "zona_geografica", "Total_Anual"], ascending=[True, True, False])
    )
    return tabla


def prueba_chi_cuadrada_tipo_region(delitos: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada año, arma una tabla de contingencia:
    Tipo de delito x zona geográfica
    y corre chi-cuadrada de independencia.
    """
    resultados = []

    for anio, df_anio in delitos.groupby("Año"):
        tabla = pd.pivot_table(
            df_anio,
            index="Tipo de delito",
            columns="zona_geografica",
            values="Total_Anual",
            aggfunc="sum",
            fill_value=0,
        )

        tabla = tabla.loc[tabla.sum(axis=1) > 0, tabla.sum(axis=0) > 0]

        if tabla.shape[0] > 1 and tabla.shape[1] > 1:
            try:
                chi2, pvalue, dof, _ = chi2_contingency(tabla)
                resultados.append(
                    {
                        "Año": int(anio),
                        "chi2": chi2,
                        "p_value": pvalue,
                        "gl": dof,
                        "hay_relacion_estadistica_5pct": pvalue < 0.05,
                    }
                )
            except ValueError:
                continue

    return pd.DataFrame(resultados)


# =========================
# PREGUNTA 2
# ¿Qué municipios han mostrado mayor crecimiento o reducción?
# =========================
def crecimiento_municipal(
    delitos: pd.DataFrame,
    anio_inicio: int = 2015,
    anio_fin: int = 2025,
) -> pd.DataFrame:
    totales = (
        delitos.groupby(["Municipio", "Año"], as_index=False)["Total_Anual"]
        .sum()
    )

    tabla = (
        totales.pivot(index="Municipio", columns="Año", values="Total_Anual")
        .fillna(0)
        .copy()
    )

    if anio_inicio not in tabla.columns or anio_fin not in tabla.columns:
        raise ValueError(f"No existen ambos años en la base: {anio_inicio}, {anio_fin}")

    tabla["cambio_absoluto"] = tabla[anio_fin] - tabla[anio_inicio]
    tabla["cambio_porcentual"] = np.where(
        tabla[anio_inicio] > 0,
        (tabla[anio_fin] - tabla[anio_inicio]) / tabla[anio_inicio] * 100,
        np.nan,
    )
    tabla["CAGR_aprox"] = np.where(
        tabla[anio_inicio] > 0,
        ((tabla[anio_fin] / tabla[anio_inicio]) ** (1 / (anio_fin - anio_inicio)) - 1) * 100,
        np.nan,
    )

    tabla = tabla.reset_index().sort_values("cambio_absoluto", ascending=False)
    return tabla


def top_crecimiento_reduccion(
    delitos: pd.DataFrame,
    anio_inicio: int = 2015,
    anio_fin: int = 2025,
    top_n: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tabla = crecimiento_municipal(delitos, anio_inicio, anio_fin)
    top_crecen = tabla.nlargest(top_n, "cambio_absoluto")
    top_reducen = tabla.nsmallest(top_n, "cambio_absoluto")
    return top_crecen, top_reducen


# =========================
# PREGUNTA 3
# ¿En el año pasado se concentraron en municipios AMG?
# =========================
def concentracion_amg_ultimo_anio(delitos: pd.DataFrame) -> pd.DataFrame:
    ultimo_anio = int(delitos["Año"].max())

    tabla = (
        delitos.loc[delitos["Año"] == ultimo_anio]
        .groupby("zona_geografica", as_index=False)["Total_Anual"]
        .sum()
    )

    total = tabla["Total_Anual"].sum()
    tabla["participacion_pct"] = np.where(total > 0, tabla["Total_Anual"] / total * 100, np.nan)
    tabla["Año"] = ultimo_anio
    return tabla.sort_values("Total_Anual", ascending=False)


def concentracion_amg_por_municipio(delitos: pd.DataFrame) -> pd.DataFrame:
    ultimo_anio = int(delitos["Año"].max())
    tabla = (
        delitos.loc[delitos["Año"] == ultimo_anio]
        .groupby(["zona_geografica", "Municipio"], as_index=False)["Total_Anual"]
        .sum()
        .sort_values(["zona_geografica", "Total_Anual"], ascending=[True, False])
    )
    return tabla


# =========================
# ANÁLISIS EXTRA INTERESANTE
# =========================
def analizar_cifra_negra() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extra útil:
    No afirma ocultamiento directo, pero sí documenta subregistro
    mediante cifra negra oficial.
    """
    _, _, _, ind_anual, ind_bienal = cargar_datos()

    col_ind_anual = detectar_columna(ind_anual, ["Indicador", "indicador"], "indicador anual")
    col_anio_anual = detectar_columna(ind_anual, ["Año", "anio", "year"], "año anual")

    col_ind_bienal = detectar_columna(ind_bienal, ["Indicador", "indicador"], "indicador bienal")
    col_anio_bienal = detectar_columna(ind_bienal, ["Año", "anio", "year"], "año bienal")

    cifra_negra_anual = (
        ind_anual[ind_anual[col_ind_anual].astype(str).str.contains("Cifra Negra", case=False, na=False)]
        .sort_values(col_anio_anual)
        .copy()
    )

    cifra_negra_bienal = (
        ind_bienal[ind_bienal[col_ind_bienal].astype(str).str.contains("Cifra negra", case=False, na=False)]
        .sort_values(col_anio_bienal)
        .copy()
    )

    return cifra_negra_anual, cifra_negra_bienal


def delitos_graves_tendencia(delitos: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa delitos graves por categoría general, evitando duplicar
    subtipos como VIOLACION SIMPLE / VIOLACION EQUIPARADA.
    """
    tabla = (
        delitos.loc[delitos["categoria_delito"] != "OTROS"]
        .groupby(["Año", "categoria_delito"], as_index=False)["Total_Anual"]
        .sum()
        .sort_values(["categoria_delito", "Año"])
    )
    return tabla


def delitos_graves_tendencia_municipio(
    delitos: pd.DataFrame,
    categoria: str | None = None,
) -> pd.DataFrame:
    """
    Permite ver tendencia por municipio para una categoría grave específica.
    """
    df = delitos.loc[delitos["categoria_delito"] != "OTROS"].copy()

    if categoria is not None:
        df = df.loc[df["categoria_delito"] == categoria]

    tabla = (
        df.groupby(["Año", "Municipio", "categoria_delito"], as_index=False)["Total_Anual"]
        .sum()
        .sort_values(["categoria_delito", "Municipio", "Año"])
    )
    return tabla


def tasa_municipal_100k(delitos_geo: gpd.GeoDataFrame, anio: int | None = None) -> gpd.GeoDataFrame:
    if anio is None:
        anio = int(delitos_geo["Año"].max())

    base = delitos_geo.loc[delitos_geo["Año"] == anio].copy()
    return base.sort_values("tasa_100k", ascending=False)


# =========================
# EXPORTACIÓN DE RESULTADOS
# =========================
def exportar_tablas():
    delitos, delitos_geo = preparar_base_delitos()

    tabla_q1 = tabla_relacion_tipo_region_tiempo(delitos)
    chi_q1 = prueba_chi_cuadrada_tipo_region(delitos)
    crecimiento = crecimiento_municipal(delitos)
    top_crecen, top_reducen = top_crecimiento_reduccion(delitos)
    tabla_q3 = concentracion_amg_ultimo_anio(delitos)
    tabla_q3_mun = concentracion_amg_por_municipio(delitos)
    cifra_anual, cifra_bienal = analizar_cifra_negra()
    graves = delitos_graves_tendencia(delitos)
    graves_municipio = delitos_graves_tendencia_municipio(delitos)
    tasas = tasa_municipal_100k(delitos_geo)

    tabla_q1.to_csv(PROCESSED_DIR / "q1_tipo_region_tiempo.csv", index=False)
    chi_q1.to_csv(PROCESSED_DIR / "q1_chi_cuadrada.csv", index=False)
    crecimiento.to_csv(PROCESSED_DIR / "q2_crecimiento_municipal.csv", index=False)
    top_crecen.to_csv(PROCESSED_DIR / "q2_top_crecimiento.csv", index=False)
    top_reducen.to_csv(PROCESSED_DIR / "q2_top_reduccion.csv", index=False)
    tabla_q3.to_csv(PROCESSED_DIR / "q3_concentracion_amg.csv", index=False)
    tabla_q3_mun.to_csv(PROCESSED_DIR / "q3_concentracion_amg_municipios.csv", index=False)
    cifra_anual.to_csv(PROCESSED_DIR / "extra_cifra_negra_anual.csv", index=False)
    cifra_bienal.to_csv(PROCESSED_DIR / "extra_cifra_negra_bienal.csv", index=False)
    graves.to_csv(PROCESSED_DIR / "extra_delitos_graves_tendencia.csv", index=False)
    graves_municipio.to_csv(PROCESSED_DIR / "extra_delitos_graves_tendencia_municipio.csv", index=False)
    tasas.drop(columns="geometry").to_csv(PROCESSED_DIR / "extra_tasas_municipales_100k.csv", index=False)

    print("Tablas exportadas en:", PROCESSED_DIR.resolve())


if __name__ == "__main__":
    exportar_tablas()