from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from delitos_region import (
    preparar_base_delitos,
    tabla_relacion_tipo_region_tiempo,
    top_crecimiento_reduccion,
    concentracion_amg_ultimo_anio,
    concentracion_amg_por_municipio,
    analizar_cifra_negra,
    delitos_graves_tendencia,
    delitos_graves_tendencia_municipio,
    tasa_municipal_100k,
)


# =========================
# RUTAS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# ESTILO GENERAL
# =========================
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.dpi"] = 300


# =========================
# HELPERS
# =========================
def guardar_fig(nombre: str) -> None:
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / nombre, bbox_inches="tight")
    plt.close()


def cargar_csv_o_generar(nombre_csv: str, generador):
    ruta = PROCESSED_DIR / nombre_csv
    if ruta.exists():
        return pd.read_csv(ruta)
    return generador()


def anotar_barras_horizontal(ax, fmt="{:,.0f}", offset=0.01) -> None:
    xmax = ax.get_xlim()[1]
    for p in ax.patches:
        valor = p.get_width()
        y = p.get_y() + p.get_height() / 2
        ax.text(
            valor + xmax * offset,
            y,
            fmt.format(valor),
            va="center",
            ha="left",
            fontsize=10,
        )


def anotar_barras_vertical(ax, fmt="{:.1f}%", offset=0.8) -> None:
    for p in ax.patches:
        valor = p.get_height()
        x = p.get_x() + p.get_width() / 2
        ax.text(
            x,
            valor + offset,
            fmt.format(valor),
            va="bottom",
            ha="center",
            fontsize=10,
        )


# =========================
# PREGUNTA 1
# ¿Existe relación entre tipo de delito y región geográfica en el tiempo?
# =========================
def plot_heatmap_tipo_region_ultimo_anio() -> None:
    """
    Dataset base:
    data/processed/q1_tipo_region_tiempo.csv
    """
    tabla = cargar_csv_o_generar(
        "q1_tipo_region_tiempo.csv",
        lambda: tabla_relacion_tipo_region_tiempo(preparar_base_delitos()[0]),
    )

    ultimo_anio = int(tabla["Año"].max())
    tabla_ua = tabla.loc[tabla["Año"] == ultimo_anio].copy()

    top_tipos = (
        tabla_ua.groupby("Tipo de delito")["Total_Anual"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .index
    )

    tabla_ua = tabla_ua[tabla_ua["Tipo de delito"].isin(top_tipos)]

    pivot = (
        tabla_ua.pivot_table(
            index="Tipo de delito",
            columns="zona_geografica",
            values="Total_Anual",
            aggfunc="sum",
            fill_value=0,
        )
        .sort_values(by=list(tabla_ua["zona_geografica"].dropna().unique()), ascending=False)
    )

    plt.figure(figsize=(11, 8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"label": "Delitos reportados"},
    )
    plt.title(
        f"Pregunta 1. Relación entre tipo de delito y región geográfica ({ultimo_anio})\n"
        "Top 15 tipos de delito con más registros",
        fontsize=16,
    )
    plt.xlabel("Zona geográfica")
    plt.ylabel("Tipo de delito")
    guardar_fig("q1_heatmap_tipo_region_ultimo_anio.png")


def plot_lineas_top_tipos_amg_interior() -> None:
    """
    Dataset base:
    data/processed/q1_tipo_region_tiempo.csv
    """
    tabla = cargar_csv_o_generar(
        "q1_tipo_region_tiempo.csv",
        lambda: tabla_relacion_tipo_region_tiempo(preparar_base_delitos()[0]),
    )

    top_tipos = (
        tabla.groupby("Tipo de delito")["Total_Anual"]
        .sum()
        .sort_values(ascending=False)
        .head(6)
        .index
    )

    tabla = tabla[tabla["Tipo de delito"].isin(top_tipos)].copy()

    g = sns.relplot(
        data=tabla,
        x="Año",
        y="Total_Anual",
        hue="zona_geografica",
        col="Tipo de delito",
        kind="line",
        marker="o",
        palette={"AMG": "#C0392B", "Interior": "#2874A6"},
        col_wrap=2,
        height=4.2,
        aspect=1.35,
        facet_kws={"sharey": False},
    )
    g.set_axis_labels("Año", "Total anual de delitos")
    g.fig.subplots_adjust(top=0.90)
    g.fig.suptitle(
        "Pregunta 1. Evolución temporal por tipo de delito y región geográfica\n"
        "Comparación AMG vs Interior",
        fontsize=16,
    )
    g.savefig(OUTPUTS_DIR / "q1_lineas_tipo_delito_amg_interior.png", bbox_inches="tight")
    plt.close("all")


def plot_chi_cuadrada_pvalues() -> None:
    """
    Dataset base:
    data/processed/q1_chi_cuadrada.csv
    """
    ruta = PROCESSED_DIR / "q1_chi_cuadrada.csv"
    if not ruta.exists():
        return

    chi = pd.read_csv(ruta)
    if chi.empty:
        return

    chi["significativo_5pct"] = chi["p_value"] < 0.05

    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=chi,
        x="Año",
        y="p_value",
        marker="o",
        color="#7D3C98",
        linewidth=2.5,
    )
    plt.axhline(0.05, color="red", linestyle="--", linewidth=1.8, label="Umbral 5%")
    plt.title(
        "Pregunta 1. Evidencia estadística de relación entre tipo de delito y región",
        fontsize=16,
    )
    plt.xlabel("Año")
    plt.ylabel("p-value prueba chi-cuadrada")
    plt.legend()
    guardar_fig("q1_pvalues_chi_cuadrada.png")


# =========================
# PREGUNTA 2
# ¿Qué municipios han mostrado mayor crecimiento o reducción en delitos?
# =========================
def plot_top_crecimiento_reduccion() -> None:
    """
    Datasets base:
    data/processed/q2_top_crecimiento.csv
    data/processed/q2_top_reduccion.csv
    """
    delitos, _ = preparar_base_delitos()

    top_crecen = cargar_csv_o_generar(
        "q2_top_crecimiento.csv",
        lambda: top_crecimiento_reduccion(delitos, 2015, 2025, top_n=12)[0],
    )
    top_reducen = cargar_csv_o_generar(
        "q2_top_reduccion.csv",
        lambda: top_crecimiento_reduccion(delitos, 2015, 2025, top_n=12)[1],
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    sns.barplot(
        data=top_crecen.sort_values("cambio_absoluto", ascending=True),
        x="cambio_absoluto",
        y="Municipio",
        hue="Municipio",
        dodge=False,
        legend=False,
        ax=axes[0],
        palette="Reds_r",
    )
    
    axes[0].set_title(
        "Pregunta 2. Municipios con mayor crecimiento absoluto de delitos (2015 vs 2025)",
        fontsize=15,
    )
    axes[0].set_xlabel("Cambio absoluto en delitos")
    axes[0].set_ylabel("Municipio")
    anotar_barras_horizontal(axes[0], fmt="{:,.0f}")

    sns.barplot(
        data=top_reducen.sort_values("cambio_absoluto", ascending=False),
        x="cambio_absoluto",
        y="Municipio",
        hue="Municipio",
        dodge=False,
        legend=False,
        ax=axes[1],
        palette="Blues_r",
    )
    axes[1].set_title(
        "Pregunta 2. Municipios con mayor reducción absoluta de delitos (2015 vs 2025)",
        fontsize=15,
    )
    axes[1].set_xlabel("Cambio absoluto en delitos")
    axes[1].set_ylabel("Municipio")
    anotar_barras_horizontal(axes[1], fmt="{:,.0f}")

    guardar_fig("q2_top_crecimiento_reduccion_municipal.png")


def plot_scatter_crecimiento_municipal() -> None:
    """
    Dataset base:
    data/processed/q2_crecimiento_municipal.csv
    """
    delitos, _ = preparar_base_delitos()

    crecimiento = cargar_csv_o_generar(
        "q2_crecimiento_municipal.csv",
        lambda: __import__("delitos_region").crecimiento_municipal(delitos),
    )

    base = crecimiento.copy()
    base["tipo_cambio"] = "Crecimiento"
    base.loc[base["cambio_absoluto"] < 0, "tipo_cambio"] = "Reducción"

    plt.figure(figsize=(12, 7))
    sns.scatterplot(
        data=base,
        x="cambio_porcentual",
        y="cambio_absoluto",
        hue="tipo_cambio",
        palette={"Crecimiento": "#C0392B", "Reducción": "#2874A6"},
        s=90,
        alpha=0.8,
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.title(
        "Pregunta 2. Cambio absoluto vs cambio porcentual por municipio",
        fontsize=16,
    )
    plt.xlabel("Cambio porcentual (%)")
    plt.ylabel("Cambio absoluto en delitos")
    guardar_fig("q2_scatter_crecimiento_municipal.png")


# =========================
# PREGUNTA 3
# ¿En el año pasado se concentraron en municipios de la AMG?
# =========================
def plot_concentracion_amg_ultimo_anio() -> None:
    """
    Dataset base:
    data/processed/q3_concentracion_amg.csv
    """
    delitos, _ = preparar_base_delitos()

    tabla = cargar_csv_o_generar(
        "q3_concentracion_amg.csv",
        lambda: concentracion_amg_ultimo_anio(delitos),
    )

    anio = int(tabla["Año"].iloc[0])

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(
        data=tabla,
        x="zona_geografica",
        y="participacion_pct",
        hue="zona_geografica",
        dodge=False,
        legend=False,
        palette={"AMG": "#C0392B", "Interior": "#2874A6"},
    )
    plt.title(
        f"Pregunta 3. Concentración de delitos en AMG vs Interior ({anio})",
        fontsize=16,
    )
    plt.xlabel("Zona geográfica")
    plt.ylabel("Participación del total (%)")
    anotar_barras_vertical(ax, fmt="{:.1f}%")
    guardar_fig("q3_concentracion_amg_ultimo_anio.png")


def plot_top_municipios_ultimo_anio() -> None:
    """
    Dataset base:
    data/processed/q3_concentracion_amg_municipios.csv
    """
    delitos, _ = preparar_base_delitos()

    tabla = cargar_csv_o_generar(
        "q3_concentracion_amg_municipios.csv",
        lambda: concentracion_amg_por_municipio(delitos),
    )

    top_total = (
        tabla.groupby("Municipio", as_index=False)["Total_Anual"]
        .sum()
        .sort_values("Total_Anual", ascending=False)
        .head(15)
    )

    plt.figure(figsize=(13, 8))
    ax = sns.barplot(
        data=top_total.sort_values("Total_Anual", ascending=True),
        x="Total_Anual",
        y="Municipio",
        hue="Municipio",
        dodge=False,
        legend=False,
        palette="magma",
    )
    plt.title(
        "Pregunta 3. Municipios con más delitos reportados en el último año",
        fontsize=16,
    )
    plt.xlabel("Total anual de delitos")
    plt.ylabel("Municipio")
    anotar_barras_horizontal(ax, fmt="{:,.0f}")
    guardar_fig("q3_top_municipios_ultimo_anio.png")


def plot_top_municipios_amg_vs_interior() -> None:
    """
    Dataset base:
    data/processed/q3_concentracion_amg_municipios.csv
    """
    delitos, _ = preparar_base_delitos()

    tabla = cargar_csv_o_generar(
        "q3_concentracion_amg_municipios.csv",
        lambda: concentracion_amg_por_municipio(delitos),
    )

    top_por_zona = (
        tabla.sort_values(["zona_geografica", "Total_Anual"], ascending=[True, False])
        .groupby("zona_geografica", group_keys=False)
        .head(8)
        .copy()
    )

    g = sns.catplot(
        data=top_por_zona,
        kind="bar",
        x="Total_Anual",
        y="Municipio",
        col="zona_geografica",
        hue="zona_geografica",
        palette={"AMG": "#C0392B", "Interior": "#2874A6"},
        sharex=False,
        sharey=False,
        height=6,
        aspect=1.1,
        legend=False,
    )

    g.set_axis_labels("Total anual de delitos", "Municipio")

    for ax in g.axes.flat:
        for container in ax.containers:
            ax.bar_label(container, fmt="%.0f", padding=3, fontsize=9)

    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle(
        "Pregunta 3. Municipios líderes por número de delitos en el último año\n"
        "Comparación AMG vs Interior",
        fontsize=16,
    )
    g.savefig(OUTPUTS_DIR / "q3_top_municipios_amg_vs_interior.png", bbox_inches="tight")
    plt.close("all")


# =========================
# EXTRAS IMPORTANTES
# =========================
def plot_cifra_negra() -> None:
    """
    Datasets base:
    data/processed/extra_cifra_negra_anual.csv
    data/processed/extra_cifra_negra_bienal.csv
    """
    cifra_anual = cargar_csv_o_generar(
        "extra_cifra_negra_anual.csv",
        lambda: analizar_cifra_negra()[0],
    )

    if not cifra_anual.empty and "Año" in cifra_anual.columns and "Valor" in cifra_anual.columns:
        plt.figure(figsize=(11, 5))
        sns.lineplot(
            data=cifra_anual,
            x="Año",
            y="Valor",
            marker="o",
            color="#AF601A",
            linewidth=2.5,
        )
        plt.title(
            "Extra. Cifra negra anual en Jalisco\n"
            "Proporción estimada de delitos no denunciados o no registrados",
            fontsize=16,
        )
        plt.xlabel("Año")
        plt.ylabel("Porcentaje")
        guardar_fig("extra_cifra_negra_anual.png")

    cifra_bienal = cargar_csv_o_generar(
        "extra_cifra_negra_bienal.csv",
        lambda: analizar_cifra_negra()[1],
    )

    if not cifra_bienal.empty and "Año" in cifra_bienal.columns and "Valor" in cifra_bienal.columns:
        plt.figure(figsize=(11, 5))
        sns.lineplot(
            data=cifra_bienal,
            x="Año",
            y="Valor",
            marker="o",
            color="#7D6608",
            linewidth=2.5,
        )
        plt.title(
            "Extra. Cifra negra bienal en unidades económicas",
            fontsize=16,
        )
        plt.xlabel("Año")
        plt.ylabel("Porcentaje")
        guardar_fig("extra_cifra_negra_bienal.png")


def plot_delitos_graves_tendencia() -> None:
    """
    Dataset base:
    data/processed/extra_delitos_graves_tendencia.csv
    """
    delitos, _ = preparar_base_delitos()

    tabla = cargar_csv_o_generar(
        "extra_delitos_graves_tendencia.csv",
        lambda: delitos_graves_tendencia(delitos),
    )

    if tabla.empty:
        return

    plt.figure(figsize=(13, 7))
    sns.lineplot(
        data=tabla,
        x="Año",
        y="Total_Anual",
        hue="categoria_delito",
        marker="o",
        linewidth=2.3,
        palette="tab10",
    )
    plt.title(
        "Extra. Tendencia de categorías de delitos graves en Jalisco",
        fontsize=16,
    )
    plt.xlabel("Año")
    plt.ylabel("Total anual")
    plt.legend(title="Categoría", bbox_to_anchor=(1.02, 1), loc="upper left")
    guardar_fig("extra_delitos_graves_tendencia.png")


def plot_delitos_graves_municipio() -> None:
    """
    Dataset base:
    data/processed/extra_delitos_graves_tendencia_municipio.csv
    """
    delitos, _ = preparar_base_delitos()

    tabla = cargar_csv_o_generar(
        "extra_delitos_graves_tendencia_municipio.csv",
        lambda: delitos_graves_tendencia_municipio(delitos),
    )

    if tabla.empty:
        return

    ultimo_anio = int(tabla["Año"].max())
    base = tabla.loc[tabla["Año"] == ultimo_anio].copy()

    top = (
        base.groupby(["Municipio", "categoria_delito"], as_index=False)["Total_Anual"]
        .sum()
    )

    top = (
        top.sort_values("Total_Anual", ascending=False)
        .head(20)
        .copy()
    )

    plt.figure(figsize=(14, 9))
    sns.barplot(
        data=top.sort_values("Total_Anual", ascending=True),
        x="Total_Anual",
        y="Municipio",
        hue="categoria_delito",
        palette="tab20",
    )
    plt.title(
        f"Extra. Principales municipios y categorías de delitos graves ({ultimo_anio})",
        fontsize=16,
    )
    plt.xlabel("Total anual")
    plt.ylabel("Municipio")
    plt.legend(title="Categoría", bbox_to_anchor=(1.02, 1), loc="upper left")
    guardar_fig("extra_delitos_graves_municipio_ultimo_anio.png")


def plot_mapa_tasa_municipal() -> None:
    """
    Dataset base:
    data/processed/extra_tasas_municipales_100k.csv
    Para el mapa se recalcula desde el geodataframe.
    """
    _, delitos_geo = preparar_base_delitos()
    ultimo_anio = int(delitos_geo["Año"].max())
    mapa = tasa_municipal_100k(delitos_geo, ultimo_anio)

    fig, ax = plt.subplots(figsize=(12, 10))
    mapa.plot(
        column="tasa_100k",
        cmap="OrRd",
        linewidth=0.35,
        edgecolor="black",
        legend=True,
        ax=ax,
        missing_kwds={"color": "lightgrey", "label": "Sin datos"},
    )
    ax.set_title(
        f"Extra. Tasa de delitos por 100 mil habitantes por municipio ({ultimo_anio})",
        fontsize=16,
    )
    ax.axis("off")
    guardar_fig("extra_mapa_tasa_delitos_100k.png")


# =========================
# ORQUESTADOR
# =========================
def generar_todos_los_plots() -> None:
    # Pregunta 1
    plot_heatmap_tipo_region_ultimo_anio()
    plot_lineas_top_tipos_amg_interior()
    plot_chi_cuadrada_pvalues()

    # Pregunta 2
    plot_top_crecimiento_reduccion()
    plot_scatter_crecimiento_municipal()

    # Pregunta 3
    plot_concentracion_amg_ultimo_anio()
    plot_top_municipios_ultimo_anio()
    plot_top_municipios_amg_vs_interior()

    # Extras
    plot_cifra_negra()
    plot_delitos_graves_tendencia()
    plot_delitos_graves_municipio()
    plot_mapa_tasa_municipal()

    print("Gráficos guardados en:", OUTPUTS_DIR.resolve())


if __name__ == "__main__":
    generar_todos_los_plots()