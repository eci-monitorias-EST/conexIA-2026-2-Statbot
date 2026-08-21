"""Gráficas (Plotly) asociadas a cada prueba estadística, en la paleta roja/negra/blanca de STATBOT."""

from typing import List

import numpy as np
import plotly.graph_objects as go
from scipy.stats import t as t_dist

COLOR_ROJO = "#C8102E"
COLOR_NEGRO = "#111111"
COLOR_ROJO_TRANSPARENTE = "rgba(200, 16, 46, 0.35)"


MUESTRA_PEQUENA_MAX = 10


def plot_histograma(datos: List[float]) -> go.Figure:
    arr = np.array(datos, dtype=float)
    media_val = float(arr.mean())

    if arr.size <= MUESTRA_PEQUENA_MAX:
        return _plot_puntos(arr, media_val)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=arr,
            marker=dict(color=COLOR_ROJO, line=dict(color=COLOR_NEGRO, width=1)),
            name="Datos",
        )
    )
    fig.add_vline(
        x=media_val,
        line=dict(color=COLOR_NEGRO, width=2, dash="dash"),
        annotation_text=f"Media = {media_val:.2f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Distribución de los datos",
        xaxis_title="Valor",
        yaxis_title="Frecuencia",
        template="plotly_white",
        showlegend=False,
        bargap=0.15,
        margin=dict(t=50, l=10, r=10, b=10),
    )
    return fig


def _plot_puntos(arr: np.ndarray, media_val: float) -> go.Figure:
    """Dot plot para muestras pequeñas (n <= 10): cada valor se muestra como un
    punto individual; los valores repetidos se apilan verticalmente en vez de
    agruparse en barras pegadas, que con pocos datos se ven como un bloque
    sólido en lugar de un histograma legible."""
    valores_unicos, conteos = np.unique(arr, return_counts=True)
    xs, ys = [], []
    for valor, conteo in zip(valores_unicos, conteos):
        xs.extend([valor] * conteo)
        ys.extend(range(1, conteo + 1))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(color=COLOR_ROJO, size=16, line=dict(color=COLOR_NEGRO, width=1.5)),
            name="Datos",
        )
    )
    fig.add_vline(
        x=media_val,
        line=dict(color=COLOR_NEGRO, width=2, dash="dash"),
        annotation_text=f"Media = {media_val:.2f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Valores individuales (muestra pequeña)",
        xaxis_title="Valor",
        yaxis_title="Repeticiones",
        template="plotly_white",
        showlegend=False,
        margin=dict(t=50, l=10, r=10, b=10),
    )
    fig.update_yaxes(dtick=1, range=[0, max(conteos) + 1])
    return fig


def plot_t_distribution(t_stat: float, df: int, alfa: float = 0.05) -> go.Figure:
    lim = max(4.5, abs(t_stat) + 1.5)
    x = np.linspace(-lim, lim, 400)
    y = t_dist.pdf(x, df)
    t_crit = t_dist.ppf(1 - alfa / 2, df)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="lines", line=dict(color=COLOR_NEGRO, width=2), name="t de Student")
    )

    x_izq = x[x <= -t_crit]
    x_der = x[x >= t_crit]
    fig.add_trace(
        go.Scatter(
            x=x_izq,
            y=t_dist.pdf(x_izq, df),
            fill="tozeroy",
            line=dict(color=COLOR_ROJO),
            fillcolor=COLOR_ROJO_TRANSPARENTE,
            name="Región de rechazo",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_der,
            y=t_dist.pdf(x_der, df),
            fill="tozeroy",
            line=dict(color=COLOR_ROJO),
            fillcolor=COLOR_ROJO_TRANSPARENTE,
            showlegend=False,
        )
    )
    fig.add_vline(
        x=t_stat,
        line=dict(color=COLOR_ROJO, width=3),
        annotation_text=f"t observado = {t_stat:.3f}",
        annotation_position="top",
    )
    fig.update_layout(
        title=f"Distribución t de Student (df = {df}, α = {alfa})",
        xaxis_title="t",
        yaxis_title="Densidad",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=70, l=10, r=10, b=10),
    )
    return fig


def plot_chi2_barras(observados: List[float], esperados: List[float]) -> go.Figure:
    categorias = [f"Cat. {i + 1}" for i in range(len(observados))]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=categorias, y=observados, name="Observados", marker_color=COLOR_ROJO))
    fig.add_trace(go.Bar(x=categorias, y=esperados, name="Esperados", marker_color=COLOR_NEGRO))
    fig.update_layout(
        title="Frecuencias observadas vs. esperadas",
        xaxis_title="Categoría",
        yaxis_title="Frecuencia",
        barmode="group",
        bargap=0.25,
        bargroupgap=0.1,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=70, l=10, r=10, b=10),
    )
    return fig
