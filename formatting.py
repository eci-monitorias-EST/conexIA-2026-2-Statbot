"""Formato de presentación (markdown) para los resultados de las pruebas estadísticas."""

from typing import Any, Dict

ALFA = 0.05


def format_resumen(res: Dict[str, Any]) -> str:
    cv = res["coeficiente_variacion"]
    cv_txt = f"{cv:.4f}" if cv is not None else "N/A"
    return (
        "**Resumen descriptivo**\n\n"
        "| Estadístico | Valor |\n|---|---|\n"
        f"| n | {res['n']} |\n"
        f"| Media | {res['media']:.4f} |\n"
        f"| Mediana | {res['mediana']:.4f} |\n"
        f"| Mínimo | {res['min']:.4f} |\n"
        f"| Máximo | {res['max']:.4f} |\n"
        f"| Rango | {res['rango']:.4f} |\n"
        f"| Varianza muestral | {res['varianza_muestral']:.4f} |\n"
        f"| Desv. estándar muestral | {res['desviacion_estandar_muestral']:.4f} |\n"
        f"| Coef. de variación | {cv_txt} |\n"
    )


def format_t_test(res: Dict[str, Any]) -> str:
    decision = "Se **rechaza H0**" if res["p_valor"] < ALFA else "**No se rechaza H0**"
    evidencia = "hay evidencia" if res["p_valor"] < ALFA else "no hay evidencia suficiente"
    return (
        "**Prueba t de una muestra**\n\n"
        "| Estadístico | Valor |\n|---|---|\n"
        f"| n | {res['n']} |\n"
        f"| Media muestral | {res['media_muestral']:.4f} |\n"
        f"| Media hipotética (H0) | {res['hipotesis_media']:.4f} |\n"
        f"| Estadístico t | {res['t_stat']:.4f} |\n"
        f"| Valor p | {res['p_valor']:.4f} |\n\n"
        f"**Conclusión (α = {ALFA}):** {decision} — {evidencia} para afirmar "
        "que la media difiere de la hipotética."
    )


def format_chi2(res: Dict[str, Any]) -> str:
    decision = "Se **rechaza H0**" if res["p_valor"] < ALFA else "**No se rechaza H0**"
    diferencia = "difieren" if res["p_valor"] < ALFA else "no difieren"
    return (
        "**Prueba chi-cuadrado de bondad de ajuste**\n\n"
        "| Estadístico | Valor |\n|---|---|\n"
        f"| Categorías (k) | {res['k']} |\n"
        f"| Chi-cuadrado | {res['chi2']:.4f} |\n"
        f"| Valor p | {res['p_valor']:.4f} |\n\n"
        f"**Conclusión (α = {ALFA}):** {decision} — las frecuencias observadas {diferencia} "
        "significativamente de las esperadas."
    )
