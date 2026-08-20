"""Funciones de cálculo estadístico y extracción de datos numéricos desde texto."""

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import chisquare, ttest_1samp


def media(datos: List[float]) -> float:
    arr = np.array(datos, dtype=float)
    return float(arr.mean())


def varianza_muestral(datos: List[float]) -> float:
    arr = np.array(datos, dtype=float)
    return float(arr.var(ddof=1))


def resumen_descriptivo(datos: List[float]) -> Dict[str, Any]:
    arr = np.array(datos, dtype=float)
    media_val = float(arr.mean())
    sd = float(arr.std(ddof=1))
    return {
        "n": int(arr.size),
        "media": media_val,
        "mediana": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "rango": float(arr.max() - arr.min()),
        "varianza_muestral": float(arr.var(ddof=1)),
        "desviacion_estandar_muestral": sd,
        "coeficiente_variacion": float(sd / media_val) if media_val != 0 else None,
    }


def t_test_una_muestra(datos: List[float], mu0: float) -> Dict[str, Any]:
    arr = np.array(datos, dtype=float)
    t_stat, p_val = ttest_1samp(arr, mu0)
    return {
        "n": int(arr.size),
        "media_muestral": float(arr.mean()),
        "hipotesis_media": float(mu0),
        "t_stat": float(t_stat),
        "p_valor": float(p_val),
    }


def chi_cuadrado_bondad(observados: List[float], esperados: List[float]) -> Dict[str, Any]:
    obs = np.array(observados, dtype=float)
    exp = np.array(esperados, dtype=float)
    stat, p_val = chisquare(obs, f_exp=exp)
    return {
        "k": int(obs.size),
        "chi2": float(stat),
        "p_valor": float(p_val),
    }


def extraer_lista_numeros(texto: str) -> List[float]:
    numeros = re.findall(r"-?\d+\.?\d*", texto)
    return [float(x) for x in numeros]


def extraer_datos_y_mu0(texto: str) -> Tuple[Optional[List[float]], Optional[float]]:
    numeros = extraer_lista_numeros(texto)
    if len(numeros) < 2:
        return None, None
    datos = numeros[:-1]
    mu0 = numeros[-1]
    return datos, mu0


def extraer_listas_corchetes(texto: str) -> List[List[float]]:
    bloques = re.findall(r"\[([^\]]*)\]", texto)
    listas = []
    for b in bloques:
        nums = re.findall(r"-?\d+\.?\d*", b)
        if nums:
            listas.append([float(x) for x in nums])
    return listas
