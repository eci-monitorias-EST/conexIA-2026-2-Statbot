"""Motor de recuperación (RAG) basado en TF-IDF sobre un corpus de teoría estadística."""

import re
import unicodedata
from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SPANISH_STOPWORDS = [
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con",
    "contra", "cual", "cuales", "cuando", "de", "del", "desde", "donde",
    "durante", "e", "el", "ella", "ellas", "ellos", "en", "entre", "era",
    "es", "está", "están", "esté",
    "esa", "esas", "ese", "eso", "esos", "esta", "estas", "este", "esto",
    "estos", "ha", "hay", "la", "las", "le", "les", "lo", "los", "mas",
    "más", "me", "mi", "mis", "mucho", "muy", "ni", "no", "nos", "nosotros",
    "o", "os", "otra", "otras", "otro", "otros", "para", "pero", "poco",
    "por", "porque", "que", "qué", "quien", "quienes", "se", "sea", "según",
    "ser", "si", "sí", "sin", "sobre", "son", "su", "sus", "también",
    "tanto", "te", "ti", "todo", "todos", "tu", "tus", "un", "una", "uno",
    "unos", "vosotros", "y", "ya", "yo",
]

textos_stats = [
    """
La estadística descriptiva se encarga de recolectar, organizar, resumir y presentar los datos.
Las medidas más importantes son la media, la varianza y la desviación estándar.
""",
    """
El valor p es la probabilidad de observar resultados tan extremos como los observados asumiendo verdadera la hipótesis nula.
Si p es menor que alfa se rechaza la hipótesis nula.
""",
    """
Las pruebas paramétricas suponen una distribución específica para los datos, por ejemplo normalidad.
Las pruebas no paramétricas no requieren este supuesto y se basan en rangos u órdenes.
""",
    """
En regresión lineal se modela la relación entre una variable respuesta y una o varias variables explicativas
mediante una ecuación lineal. El coeficiente de determinación R cuadrado mide la proporción de variabilidad explicada.
""",
    """
La prueba chi-cuadrado permite evaluar independencia entre variables categóricas usando tablas de contingencia.
""",
]

textos_stats_extra = [
    """
Una variable aleatoria es discreta si toma valores numerables y continua si puede tomar cualquier valor en un intervalo.
""",
    """
Un intervalo de confianza para la media poblacional es un rango de valores que, con cierto nivel de confianza, contiene al parámetro.
""",
    """
R cuadrado cercano a 1 indica que el modelo de regresión explica bien la variabilidad de la respuesta.
""",
    """
Las pruebas no paramétricas como Wilcoxon, Mann-Whitney o Kruskal-Wallis se utilizan cuando no se cumplen supuestos de normalidad
o cuando los datos están en escala ordinal.
""",
]

corpus_docs = [t.strip() for t in textos_stats + textos_stats_extra]

vectorizer = TfidfVectorizer(stop_words=SPANISH_STOPWORDS)
tfidf_matrix = vectorizer.fit_transform(corpus_docs)

# Similitud mínima para considerar un documento relevante. Por debajo de este
# umbral, el documento con mayor score suele ser ruido (p.ej. una pregunta
# sobre un término que no está en el corpus) y no debe mostrarse como si
# fuera contexto pertinente.
MIN_SIMILARITY = 0.1


def retrieve_context(question: str, k: int = 3) -> List[str]:
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]
    idx = np.argsort(sims)[::-1][:k]
    return [corpus_docs[i] for i in idx if sims[i] >= MIN_SIMILARITY]


def build_context_text(question: str, k: int = 3) -> str:
    ctx_list = retrieve_context(question, k=k)
    if not ctx_list:
        return (
            "No encontré teoría específica sobre ese tema en la base de conocimiento. "
            "Intenta reformular la pregunta o consulta otro concepto estadístico."
        )
    return "\n".join("- " + c for c in ctx_list)


def _normalizar(texto: str) -> str:
    """Minúsculas y sin acentos, para que 'mediana' y 'medianas' o
    'desviación'/'desviacion' se traten igual al buscar coincidencias."""
    texto = texto.lower().replace("-", " ")
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


# Diccionario de respaldo: respuestas directas para términos clave de teoría
# estadística. Se consulta ANTES de depender de la similitud TF-IDF, que con
# un corpus pequeño puede fallar (p.ej. si el término ni siquiera aparece en
# el corpus, como pasaba con "mediana" y "moda") o confundirse con preguntas
# cortas donde predominan palabras genéricas.
DEFINICIONES: Dict[str, str] = {
    "media": (
        "La **media** (o promedio) es la suma de todos los valores de un conjunto de "
        "datos dividida entre el número de observaciones. Es la medida de tendencia "
        "central más usada, pero es sensible a valores atípicos (outliers)."
    ),
    # Alias porque en lenguaje coloquial "promedio" es el término más usado
    # para referirse a la media, y muchas preguntas no dicen "media" nunca.
    "promedio": (
        "La **media** (o promedio) es la suma de todos los valores de un conjunto de "
        "datos dividida entre el número de observaciones. Es la medida de tendencia "
        "central más usada, pero es sensible a valores atípicos (outliers)."
    ),
    "mediana": (
        "La **mediana** es el valor que ocupa la posición central de un conjunto de "
        "datos ordenados de menor a mayor. Si el número de datos es par, la mediana "
        "es el promedio de los dos valores centrales. A diferencia de la media, no se "
        "ve afectada por valores atípicos."
    ),
    "punto medio": (
        "La **mediana** es el valor que ocupa la posición central de un conjunto de "
        "datos ordenados de menor a mayor. Si el número de datos es par, la mediana "
        "es el promedio de los dos valores centrales. A diferencia de la media, no se "
        "ve afectada por valores atípicos."
    ),
    "moda": (
        "La **moda** es el valor (o valores) que se repite con mayor frecuencia en un "
        "conjunto de datos. Un conjunto puede no tener moda, tener una sola (unimodal) "
        "o varias (multimodal)."
    ),
    "valor mas repetido": (
        "La **moda** es el valor (o valores) que se repite con mayor frecuencia en un "
        "conjunto de datos. Un conjunto puede no tener moda, tener una sola (unimodal) "
        "o varias (multimodal)."
    ),
    "valor mas frecuente": (
        "La **moda** es el valor (o valores) que se repite con mayor frecuencia en un "
        "conjunto de datos. Un conjunto puede no tener moda, tener una sola (unimodal) "
        "o varias (multimodal)."
    ),
    "varianza": (
        "La **varianza** mide la dispersión de los datos respecto a la media. Se "
        "calcula como el promedio de las diferencias al cuadrado entre cada dato y la "
        "media, y se expresa en las unidades originales elevadas al cuadrado."
    ),
    "desviacion estandar": (
        "La **desviación estándar** es la raíz cuadrada de la varianza. Mide la "
        "dispersión de los datos en las mismas unidades que la variable original, por "
        "lo que es más fácil de interpretar que la varianza."
    ),
    "rango": (
        "El **rango** es la diferencia entre el valor máximo y el valor mínimo de un "
        "conjunto de datos. Es la medida de dispersión más simple, pero muy sensible a "
        "valores atípicos."
    ),
    "coeficiente de variacion": (
        "El **coeficiente de variación** es el cociente entre la desviación estándar y "
        "la media (a veces expresado en porcentaje). Permite comparar la dispersión "
        "relativa entre conjuntos de datos con escalas distintas."
    ),
    "valor p": (
        "El **valor p** es la probabilidad de observar resultados tan extremos como "
        "los observados, asumiendo que la hipótesis nula es verdadera. Si p es menor "
        "que el nivel de significancia (alfa, usualmente 0.05), se rechaza la "
        "hipótesis nula."
    ),
    # Alias porque "p-valor" (orden invertido a "valor p") normaliza a "p valor".
    "p valor": (
        "El **valor p** es la probabilidad de observar resultados tan extremos como "
        "los observados, asumiendo que la hipótesis nula es verdadera. Si p es menor "
        "que el nivel de significancia (alfa, usualmente 0.05), se rechaza la "
        "hipótesis nula."
    ),
    "hipotesis nula": (
        "La **hipótesis nula (H0)** es el supuesto inicial que se pone a prueba en un "
        "contraste de hipótesis; típicamente representa la ausencia de efecto o "
        "diferencia. Se rechaza o no se rechaza según la evidencia de los datos."
    ),
    "hipotesis alterna": (
        "La **hipótesis alterna (H1)** es la afirmación que se acepta si los datos "
        "proveen suficiente evidencia en contra de la hipótesis nula; representa el "
        "efecto o diferencia que se busca detectar."
    ),
    "intervalo de confianza": (
        "Un **intervalo de confianza** para un parámetro poblacional (por ejemplo la "
        "media) es un rango de valores que, con cierto nivel de confianza (p.ej. 95%), "
        "se espera que contenga al verdadero valor del parámetro."
    ),
    "margen de error": (
        "Un **intervalo de confianza** para un parámetro poblacional (por ejemplo la "
        "media) es un rango de valores que, con cierto nivel de confianza (p.ej. 95%), "
        "se espera que contenga al verdadero valor del parámetro."
    ),
    "variable aleatoria": (
        "Una **variable aleatoria** es discreta si toma valores numerables (por "
        "ejemplo, conteos) y continua si puede tomar cualquier valor dentro de un "
        "intervalo (por ejemplo, mediciones)."
    ),
    "regresion lineal": (
        "La **regresión lineal** modela la relación entre una variable respuesta y una "
        "o varias variables explicativas mediante una ecuación lineal, estimando cómo "
        "cambia la respuesta en función de los predictores."
    ),
    "r cuadrado": (
        "El **coeficiente de determinación (R²)** mide la proporción de la "
        "variabilidad de la variable respuesta que es explicada por el modelo de "
        "regresión. Un valor cercano a 1 indica un buen ajuste."
    ),
    "chi cuadrado": (
        "La **prueba chi-cuadrado** se usa, entre otros casos, para evaluar la "
        "independencia entre variables categóricas (mediante tablas de contingencia) o "
        "la bondad de ajuste de frecuencias observadas frente a esperadas."
    ),
    "prueba parametrica": (
        "Las **pruebas paramétricas** suponen una distribución específica para los "
        "datos (por ejemplo, normalidad) y suelen basarse en parámetros como la media "
        "y la varianza."
    ),
    "prueba no parametrica": (
        "Las **pruebas no paramétricas** (como Wilcoxon, Mann-Whitney o "
        "Kruskal-Wallis) no requieren supuestos sobre la distribución de los datos y se "
        "basan en rangos u órdenes; se usan cuando no se cumple el supuesto de "
        "normalidad o los datos son ordinales."
    ),
    "distribucion normal": (
        "La **distribución normal** (o gaussiana) es una distribución de probabilidad "
        "continua, simétrica y con forma de campana, caracterizada por su media y su "
        "desviación estándar. Muchos métodos estadísticos clásicos asumen normalidad "
        "en los datos."
    ),
    "campana de gauss": (
        "La **distribución normal** (o gaussiana) es una distribución de probabilidad "
        "continua, simétrica y con forma de campana, caracterizada por su media y su "
        "desviación estándar. Muchos métodos estadísticos clásicos asumen normalidad "
        "en los datos."
    ),
    "curva de campana": (
        "La **distribución normal** (o gaussiana) es una distribución de probabilidad "
        "continua, simétrica y con forma de campana, caracterizada por su media y su "
        "desviación estándar. Muchos métodos estadísticos clásicos asumen normalidad "
        "en los datos."
    ),
}

# Orden por longitud descendente para priorizar coincidencias de frases
# multi-palabra (p.ej. "desviación estándar") sobre términos más cortos.
_DEF_KEYS_ORDENADAS = sorted(DEFINICIONES.keys(), key=len, reverse=True)


def buscar_definicion(question: str) -> Optional[str]:
    """Busca coincidencias claras de términos clave dentro de la pregunta.

    Usa límites de palabra (\\b) para evitar falsos positivos como que
    "media" empate dentro de "mediana". Si la pregunta menciona varios
    términos (p.ej. "¿qué es la mediana y el valor p?"), devuelve las
    definiciones de todos ellos, en el orden en que aparecen en la pregunta.
    Devuelve None si ningún término aplica (en cuyo caso se debe recurrir al
    retrieval por TF-IDF)."""
    q_norm = _normalizar(question)
    coincidencias = []
    for key in _DEF_KEYS_ORDENADAS:
        patron = r"\b" + re.escape(key).replace(r"\ ", r"\s+") + r"\b"
        m = re.search(patron, q_norm)
        if m:
            coincidencias.append((m.start(), DEFINICIONES[key]))
    if not coincidencias:
        return None
    coincidencias.sort(key=lambda par: par[0])
    return "\n\n".join(definicion for _, definicion in coincidencias)
