"""Motor de recuperación (RAG) basado en TF-IDF sobre un corpus de teoría estadística."""

from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus_docs)


def retrieve_context(question: str, k: int = 3) -> List[str]:
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]
    idx = np.argsort(sims)[::-1][:k]
    return [corpus_docs[i] for i in idx]


def build_context_text(question: str, k: int = 3) -> str:
    ctx_list = retrieve_context(question, k=k)
    return "\n".join("- " + c for c in ctx_list)
