"""Agente basado en Gemini (function calling) para responder preguntas en lenguaje
natural sobre el dataset de partidos del Mundial 2026 (worldcup-full.json).

Usa la API gratuita de Google AI Studio en vez de un proveedor de pago. A
diferencia de agent.py (reglas fijas + TF-IDF), aquí el modelo decide, pregunta
a pregunta, cuál función de dataset.py conviene llamar para obtener el dato
exacto, en vez de intentar contar o inventar cifras por su cuenta.
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import errors, types

# El tier gratuito de Gemini a veces devuelve 503 (modelo saturado) o 429
# (limite de peticiones por minuto, relevante si varios estudiantes preguntan
# a la vez con la misma API key); reintentar con una pausa corta evita que eso
# se vea como un error en plena demostración.
MAX_REINTENTOS = 3
ESPERA_ENTRE_REINTENTOS_SEG = 3
ESPERA_RATE_LIMIT_SEG = 8

from dataset import (
    estadisticas_jugador,
    partidos_de_equipo,
    resumen_torneo,
    top_goleadores,
    top_tarjetas,
)

MODEL = "gemini-3.6-flash"

SYSTEM_PROMPT = (
    "Eres un asistente que responde preguntas sobre los partidos y jugadores del "
    "Mundial de Futbol 2026, usando exclusivamente las herramientas disponibles "
    "para consultar el dataset. No inventes cifras ni nombres: si una herramienta "
    "no devuelve informacion relevante, dilo explicitamente en vez de adivinar. "
    "Responde siempre en espanol, de forma breve y con los datos concretos que "
    "arrojen las herramientas."
)

_FUNCTION_DECLARATIONS = [
    types.FunctionDeclaration(
        name="top_goleadores",
        description="Ranking de jugadores con mas goles en el torneo, opcionalmente filtrado por seleccion.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "n": types.Schema(type=types.Type.INTEGER, description="Cantidad de jugadores a devolver (por defecto 10)"),
                "equipo": types.Schema(type=types.Type.STRING, description="Filtrar por seleccion (opcional)"),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="estadisticas_jugador",
        description="Goles, tarjetas y partidos disputados por un jugador especifico (busqueda parcial por nombre).",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={"nombre": types.Schema(type=types.Type.STRING, description="Nombre o parte del nombre del jugador")},
            required=["nombre"],
        ),
    ),
    types.FunctionDeclaration(
        name="partidos_de_equipo",
        description="Lista de partidos jugados por una seleccion, con resultado, sede y asistencia.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={"equipo": types.Schema(type=types.Type.STRING, description="Nombre de la seleccion")},
            required=["equipo"],
        ),
    ),
    types.FunctionDeclaration(
        name="resumen_torneo",
        description="Estadisticas generales del torneo: numero de partidos, goles totales, promedio de goles y asistencia.",
        parameters=types.Schema(type=types.Type.OBJECT, properties={}),
    ),
    types.FunctionDeclaration(
        name="top_tarjetas",
        description="Ranking de jugadores con mas tarjetas (amarillas y/o rojas) en el torneo.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "n": types.Schema(type=types.Type.INTEGER, description="Cantidad de jugadores a devolver (por defecto 10)"),
                "tipo": types.Schema(type=types.Type.STRING, enum=["roja", "amarilla"], description="Filtrar por tipo de tarjeta (opcional)"),
            },
        ),
    ),
]

_TOOL_FUNCS = {
    "top_goleadores": top_goleadores,
    "estadisticas_jugador": estadisticas_jugador,
    "partidos_de_equipo": partidos_de_equipo,
    "resumen_torneo": resumen_torneo,
    "top_tarjetas": top_tarjetas,
}


def get_default_api_key() -> Optional[str]:
    """Busca la API key en variables de entorno o en .streamlit/secrets.toml."""
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GOOGLE_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


def _ejecutar_tool(nombre: str, args: Dict[str, Any]) -> Any:
    return _TOOL_FUNCS[nombre](**args)


def _generar_con_reintentos(client: genai.Client, contents: List[types.Content], config: types.GenerateContentConfig):
    for intento in range(MAX_REINTENTOS):
        try:
            return client.models.generate_content(model=MODEL, contents=contents, config=config)
        except errors.ServerError:
            if intento == MAX_REINTENTOS - 1:
                raise
            time.sleep(ESPERA_ENTRE_REINTENTOS_SEG)
        except errors.ClientError as e:
            if e.code != 429 or intento == MAX_REINTENTOS - 1:
                raise
            time.sleep(ESPERA_RATE_LIMIT_SEG)


def responder_pregunta_dataset(
    pregunta: str,
    api_key: str,
    historial: Optional[List[types.Content]] = None,
) -> Tuple[str, List[types.Content]]:
    """Responde una pregunta sobre el dataset del Mundial 2026 usando Gemini con
    function calling. Devuelve (respuesta_en_texto, historial_actualizado) para
    que el caller pueda mantener el hilo de la conversación."""
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[types.Tool(function_declarations=_FUNCTION_DECLARATIONS)],
    )

    contents: List[types.Content] = list(historial or [])
    contents.append(types.Content(role="user", parts=[types.Part(text=pregunta)]))

    for _ in range(5):
        response = _generar_con_reintentos(client, contents, config)
        candidate = response.candidates[0]
        contents.append(candidate.content)

        function_calls = [p.function_call for p in candidate.content.parts if p.function_call]
        if not function_calls:
            texto = "".join(p.text for p in candidate.content.parts if p.text)
            return texto, contents

        response_parts = []
        for fc in function_calls:
            try:
                resultado = _ejecutar_tool(fc.name, dict(fc.args or {}))
                response_parts.append(
                    types.Part.from_function_response(name=fc.name, response={"resultado": resultado})
                )
            except Exception as e:
                response_parts.append(
                    types.Part.from_function_response(name=fc.name, response={"error": str(e)})
                )
        contents.append(types.Content(role="user", parts=response_parts))

    return (
        "No pude completar la respuesta tras varias consultas al dataset. Intenta reformular la pregunta.",
        contents,
    )
