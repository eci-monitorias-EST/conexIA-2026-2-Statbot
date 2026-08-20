"""Flujo LangGraph (RAG -> ANSWER) que decide entre ejecutar una prueba estadística o
responder únicamente con contexto teórico.

Corrige el bug del notebook original: `answer_node` ya no concatena el contexto
teórico del RAG cuando ya se calculó un resultado numérico; el contexto solo se
muestra cuando la pregunta es puramente teórica.
"""

from typing import Any, Dict, Optional, Tuple, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from formatting import format_chi2, format_resumen, format_t_test
from rag import build_context_text
from tools import (
    chi_cuadrado_bondad,
    extraer_datos_y_mu0,
    extraer_lista_numeros,
    extraer_listas_corchetes,
    resumen_descriptivo,
    t_test_una_muestra,
)


class StatsState(TypedDict):
    question: str
    context: str
    tool_type: Optional[str]
    tool_result: Optional[Dict[str, Any]]
    answer: str


def rag_node(state: StatsState) -> StatsState:
    q = state["question"]
    ctx = build_context_text(q, k=3)
    return {"question": q, "context": ctx, "tool_type": None, "tool_result": None, "answer": ""}


def _ejecutar_tool(q: str) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    """Detecta y ejecuta la prueba estadística implícita en la pregunta.

    Devuelve (tool_type, tool_result, texto_formateado), o (None, None, None)
    si la pregunta es teórica o no trae datos suficientes.
    """
    q_lower = q.lower()

    if "[" not in q or "]" not in q:
        return None, None, None

    if "chi" in q_lower:
        listas = extraer_listas_corchetes(q)
        if len(listas) >= 2:
            observados, esperados = listas[0], listas[1]
            res_chi = chi_cuadrado_bondad(observados, esperados)
            tool_result = {"chi2": res_chi, "observados": observados, "esperados": esperados}
            return "chi2", tool_result, format_chi2(res_chi)
        return None, None, None

    datos = extraer_lista_numeros(q)
    if len(datos) < 2:
        return None, None, None

    if "prueba t" in q_lower or "t de una muestra" in q_lower:
        datos_t, mu0 = extraer_datos_y_mu0(q)
        if datos_t is not None:
            res_t = t_test_una_muestra(datos_t, mu0)
            tool_result = {"t_test": res_t, "datos": datos_t, "mu0": mu0}
            return "t_test", tool_result, format_t_test(res_t)
        return None, None, None

    res_resumen = resumen_descriptivo(datos)
    tool_result = {"resumen": res_resumen, "datos": datos}
    return "resumen", tool_result, format_resumen(res_resumen)


def answer_node(state: StatsState) -> StatsState:
    q = state["question"]
    ctx = state["context"]

    tool_type, tool_result, texto_formateado = _ejecutar_tool(q)

    if tool_type is not None:
        answer = texto_formateado
    else:
        answer = ctx

    return {
        "question": q,
        "context": ctx,
        "tool_type": tool_type,
        "tool_result": tool_result,
        "answer": answer,
    }


builder = StateGraph(StatsState)
builder.add_node("RAG", rag_node)
builder.add_node("ANSWER", answer_node)
builder.add_edge(START, "RAG")
builder.add_edge("RAG", "ANSWER")
builder.add_edge("ANSWER", END)

memory = MemorySaver()
graph_stats = builder.compile(checkpointer=memory)


def answer_stats_langgraph(question: str, thread_id: str = "stats-demo") -> StatsState:
    """Ejecuta el grafo completo y devuelve el estado final (pregunta, contexto,
    tipo/valor de la tool ejecutada y respuesta final ya formateada)."""
    state: StatsState = {
        "question": question,
        "context": "",
        "tool_type": None,
        "tool_result": None,
        "answer": "",
    }
    config = {"configurable": {"thread_id": thread_id}}
    return graph_stats.invoke(state, config)
