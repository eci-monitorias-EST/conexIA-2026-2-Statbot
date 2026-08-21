"""STATBOT — asistente de estadística (Streamlit).

Combina RAG (contexto teórico) con herramientas de cálculo estadístico a través
de un flujo LangGraph (ver agent.py), expuesto aquí en dos modos: chat en
lenguaje natural y formularios guiados por prueba.
"""

import os
import uuid

import streamlit as st

from agent import answer_stats_langgraph
from formatting import format_chi2, format_resumen, format_t_test
from plots import plot_chi2_barras, plot_histograma, plot_t_distribution
from tools import chi_cuadrado_bondad, resumen_descriptivo, t_test_una_muestra

LOGO_PATH = os.path.join("assets", "logo.png")

TOOL_LABELS = {
    "resumen": "🧮 Resumen descriptivo",
    "t_test": "🧮 Prueba t de una muestra",
    "chi2": "🧮 Prueba chi-cuadrado",
    "definicion": "📖 Definición",
    None: "📚 Contexto teórico",
}

st.set_page_config(page_title="STATBOT", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #FFFFFF; }
    .statbot-banner {
        background-color: #111111;
        color: #FFFFFF;
        padding: 1.1rem 1.5rem;
        border-radius: 10px;
        border-left: 8px solid #C8102E;
        margin-bottom: 1rem;
    }
    .statbot-banner h1 { color: #FFFFFF; margin: 0; font-size: 1.6rem; }
    .statbot-banner p { color: #DDDDDD; margin: 0.2rem 0 0 0; font-size: 0.95rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #F2F2F2;
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #C8102E !important;
        color: #FFFFFF !important;
    }
    div.stButton > button, button[kind="primary"], button[kind="secondary"] {
        background-color: #C8102E;
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
    }
    div.stButton > button:hover { background-color: #a50d24; color: #FFFFFF; }
    .tool-badge {
        display: inline-block;
        background-color: #C8102E;
        color: #FFFFFF;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.8rem;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

col_logo, col_title = st.columns([1, 5])
with col_logo:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=140)
with col_title:
    st.markdown(
        """
        <div class="statbot-banner">
            <h1>STATBOT · Asistente de Estadística</h1>
            <p>RAG + herramientas de cálculo estadístico sobre un flujo LangGraph</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.header("Acerca de STATBOT")
    st.write(
        "Asistente construido para el evento universitario, con dos formas de "
        "interactuar:"
    )
    st.markdown(
        "- **Chat**: preguntas en lenguaje natural (teoría o cálculo).\n"
        "- **Pruebas guiadas**: formularios explícitos por tipo de prueba."
    )
    st.divider()
    st.caption(
        "Tip para el chat: incluye los datos entre corchetes, ej. "
        "`[10, 12, 9, 11, 13, 8]`, y palabras clave como *prueba t* o *chi cuadrado*."
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


def render_tool_plot(tool_type, tool_result, key_suffix):
    if tool_type == "resumen" and tool_result:
        st.plotly_chart(
            plot_histograma(tool_result["datos"]), use_container_width=True, key=f"plot_{key_suffix}"
        )
    elif tool_type == "t_test" and tool_result:
        res_t = tool_result["t_test"]
        st.plotly_chart(
            plot_t_distribution(res_t["t_stat"], res_t["n"] - 1),
            use_container_width=True,
            key=f"plot_{key_suffix}",
        )
    elif tool_type == "chi2" and tool_result:
        st.plotly_chart(
            plot_chi2_barras(tool_result["observados"], tool_result["esperados"]),
            use_container_width=True,
            key=f"plot_{key_suffix}",
        )


tab_chat, tab_guiado = st.tabs(["💬 Chat", "📋 Pruebas guiadas"])

with tab_chat:
    for i, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown(f'<span class="tool-badge">{TOOL_LABELS[msg.get("tool_type")]}</span>', unsafe_allow_html=True)
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_tool_plot(msg.get("tool_type"), msg.get("tool_result"), key_suffix=f"hist_{i}")

    pregunta = st.chat_input("Escribe tu pregunta de estadística...")
    if pregunta:
        st.session_state.chat_history.append({"role": "user", "content": pregunta})
        resultado = answer_stats_langgraph(pregunta, thread_id=st.session_state.thread_id)
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": resultado["answer"],
                "tool_type": resultado["tool_type"],
                "tool_result": resultado["tool_result"],
            }
        )
        st.rerun()

with tab_guiado:
    prueba = st.radio(
        "Selecciona la prueba estadística",
        ["Resumen descriptivo", "Prueba t de una muestra", "Chi-cuadrado bondad de ajuste"],
        horizontal=True,
    )

    def parse_floats(texto):
        return [float(x.strip()) for x in texto.split(",") if x.strip() != ""]

    if prueba == "Resumen descriptivo":
        with st.form("form_resumen"):
            datos_txt = st.text_area("Datos (separados por comas)", value="10, 12, 9, 11, 13, 8")
            enviado = st.form_submit_button("Calcular")
        if enviado:
            try:
                datos = parse_floats(datos_txt)
                if len(datos) < 2:
                    st.error("Ingresa al menos 2 valores numéricos.")
                else:
                    res = resumen_descriptivo(datos)
                    st.markdown(format_resumen(res))
                    st.plotly_chart(plot_histograma(datos), use_container_width=True, key="guiado_resumen")
            except ValueError:
                st.error("No se pudieron interpretar los datos. Usa números separados por comas.")

    elif prueba == "Prueba t de una muestra":
        with st.form("form_ttest"):
            datos_txt = st.text_area("Datos (separados por comas)", value="10, 12, 9, 11, 13, 8, 10")
            mu0 = st.number_input("Media hipotética (H0)", value=10.0, step=0.1)
            enviado = st.form_submit_button("Calcular")
        if enviado:
            try:
                datos = parse_floats(datos_txt)
                if len(datos) < 2:
                    st.error("Ingresa al menos 2 valores numéricos.")
                else:
                    res = t_test_una_muestra(datos, mu0)
                    st.markdown(format_t_test(res))
                    st.plotly_chart(
                        plot_t_distribution(res["t_stat"], res["n"] - 1),
                        use_container_width=True,
                        key="guiado_ttest",
                    )
            except ValueError:
                st.error("No se pudieron interpretar los datos. Usa números separados por comas.")

    else:
        with st.form("form_chi2"):
            obs_txt = st.text_area("Frecuencias observadas (separadas por comas)", value="12, 15, 8")
            esp_txt = st.text_area("Frecuencias esperadas (separadas por comas)", value="10, 10, 15")
            enviado = st.form_submit_button("Calcular")
        if enviado:
            try:
                observados = parse_floats(obs_txt)
                esperados = parse_floats(esp_txt)
                if len(observados) < 2 or len(observados) != len(esperados):
                    st.error("Observados y esperados deben tener la misma cantidad de categorías (mínimo 2).")
                else:
                    res = chi_cuadrado_bondad(observados, esperados)
                    st.markdown(format_chi2(res))
                    st.plotly_chart(
                        plot_chi2_barras(observados, esperados), use_container_width=True, key="guiado_chi2"
                    )
            except ValueError:
                st.error("No se pudieron interpretar los datos. Usa números separados por comas.")
