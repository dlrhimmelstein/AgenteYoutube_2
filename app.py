# -*- coding: utf-8 -*-

import os

import streamlit as st


# =========================
# 1. CONFIGURACION DE PAGINA
# =========================

st.set_page_config(
    page_title="Las Damitas Histeria | Agente YouTube",
    page_icon="play",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# 2. CREDENCIALES
# =========================

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

if not os.environ.get("GOOGLE_API_KEY"):
    st.error("Error critico: no se encontro GOOGLE_API_KEY en Secrets o en el archivo .env.")
    st.stop()

try:
    has_gcp_secret = "gcp_service_account" in st.secrets
except Exception:
    has_gcp_secret = False

if not has_gcp_secret and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    st.warning(
        "No se encontro gcp_service_account en Secrets ni GOOGLE_APPLICATION_CREDENTIALS. "
        "En local se intentara usar credenciales ADC de Google."
    )


# =========================
# 3. IMPORTACION DEL AGENTE
# =========================

try:
    from agent import (
        AGENT_BUILD_ID,
        CHANNEL_ID,
        DATASET_ID,
        PROJECT_ID,
        SEGMENTS_TABLE_ID,
        TABLE_NAME,
        get_agent,
        get_retriever,
    )
except Exception as exc:
    st.error("Error al importar el agente desde agent.py.")
    st.exception(exc)
    st.stop()

AGENT_BUILD_ID = globals().get("AGENT_BUILD_ID", "sin build id")


# =========================
# 4. RECURSOS
# =========================

try:
    retriever = get_retriever()
    agent = get_agent()
except Exception as exc:
    st.error(
        "No se pudo inicializar BigQuery. Revisa gcp_service_account en "
        "Streamlit Secrets o configura credenciales ADC."
    )
    st.exception(exc)
    st.stop()


def format_compact_number(value):
    try:
        value = float(value or 0)
    except Exception:
        return "0"

    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{int(value):,}"


@st.cache_data(show_spinner=False, ttl=900)
def load_sidebar_stats():
    try:
        metrics = retriever.analytics_summary() or {}
    except Exception:
        metrics = {}

    try:
        segment_stats = retriever.transcript_segments_stats()
    except Exception:
        segment_stats = {
            "existe": False,
            "videos": 0,
            "segmentos": 0,
            "actualizado": None,
            "embedding_model": None,
        }

    return metrics, segment_stats


metrics, segment_stats = load_sidebar_stats()


# =========================
# 5. ESTILOS
# =========================

st.markdown(
    """
    <style>
    .stApp {
        background: #f7f7f7;
        color: #111827;
    }

    .block-container {
        padding-top: 0rem;
        padding-bottom: 5rem;
        padding-left: 1.4rem;
        padding-right: 1.4rem;
        max-width: 100%;
    }

    html, body, [class*="css"] {
        font-family: Inter, "Segoe UI", sans-serif;
    }

    header, footer, #MainMenu {
        display: none !important;
        visibility: hidden;
    }

    .yt-header-wrapper {
        width: 100%;
        min-height: 60px;
        background: #ffffff;
        border-bottom: 1px solid #e5e7eb;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 1.2rem;
        margin: 0rem -1.4rem 1rem -1.4rem;
        box-sizing: border-box;
    }

    .yt-header-left {
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }

    .yt-logo, .sidebar-logo, .empty-logo {
        background: #ff0000;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        letter-spacing: 0;
    }

    .yt-logo {
        width: 34px;
        height: 34px;
        border-radius: 10px;
        font-size: 0.8rem;
    }

    .yt-title {
        color: #0f0f0f;
        font-size: 1rem;
        font-weight: 850;
        line-height: 1.1;
    }

    .yt-subtitle {
        color: #6b7280;
        font-size: 0.75rem;
        margin-top: 2px;
    }

    .yt-header-right {
        display: flex;
        gap: 0.55rem;
        align-items: center;
        flex-wrap: wrap;
    }

    .yt-pill {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 999px;
        padding: 0.35rem 0.8rem;
        font-size: 0.76rem;
        font-weight: 650;
        color: #4b5563;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        white-space: nowrap;
    }

    .welcome-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 1rem 1.15rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .panel-title {
        color: #111827;
        font-size: 0.92rem;
        font-weight: 850;
        margin-bottom: 0.35rem;
    }

    .panel-subtitle {
        color: #6b7280;
        font-size: 0.78rem;
        line-height: 1.45;
        margin-bottom: 0.8rem;
    }

    .helper-stat {
        display: flex;
        justify-content: space-between;
        gap: 0.8rem;
        padding: 0.55rem 0;
        border-top: 1px solid #f0f0f0;
        color: #4b5563;
        font-size: 0.78rem;
    }

    .helper-stat b {
        color: #111827;
    }

    .chat-history-note {
        color: #6b7280;
        font-size: 0.75rem;
        margin: 0.7rem 0 0.4rem;
    }

    .welcome-top {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
    }

    .welcome-icon {
        width: 42px;
        height: 42px;
        border-radius: 14px;
        background: linear-gradient(135deg, #ff0033, #ff4d6d);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        font-weight: 900;
        flex-shrink: 0;
    }

    .welcome-title {
        font-size: 1rem;
        font-weight: 850;
        color: #0f0f0f;
        margin-bottom: 0.25rem;
    }

    .welcome-subtitle {
        font-size: 0.84rem;
        line-height: 1.5;
        color: #6b7280;
    }

    .welcome-tags-text {
        margin-top: 1rem;
        color: #374151;
        font-size: 0.78rem;
        font-weight: 650;
    }

    [data-testid="stSidebar"] {
        background-color: #f2f2f2;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0.45rem !important;
    }

    .sidebar-title {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.8rem;
    }

    .sidebar-logo {
        width: 30px;
        height: 30px;
        border-radius: 8px;
        font-size: 0.7rem;
    }

    .sidebar-main-title {
        font-size: 0.9rem;
        font-weight: 850;
        color: #0f0f0f;
    }

    .sidebar-subtitle {
        font-size: 0.67rem;
        color: #8a8a8a;
    }

    .sidebar-section-title {
        font-size: 0.64rem;
        font-weight: 850;
        color: #9ca3af;
        letter-spacing: 0.08rem;
        margin: 0.48rem 0 0.32rem 0;
    }

    .sidebar-item {
        background: transparent;
        border-radius: 10px;
        padding: 0.34rem 0.38rem;
        margin-bottom: 0.04rem;
        color: #0f0f0f;
        font-size: 0.74rem;
        display: grid;
        grid-template-columns: 24px 1fr;
        column-gap: 0.35rem;
        align-items: center;
    }

    .sidebar-item:hover {
        background: #e5e5e5;
    }

    .sidebar-item span {
        font-weight: 750;
    }

    .sidebar-item small {
        grid-column: 2;
        color: #8a8a8a;
        font-size: 0.63rem;
        margin-top: -0.08rem;
    }

    .sidebar-divider {
        height: 1px;
        background: #dddddd;
        margin: 0.55rem 0;
    }

    .channel-status-card {
        border-top: 1px solid #dddddd;
        padding-top: 0.55rem;
        margin-top: 0.35rem;
    }

    .channel-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.72rem;
        margin-bottom: 0.28rem;
        gap: 0.8rem;
    }

    .channel-row span {
        color: #8a8a8a;
        font-weight: 520;
    }

    .channel-row b {
        color: #0f0f0f;
        font-weight: 780;
        text-align: right;
    }

    .agent-active {
        color: #e60023 !important;
    }

    .connection-info {
        margin-top: 0.8rem;
        color: #0f0f0f;
        font-size: 0.82rem;
    }

    .connection-info code {
        background: #111827;
        color: #22c55e;
        padding: 0.15rem 0.35rem;
        border-radius: 6px;
    }

    .stButton > button {
        border-radius: 999px;
        border: 1px solid #d1d5db;
        background-color: #ffffff;
        color: #374151;
        font-weight: 700;
        min-height: 32px;
        padding-top: 0.25rem;
        padding-bottom: 0.25rem;
        font-size: 0.78rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background-color: #f1f1f1;
        border-color: #c7c7c7;
        color: #0f0f0f;
        transform: translateY(-1px);
    }

    .empty-logo, .empty-title, .empty-text {
        max-width: 560px;
        margin-left: auto;
        margin-right: auto;
        text-align: center;
    }

    .empty-logo {
        width: 64px;
        height: 48px;
        margin-top: 4rem;
        margin-bottom: 1.2rem;
        border-radius: 16px;
        font-size: 0.9rem;
        box-shadow: 0 8px 20px rgba(255,0,0,0.25);
    }

    .empty-title {
        font-size: 1.35rem;
        font-weight: 850;
        color: #111827;
        margin-bottom: 0.8rem;
    }

    .empty-text {
        font-size: 0.95rem;
        line-height: 1.6;
        color: #4b5563;
    }

    [data-testid="stChatMessage"] {
        background: transparent !important;
        padding: 0 !important;
    }

    [data-testid="stChatMessageContent"] {
        background: white;
        color: #0f0f0f;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 0.75rem 0.9rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 0.55rem;
        overflow-wrap: anywhere;
        word-break: normal;
    }

    .thinking-box {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 0.9rem 1rem;
        width: fit-content;
        color: #4b5563;
        font-size: 0.9rem;
    }

    .thinking-dot {
        width: 10px;
        height: 10px;
        border-radius: 999px;
        background: #ff0000;
        animation: pulse 1s infinite;
    }

    @keyframes pulse {
        0% { opacity: 0.4; transform: scale(0.9); }
        50% { opacity: 1; transform: scale(1.1); }
        100% { opacity: 0.4; transform: scale(0.9); }
    }

    [data-testid="stBottom"] {
        background: #f1f3f4 !important;
        border-top: 1px solid #e5e7eb !important;
        padding: 0.8rem 2rem !important;
    }

    [data-testid="stBottom"] > div {
        max-width: 1180px !important;
        margin: 0 auto !important;
    }

    [data-baseweb="textarea"] {
        border-radius: 999px !important;
        border: 1px solid #d1d5db !important;
        background: #ffffff !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
    }

    [data-baseweb="textarea"] textarea {
        background: #ffffff !important;
        color: #111827 !important;
        font-size: 0.95rem !important;
        padding-top: 0.95rem !important;
        padding-left: 1.2rem !important;
    }

    [data-baseweb="textarea"] textarea::placeholder {
        color: #9ca3af !important;
        opacity: 1 !important;
    }

    [data-testid="stChatInput"] button {
        background: #ff0000 !important;
        color: white !important;
        border-radius: 999px !important;
        border: none !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# 6. HEADER
# =========================

videos_count = segment_stats.get("videos") or metrics.get("videos") or 0

st.markdown(
    f"""
    <div class="yt-header-wrapper">
        <div class="yt-header-left">
            <div class="yt-logo">PLAY</div>
            <div>
                <div class="yt-title">Las Damitas Histeria</div>
                <div class="yt-subtitle">Agente de analisis · Gemini + BigQuery</div>
            </div>
        </div>
        <div class="yt-header-right">
            <div class="yt-pill">Gemini conectado</div>
            <div class="yt-pill">{format_compact_number(videos_count)} videos</div>
            <div class="yt-pill">{format_compact_number(segment_stats.get("segmentos"))} segmentos</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================
# 7. SIDEBAR
# =========================

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-title">
            <span class="sidebar-logo">YT</span>
            <div>
                <div class="sidebar-main-title">Las Damitas Histeria</div>
                <div class="sidebar-subtitle">Agente YouTube Analytics</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section-title">ACCESOS RAPIDOS</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-item">TOP <span>Top videos</span><small>Ranking por vistas</small></div>
        <div class="sidebar-item">DIA <span>Mejor dia para publicar</span><small>Views, likes y engagement</small></div>
        <div class="sidebar-item">TEM <span>Temas exitosos</span><small>Por interaccion</small></div>
        <div class="sidebar-item">RES <span>Resumen del canal</span><small>Metricas generales</small></div>
        <div class="sidebar-item">BUS <span>Buscar por tema</span><small>En que episodio hablaron de X</small></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="channel-status-card">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">CANAL AL DIA</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="channel-row"><span>Videos</span><b>{format_compact_number(metrics.get("videos") or videos_count)}</b></div>
        <div class="channel-row"><span>Views</span><b>{format_compact_number(metrics.get("views"))}</b></div>
        <div class="channel-row"><span>Likes</span><b>{format_compact_number(metrics.get("likes"))}</b></div>
        <div class="channel-row"><span>Comentarios</span><b>{format_compact_number(metrics.get("comentarios"))}</b></div>
        <div class="channel-row"><span>Segmentos</span><b>{format_compact_number(segment_stats.get("segmentos"))}</b></div>
        <div class="channel-row"><span>Estado</span><b class="agent-active">Activo</b></div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">BIGQUERY</div>', unsafe_allow_html=True)

    if segment_stats.get("existe"):
        st.success("Tabla de segmentos lista")
        st.caption(f"Tabla: `{SEGMENTS_TABLE_ID}`")
        if segment_stats.get("embedding_model"):
            st.caption(f"Embedding model: {segment_stats['embedding_model']}")
        st.caption(f"Actualizado: {segment_stats.get('actualizado')}")
    else:
        st.warning("Tabla de segmentos no encontrada")
        st.caption(f"Esperada: `{SEGMENTS_TABLE_ID}`")

    if st.button("Probar BigQuery", use_container_width=True):
        with st.spinner("Verificando conexion con BigQuery..."):
            try:
                info = retriever.test_connection()
                st.success("Conexion exitosa")
                st.markdown(
                    f"""
                    <div class="connection-info">
                        <p><b>Tabla:</b> <code>{info["tabla"]}</code></p>
                        <p><b>Filas:</b> <code>{info["filas"]}</code></p>
                        <p><b>Columnas:</b> <code>{info["columnas"]}</code></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as exc:
                st.error("No se pudo conectar con BigQuery.")
                st.exception(exc)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    if st.button("Limpiar conversacion", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# =========================
# 8. BIENVENIDA
# =========================

st.markdown(
    """
    <div class="welcome-card">
        <div class="welcome-top">
            <div class="welcome-icon">AI</div>
            <div>
                <div class="welcome-title">Que puede hacer este agente?</div>
                <div class="welcome-subtitle">
                    Consulta metricas, rendimiento, temas, transcripciones, recomendaciones,
                    mejores dias para publicar y momentos aproximados dentro de episodios.
                </div>
            </div>
        </div>
        <div class="welcome-tags-text">
            Analytics · Videos · Engagement · Transcripciones · Gemini AI · BigQuery
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================
# 9. MEMORIA
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================
# 10. CHAT
# =========================

prompt = st.chat_input("Pregunta sobre el canal... ej: Que temas tuvieron mas engagement?")

if "prompt_sugerido" in st.session_state:
    prompt = st.session_state.pop("prompt_sugerido")

main_col, helper_col = st.columns([1.7, 0.75], gap="medium")

with main_col:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Conversacion con el agente</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-subtitle">La respuesta nueva aparece arriba. El historial queda abajo con scroll para que las respuestas largas no empujen toda la pantalla.</div>',
            unsafe_allow_html=True,
        )

        if prompt:
            history_for_agent = st.session_state.messages[-8:]
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.container(height=320, border=True):
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    thinking_placeholder = st.empty()
                    thinking_placeholder.markdown(
                        """
                        <div class="thinking-box">
                            <div class="thinking-dot"></div>
                            Analizando metricas y transcripciones...
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    try:
                        answer = agent.answer(prompt, history=history_for_agent)
                        thinking_placeholder.empty()
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as exc:
                        thinking_placeholder.empty()
                        error_message = (
                            "**Ocurrio un error al procesar tu pregunta.**\n\n"
                            f"`{str(exc)}`\n\n"
                            "Revisa Secrets, permisos de BigQuery y la tabla de segmentos."
                        )
                        st.error(error_message)
                        st.exception(exc)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})

        if not st.session_state.messages:
            st.markdown('<div class="empty-logo">PLAY</div>', unsafe_allow_html=True)
            st.markdown('<div class="empty-title">Hola, soy tu agente de YouTube</div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="empty-text">
                    Puedo analizar el rendimiento de <b>Las Damitas Histeria</b>, encontrar
                    en que episodio hablaron de un tema y recomendarte decisiones con datos.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="chat-history-note">Historial reciente</div>', unsafe_allow_html=True)
        history_messages = st.session_state.messages[:-2] if prompt else st.session_state.messages

        with st.container(height=420, border=False):
            if history_messages:
                for message in history_messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
            else:
                st.caption("Aqui se guardaran tus preguntas anteriores.")

with helper_col:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Panel rapido</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-subtitle">Datos del canal y acceso para limpiar la conversacion sin ocupar espacio del chat.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="helper-stat"><span>Videos</span><b>{format_compact_number(metrics.get("videos") or videos_count)}</b></div>
            <div class="helper-stat"><span>Views</span><b>{format_compact_number(metrics.get("views"))}</b></div>
            <div class="helper-stat"><span>Likes</span><b>{format_compact_number(metrics.get("likes"))}</b></div>
            <div class="helper-stat"><span>Segmentos</span><b>{format_compact_number(segment_stats.get("segmentos"))}</b></div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Limpiar conversacion", key="clear_chat_main", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
