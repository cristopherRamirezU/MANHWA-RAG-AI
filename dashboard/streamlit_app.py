"""
Dashboard de Observabilidad — ManhwaBot EP3 (IE5, IE4, IE7).

Visualiza en tiempo real las métricas registradas por backend/observability.py.

Secciones:
  1. KPIs globales  (IE1, IE2)
  2. Latencia       — línea temporal + histograma + percentiles
  3. Comportamiento del agente — iteraciones, herramientas, tokens
  4. Alertas / Anomalías detectadas  (IE4)
  5. Análisis de herramientas — latencia y tasa de éxito por herramienta
  6. Consistencia de respuestas  (IE1)
  7. Log explorer   — tabla filtrable de registros  (IE3)
  8. Recomendaciones de optimización  (IE7)

Uso:
    cd MANHWA-RAG-AI
    streamlit run dashboard/streamlit_app.py
"""

import json
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ManhwaBot — Observabilidad EP3",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Rutas a los logs (relativas a la raíz del repo)
ROOT = os.path.join(os.path.dirname(__file__), "..")
REQUESTS_LOG = os.path.join(ROOT, "backend", "data", "logs", "requests.jsonl")
TOOLS_LOG = os.path.join(ROOT, "backend", "data", "logs", "tools.jsonl")

# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def load_jsonl(path: str) -> pd.DataFrame:
    """Lee un archivo JSONL y lo convierte en DataFrame. Cache de 30 s."""
    if not os.path.exists(path):
        return pd.DataFrame()
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_data():
    req = load_jsonl(REQUESTS_LOG)
    tools = load_jsonl(TOOLS_LOG)
    return req, tools


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def color_metric(value: float, good_threshold: float, bad_threshold: float,
                 higher_is_better: bool = True) -> str:
    """Devuelve emoji de color según el valor."""
    if higher_is_better:
        if value >= good_threshold:
            return "🟢"
        if value >= bad_threshold:
            return "🟡"
        return "🔴"
    else:
        if value <= good_threshold:
            return "🟢"
        if value <= bad_threshold:
            return "🟡"
        return "🔴"


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica requests anómalos según tres criterios:
      1. Latencia > media + 2·desv_std  (outlier estadístico)
      2. Alcanzó MAX_ITERATIONS sin Final Answer (timeout de planificación)
      3. Request con error de excepción
    Retorna subset del DataFrame con columna 'tipo_anomalia'.
    """
    if df.empty:
        return pd.DataFrame()
    anomalies = []
    mean_lat = df["latencia_ms"].mean()
    std_lat = df["latencia_ms"].std()
    threshold = mean_lat + 2 * std_lat

    for _, row in df.iterrows():
        motivos = []
        if row["latencia_ms"] > threshold:
            motivos.append("Latencia alta (>μ+2σ)")
        if row.get("iteraciones", 0) >= 6:
            motivos.append("Timeout de planificación (6 iteraciones)")
        if row.get("error") and pd.notna(row.get("error")):
            motivos.append("Excepción en tiempo de ejecución")
        if motivos:
            r = row.to_dict()
            r["tipo_anomalia"] = " | ".join(motivos)
            anomalies.append(r)

    return pd.DataFrame(anomalies) if anomalies else pd.DataFrame()


def build_recommendations(req: pd.DataFrame, tools: pd.DataFrame) -> list[dict]:
    """
    Genera recomendaciones automáticas basadas en las métricas observadas.
    Cada recomendación tiene: titulo, descripcion, prioridad (Alta/Media/Baja).
    """
    recs = []
    if req.empty:
        return recs

    avg_lat = req["latencia_ms"].mean()
    success_rate = req["exito"].mean()
    p95 = req["latencia_ms"].quantile(0.95)
    timeout_rate = (req["iteraciones"] >= 6).mean()
    avg_tokens = req["tokens_total"].mean() if "tokens_total" in req.columns else 0

    if avg_lat > 3000:
        recs.append({
            "titulo": "Implementar caché de respuestas frecuentes",
            "descripcion": (
                f"La latencia promedio es {avg_lat:.0f} ms. "
                "Para queries repetidas, almacenar la respuesta en Redis o un dict "
                "en memoria reduciría la latencia a <50 ms para el 20-30% de los casos "
                "más frecuentes (Solo Leveling, recomendaciones por género)."
            ),
            "prioridad": "Alta",
        })
    elif avg_lat > 1500:
        recs.append({
            "titulo": "Optimizar el System Prompt para reducir tokens",
            "descripcion": (
                f"Latencia promedio: {avg_lat:.0f} ms. Reducir el System Prompt en un "
                "30% (eliminar ejemplos redundantes) disminuiría los prompt_tokens y "
                "en consecuencia la latencia de inferencia."
            ),
            "prioridad": "Media",
        })

    if success_rate < 0.90:
        recs.append({
            "titulo": "Mejorar el parsing ReAct con ejemplos few-shot",
            "descripcion": (
                f"Tasa de éxito: {success_rate*100:.1f}%. Agregar 2-3 ejemplos "
                "Thought/Action/Observation al System Prompt aumenta la tasa de "
                "adherencia al formato ReAct en modelos Llama (Yao et al., 2022)."
            ),
            "prioridad": "Alta",
        })

    if p95 > 5000:
        recs.append({
            "titulo": "Agregar timeout explícito a llamadas Groq",
            "descripcion": (
                f"Latencia p95 = {p95:.0f} ms. Configurar timeout=10 s en el cliente "
                "Groq evita que requests lentos bloqueen el servidor Flask. "
                "Combinar con retry exponencial (max 2 reintentos)."
            ),
            "prioridad": "Alta",
        })

    if timeout_rate > 0.10:
        recs.append({
            "titulo": "Aumentar max_tokens o simplificar descripciones de herramientas",
            "descripcion": (
                f"{timeout_rate*100:.1f}% de requests alcanza el límite de 6 iteraciones. "
                "Aumentar max_tokens de 1024 a 1536, o acortar los nombres en el "
                "System Prompt puede reducir la confusión del modelo al elegir herramientas."
            ),
            "prioridad": "Media",
        })

    if avg_tokens > 800:
        recs.append({
            "titulo": "Usar caché de prompts de la API Groq",
            "descripcion": (
                f"Tokens promedio por request: {avg_tokens:.0f}. El System Prompt es "
                "constante entre requests; habilitar prompt caching en Groq reduciría "
                "el costo de tokens de entrada en un ~60%."
            ),
            "prioridad": "Media",
        })

    if not tools.empty:
        tool_errors = tools.groupby("herramienta")["exito"].mean()
        low_success = tool_errors[tool_errors < 0.85]
        for tool_name, rate in low_success.items():
            recs.append({
                "titulo": f"Agregar fallback para herramienta '{tool_name}'",
                "descripcion": (
                    f"'{tool_name}' tiene tasa de éxito {rate*100:.1f}%. "
                    "Implementar un manejo de excepción más robusto y un mensaje "
                    "de fallback informativo mejorará la experiencia del usuario."
                ),
                "prioridad": "Alta",
            })

    recs.append({
        "titulo": "Migrar logs a base de datos para escalabilidad",
        "descripcion": (
            "El formato JSONL es adecuado para prototipado, pero en producción "
            "con >10k requests/día se recomienda migrar a PostgreSQL con la extensión "
            "TimescaleDB para consultas temporales eficientes y mayor resiliencia."
        ),
        "prioridad": "Baja",
    })

    return recs


# ---------------------------------------------------------------------------
# Sidebar — filtros
# ---------------------------------------------------------------------------

def render_sidebar(req: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.title("Filtros")
    st.sidebar.markdown("---")

    if req.empty:
        return req

    min_date = req["timestamp"].min().date()
    max_date = req["timestamp"].max().date()

    col1, col2 = st.sidebar.columns(2)
    with col1:
        start = st.date_input("Desde", value=min_date, min_value=min_date, max_value=max_date)
    with col2:
        end = st.date_input("Hasta", value=max_date, min_value=min_date, max_value=max_date)

    st.sidebar.markdown("---")
    solo_errores = st.sidebar.checkbox("Solo mostrar errores", value=False)
    min_iter = st.sidebar.slider("Iteraciones mínimas", 1, 6, 1)

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Actualización automática**\n\nEl dashboard se refresca cada 30 s. "
        "Presiona **R** para forzar recarga."
    )

    mask = (
        (req["timestamp"].dt.date >= start)
        & (req["timestamp"].dt.date <= end)
        & (req["iteraciones"] >= min_iter)
    )
    if solo_errores:
        mask &= ~req["exito"]

    return req[mask].copy()


# ---------------------------------------------------------------------------
# Sección 1 — KPIs globales
# ---------------------------------------------------------------------------

def render_kpis(df: pd.DataFrame):
    st.markdown("## Indicadores Clave de Desempeño")

    total = len(df)
    avg_lat = df["latencia_ms"].mean()
    p95_lat = df["latencia_ms"].quantile(0.95)
    success = df["exito"].mean() * 100
    error_rate = (1 - df["exito"].mean()) * 100
    avg_tokens = df["tokens_total"].mean() if "tokens_total" in df.columns else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("Total Requests", f"{total:,}")
    c2.metric(
        "Latencia Prom.",
        f"{avg_lat:,.0f} ms",
        delta=f"p95: {p95_lat:,.0f} ms",
        delta_color="off",
    )
    c3.metric(
        "Tasa de Éxito",
        f"{success:.1f}%",
        delta=color_metric(success / 100, 0.9, 0.7) + " umbral 90%",
        delta_color="off",
    )
    c4.metric(
        "Tasa de Error",
        f"{error_rate:.1f}%",
        delta=color_metric(error_rate / 100, 0.1, 0.3, higher_is_better=False) + " umbral 10%",
        delta_color="off",
    )
    c5.metric("Tokens Prom./req", f"{avg_tokens:,.0f}")
    c6.metric(
        "Iteraciones Prom.",
        f"{df['iteraciones'].mean():.1f}",
        delta=f"máx: {df['iteraciones'].max()}",
        delta_color="off",
    )


# ---------------------------------------------------------------------------
# Sección 2 — Latencia
# ---------------------------------------------------------------------------

def render_latency(df: pd.DataFrame):
    st.markdown("## Latencia (ms)")
    c1, c2 = st.columns([2, 1])

    with c1:
        df_time = df.set_index("timestamp").resample("10min")["latencia_ms"].agg(["mean", "max", "min"]).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_time["timestamp"], y=df_time["mean"],
                                  name="Promedio", line=dict(color="#4a90d9", width=2)))
        fig.add_trace(go.Scatter(x=df_time["timestamp"], y=df_time["max"],
                                  name="Máximo", line=dict(color="#e74c3c", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=df_time["timestamp"], y=df_time["min"],
                                  name="Mínimo", line=dict(color="#50b86c", width=1, dash="dot")))
        fig.update_layout(
            title="Latencia por ventana de 10 min",
            xaxis_title="Tiempo",
            yaxis_title="ms",
            height=320,
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.histogram(
            df, x="latencia_ms", nbins=20,
            title="Distribución de latencia",
            labels={"latencia_ms": "ms", "count": "Requests"},
            color_discrete_sequence=["#4a90d9"],
        )
        fig2.update_layout(height=320, margin=dict(t=40, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    # Percentiles
    st.markdown("**Percentiles de latencia**")
    p_cols = st.columns(5)
    for i, p in enumerate([50, 75, 90, 95, 99]):
        val = df["latencia_ms"].quantile(p / 100)
        p_cols[i].metric(f"p{p}", f"{val:,.0f} ms")


# ---------------------------------------------------------------------------
# Sección 3 — Comportamiento del agente
# ---------------------------------------------------------------------------

def render_agent_behavior(df: pd.DataFrame):
    st.markdown("## Comportamiento del Agente")
    c1, c2, c3 = st.columns(3)

    with c1:
        iter_counts = df["iteraciones"].value_counts().sort_index().reset_index()
        iter_counts.columns = ["iteraciones", "requests"]
        fig = px.bar(iter_counts, x="iteraciones", y="requests",
                     title="Distribución de iteraciones ReAct",
                     labels={"iteraciones": "# Ciclos", "requests": "Requests"},
                     color_discrete_sequence=["#9b59b6"])
        fig.update_layout(height=300, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        all_tools = []
        for tools in df["herramientas"]:
            if isinstance(tools, list):
                all_tools.extend(tools)
        if all_tools:
            tool_counts = pd.Series(all_tools).value_counts().reset_index()
            tool_counts.columns = ["herramienta", "llamadas"]
            fig2 = px.bar(tool_counts, x="herramienta", y="llamadas",
                          title="Frecuencia de uso de herramientas",
                          color_discrete_sequence=["#e8a838"])
            fig2.update_layout(height=300, margin=dict(t=40, b=20))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Sin herramientas registradas en el rango seleccionado.")

    with c3:
        if "tokens_total" in df.columns:
            df_tok = df.set_index("timestamp").resample("30min")[["tokens_entrada", "tokens_salida"]].mean().reset_index()
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=df_tok["timestamp"], y=df_tok["tokens_entrada"],
                                   name="Entrada", marker_color="#4a90d9"))
            fig3.add_trace(go.Bar(x=df_tok["timestamp"], y=df_tok["tokens_salida"],
                                   name="Salida", marker_color="#50b86c"))
            fig3.update_layout(
                barmode="stack",
                title="Tokens promedio por ventana 30 min",
                xaxis_title="Tiempo",
                yaxis_title="Tokens",
                height=300,
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig3, use_container_width=True)


# ---------------------------------------------------------------------------
# Sección 4 — Alertas / Anomalías (IE4)
# ---------------------------------------------------------------------------

def render_anomalies(df: pd.DataFrame):
    st.markdown("## Alertas y Anomalías Detectadas")

    anomalies = detect_anomalies(df)

    if anomalies.empty:
        st.success("No se detectaron anomalías en el rango de tiempo seleccionado.")
        return

    st.warning(f"Se detectaron **{len(anomalies)}** requests anómalos.")

    display_cols = ["timestamp", "latencia_ms", "iteraciones", "exito", "tipo_anomalia"]
    available = [c for c in display_cols if c in anomalies.columns]
    st.dataframe(
        anomalies[available].sort_values("timestamp", ascending=False).head(20),
        use_container_width=True,
        hide_index=True,
    )

    # Gráfico de anomalías sobre línea temporal de latencia
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["latencia_ms"],
        mode="markers", name="Normal",
        marker=dict(color="#4a90d9", size=5, opacity=0.5),
    ))
    if not anomalies.empty and "timestamp" in anomalies.columns:
        fig.add_trace(go.Scatter(
            x=anomalies["timestamp"], y=anomalies["latencia_ms"],
            mode="markers", name="Anomalía",
            marker=dict(color="#e74c3c", size=10, symbol="x"),
        ))
    mean_lat = df["latencia_ms"].mean()
    std_lat = df["latencia_ms"].std()
    fig.add_hline(y=mean_lat + 2 * std_lat, line_dash="dash",
                  line_color="orange", annotation_text="μ + 2σ")
    fig.update_layout(
        title="Latencia con anomalías marcadas",
        xaxis_title="Tiempo",
        yaxis_title="ms",
        height=300,
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Sección 5 — Análisis de herramientas
# ---------------------------------------------------------------------------

def render_tools_analysis(tools_df: pd.DataFrame):
    st.markdown("## Análisis de Herramientas")

    if tools_df.empty:
        st.info("Sin datos de herramientas registrados.")
        return

    c1, c2 = st.columns(2)

    with c1:
        lat_by_tool = tools_df.groupby("herramienta")["latencia_ms"].mean().reset_index()
        lat_by_tool.columns = ["herramienta", "latencia_promedio_ms"]
        fig = px.bar(lat_by_tool, x="herramienta", y="latencia_promedio_ms",
                     title="Latencia promedio por herramienta (ms)",
                     color="latencia_promedio_ms",
                     color_continuous_scale="RdYlGn_r",
                     text_auto=".0f")
        fig.update_layout(height=320, margin=dict(t=40, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        success_by_tool = tools_df.groupby("herramienta")["exito"].mean().reset_index()
        success_by_tool.columns = ["herramienta", "tasa_exito"]
        success_by_tool["tasa_exito_pct"] = success_by_tool["tasa_exito"] * 100
        fig2 = px.bar(success_by_tool, x="herramienta", y="tasa_exito_pct",
                      title="Tasa de éxito por herramienta (%)",
                      color="tasa_exito_pct",
                      color_continuous_scale="RdYlGn",
                      range_y=[0, 105],
                      text_auto=".1f")
        fig2.update_layout(height=320, margin=dict(t=40, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Tabla resumen de herramientas
    summary = tools_df.groupby("herramienta").agg(
        llamadas=("latencia_ms", "count"),
        latencia_prom_ms=("latencia_ms", "mean"),
        latencia_max_ms=("latencia_ms", "max"),
        tasa_exito=("exito", "mean"),
    ).round(2).reset_index()
    summary["tasa_exito"] = (summary["tasa_exito"] * 100).round(1).astype(str) + "%"
    st.dataframe(summary, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Sección 6 — Consistencia
# ---------------------------------------------------------------------------

def render_consistency(df: pd.DataFrame):
    st.markdown("## Consistencia de Respuestas")

    cons_data = df[df["consistencia"].notna()].copy()

    if cons_data.empty:
        st.info(
            "Sin datos de consistencia aún. La consistencia se calcula a partir de "
            "la segunda vez que se realiza la misma pregunta."
        )
        return

    c1, c2 = st.columns([1, 2])

    with c1:
        avg_c = cons_data["consistencia"].mean()
        st.metric("Consistencia promedio", f"{avg_c:.3f}", help="Jaccard similarity (0=diferente, 1=idéntica)")
        st.metric("Mín.", f"{cons_data['consistencia'].min():.3f}")
        st.metric("Máx.", f"{cons_data['consistencia'].max():.3f}")
        low = (cons_data["consistencia"] < 0.5).sum()
        st.metric("Respuestas < 0.5 (baja consistencia)", f"{low}")

    with c2:
        fig = px.histogram(
            cons_data, x="consistencia", nbins=20,
            title="Distribución de consistencia (Jaccard)",
            labels={"consistencia": "Score", "count": "Ocurrencias"},
            color_discrete_sequence=["#50b86c"],
        )
        fig.add_vline(x=0.7, line_dash="dash", line_color="orange",
                      annotation_text="umbral recomendado 0.7")
        fig.update_layout(height=280, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Sección 7 — Log Explorer
# ---------------------------------------------------------------------------

def render_log_explorer(df: pd.DataFrame):
    st.markdown("## Explorador de Registros")

    display_df = df[["timestamp", "latencia_ms", "iteraciones", "herramientas",
                      "tokens_total", "exito", "error"]].copy()
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    display_df["herramientas"] = display_df["herramientas"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else str(x)
    )
    display_df = display_df.sort_values("timestamp", ascending=False).head(50)

    st.dataframe(display_df, use_container_width=True, hide_index=True,
                 column_config={
                     "latencia_ms": st.column_config.NumberColumn("Latencia (ms)", format="%.0f"),
                     "tokens_total": st.column_config.NumberColumn("Tokens"),
                     "exito": st.column_config.CheckboxColumn("Éxito"),
                 })

    # Errores recientes
    errors = df[df["error"].notna() & df["error"].ne("")].copy()
    if not errors.empty:
        st.markdown("**Errores recientes**")
        err_display = errors[["timestamp", "error", "iteraciones"]].copy()
        err_display["timestamp"] = err_display["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(err_display.tail(10), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Sección 8 — Recomendaciones (IE7)
# ---------------------------------------------------------------------------

def render_recommendations(req: pd.DataFrame, tools: pd.DataFrame):
    st.markdown("## Recomendaciones de Optimización")
    st.caption("Generadas automáticamente a partir de las métricas observadas.")

    recs = build_recommendations(req, tools)
    if not recs:
        st.info("No hay suficientes datos para generar recomendaciones.")
        return

    priority_color = {"Alta": "🔴", "Media": "🟡", "Baja": "🟢"}
    for rec in sorted(recs, key=lambda r: {"Alta": 0, "Media": 1, "Baja": 2}[r["prioridad"]]):
        icon = priority_color.get(rec["prioridad"], "")
        with st.expander(f"{icon} **[{rec['prioridad']}]** {rec['titulo']}"):
            st.markdown(rec["descripcion"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("📊 ManhwaBot — Dashboard de Observabilidad")
    st.caption(
        "EP3 — ISY0101 Ingeniería de Soluciones con IA | "
        "Autor: Cristopher Alexander Ramírez Ubilla"
    )

    req_all, tools_all = load_data()

    if req_all.empty:
        st.warning(
            "No hay datos de métricas disponibles.\n\n"
            "**Opciones para generar datos:**\n"
            "1. Ejecuta el agente realizando consultas: `cd backend && python app.py`\n"
            "2. Genera datos de demostración: `python tests/seed_metrics.py`"
        )
        st.stop()

    # Sidebar con filtros — devuelve DataFrame filtrado
    req = render_sidebar(req_all)

    if req.empty:
        st.warning("No hay datos para el rango y filtros seleccionados.")
        st.stop()

    # Secciones del dashboard
    render_kpis(req)
    st.markdown("---")
    render_latency(req)
    st.markdown("---")
    render_agent_behavior(req)
    st.markdown("---")
    render_anomalies(req)
    st.markdown("---")
    render_tools_analysis(tools_all)
    st.markdown("---")
    render_consistency(req)
    st.markdown("---")
    render_log_explorer(req)
    st.markdown("---")
    render_recommendations(req, tools_all)

    # Footer
    st.markdown("---")
    st.caption(
        f"Datos cargados desde `backend/data/logs/`. "
        f"Total en período: {len(req):,} requests | "
        f"Fuente completa: {len(req_all):,} requests"
    )


if __name__ == "__main__":
    main()
