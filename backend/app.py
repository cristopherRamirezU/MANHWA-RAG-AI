from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_pipeline import buscar_contexto, generar_respuesta

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


# ---------------------------------------------------------------------------
# Endpoint legado — pipeline RAG simple (mantiene compatibilidad EP1)
# ---------------------------------------------------------------------------

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "pregunta" not in data:
        return jsonify({"error": "No se recibió la pregunta"}), 400

    pregunta = data["pregunta"]
    contexto = buscar_contexto(pregunta)
    respuesta = generar_respuesta(pregunta, contexto)
    return jsonify({"respuesta": respuesta})


# ---------------------------------------------------------------------------
# Endpoint principal — agente ReAct con memoria y observabilidad (EP2 / EP3)
# ---------------------------------------------------------------------------

@app.route("/agent", methods=["POST"])
def agent_chat():
    """
    Procesa una pregunta usando el agente ManhwaBot (ReAct + Groq).

    Aplica validación de entrada y rate limiting (EP3 — IE6) antes de invocar
    al agente. Registra métricas de observabilidad automáticamente (via agent.py).

    Body JSON:
        {
            "pregunta": "¿Qué manhwas de acción me recomiendas?",
            "user_id": "usuario_123"   (opcional, default: "default")
        }

    Response JSON (éxito):
        { "respuesta": "...", "user_id": "..." }

    Response JSON (error de validación):
        { "error": "..." }  — HTTP 400 o 429
    """
    from security import validate_input, check_rate_limit, sanitize_user_id

    data = request.get_json()
    if not data or "pregunta" not in data:
        return jsonify({"error": "No se recibió la pregunta"}), 400

    raw_pregunta = data["pregunta"]
    user_id = sanitize_user_id(data.get("user_id", "default"))

    # ── Seguridad: rate limiting ───────────────────────────────────────────
    if not check_rate_limit(user_id):
        return jsonify({"error": "Límite de requests superado. Espera un minuto."}), 429

    # ── Seguridad: validación de entrada ──────────────────────────────────
    ok, result = validate_input(raw_pregunta)
    if not ok:
        return jsonify({"error": result}), 400
    pregunta = result

    try:
        from agent import get_agent
        agente = get_agent(user_id)
        respuesta = agente.chat(pregunta)
        return jsonify({"respuesta": respuesta, "user_id": user_id})
    except Exception as e:
        # Nunca exponer stack trace al cliente (IE6)
        print(f"[app.py ERROR] {e}")
        return jsonify({"error": "Error interno del servidor."}), 500


# ---------------------------------------------------------------------------
# Endpoint de métricas — resumen en tiempo real (EP3 — IE2, IE3)
# ---------------------------------------------------------------------------

@app.route("/metrics/summary", methods=["GET"])
def metrics_summary():
    """
    Devuelve un resumen estadístico de las métricas de observabilidad.

    Útil para monitoreo rápido sin necesidad de abrir el dashboard Streamlit.

    Response JSON:
        {
          "total_requests": 127,
          "tasa_exito": 0.913,
          "latencia_promedio_ms": 1245.3,
          "latencia_p95_ms": 2890.1,
          "tokens_promedio": 637,
          "herramienta_mas_usada": "buscar_manhwa",
          "total_errores": 11
        }
    """
    import json
    import os
    import statistics

    logs_dir = os.path.join(os.path.dirname(__file__), "data", "logs")
    requests_log = os.path.join(logs_dir, "requests.jsonl")

    if not os.path.exists(requests_log):
        return jsonify({"mensaje": "Sin métricas aún. Realiza consultas al agente primero."})

    records = []
    with open(requests_log, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        return jsonify({"mensaje": "Sin métricas aún."})

    latencias = [r["latencia_ms"] for r in records]
    exitos = [r["exito"] for r in records]
    tokens = [r.get("tokens_total", 0) for r in records]
    errores = [r for r in records if not r["exito"]]

    from collections import Counter
    herramientas_todas = []
    for r in records:
        herramientas_todas.extend(r.get("herramientas", []))
    herramienta_top = Counter(herramientas_todas).most_common(1)

    latencias_sorted = sorted(latencias)
    p95_idx = int(len(latencias_sorted) * 0.95)

    return jsonify({
        "total_requests": len(records),
        "tasa_exito": round(sum(exitos) / len(exitos), 3),
        "latencia_promedio_ms": round(statistics.mean(latencias), 2),
        "latencia_p95_ms": round(latencias_sorted[min(p95_idx, len(latencias_sorted) - 1)], 2),
        "tokens_promedio": round(statistics.mean(tokens), 1) if tokens else 0,
        "herramienta_mas_usada": herramienta_top[0][0] if herramienta_top else "N/A",
        "total_errores": len(errores),
    })


# ---------------------------------------------------------------------------
# Endpoint de estado
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "3.0-observability"})


if __name__ == "__main__":
    app.run(debug=True)
