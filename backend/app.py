from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_pipeline import buscar_contexto, generar_respuesta

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


# ---------------------------------------------------------------------------
# Endpoint legado — pipeline RAG simple (mantiene compatibilidad)
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
# Endpoint nuevo — agente LangChain ReAct con memoria
# ---------------------------------------------------------------------------

@app.route("/agent", methods=["POST"])
def agent_chat():
    """
    Procesa una pregunta usando el agente ManhwaBot (LangChain + Groq).

    Body JSON:
        {
            "pregunta": "¿Qué manhwas de acción me recomiendas?",
            "user_id": "usuario_123"   (opcional, default: "default")
        }

    Response JSON:
        { "respuesta": "...", "user_id": "..." }
    """
    data = request.get_json()
    if not data or "pregunta" not in data:
        return jsonify({"error": "No se recibió la pregunta"}), 400

    pregunta = data["pregunta"]
    user_id = data.get("user_id", "default")

    try:
        from agent import get_agent
        agente = get_agent(user_id)
        respuesta = agente.chat(pregunta)
        return jsonify({"respuesta": respuesta, "user_id": user_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Endpoint de estado
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "2.0-agent"})


if __name__ == "__main__":
    app.run(debug=True)
