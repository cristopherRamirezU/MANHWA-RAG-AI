from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_pipeline import buscar_contexto, generar_respuesta

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    if not data or "pregunta" not in data:
        return jsonify({"error": "No se recibió la pregunta"}), 400

    pregunta = data["pregunta"]
    print("Pregunta recibida:", pregunta)

    contexto = buscar_contexto(pregunta)
    respuesta = generar_respuesta(pregunta, contexto)

    return jsonify({"respuesta": respuesta})

if __name__ == "__main__":
    app.run(debug=True)