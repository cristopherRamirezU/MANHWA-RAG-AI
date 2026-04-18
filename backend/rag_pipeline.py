import json
import requests
from transformers import MarianMTModel, MarianTokenizer

# ================================
# MODELO DE TRADUCCIÓN
# ================================
modelo_nombre = "Helsinki-NLP/opus-mt-en-es"
tokenizer = MarianTokenizer.from_pretrained(modelo_nombre)
model = MarianMTModel.from_pretrained(modelo_nombre)

# ================================
# DATOS INTERNOS
# ================================
def cargar_datos():
    with open("data/manhwas.json", "r", encoding="utf-8") as f:
        return json.load(f)
# ================================
# Limpieza en la pregunta
# ================================
def limpiar_pregunta(pregunta):
    pregunta = pregunta.lower()

    palabras_basura = [
        "de que se trata",
        "qué es",
        "que es",
        "dime sobre",
        "informacion de",
        "valoracion",
        "nota",
        "puntuacion"
    ]

    for p in palabras_basura:
        pregunta = pregunta.replace(p, "")

    return pregunta.strip()
# ================================
# API JIKAN
# ================================
def buscar_en_jikan(pregunta):
    try:
        pregunta_limpia = limpiar_pregunta(pregunta)

        url = f"https://api.jikan.moe/v4/manga?q={pregunta_limpia}"
        res = requests.get(url).json()

        if res["data"]:
            manga = res["data"][0]

            return {
                "titulo": manga["title"],
                "sinopsis": manga["synopsis"],
                "tipo": manga["type"],
                "score": manga["score"],
                "imagen": manga["images"]["jpg"]["image_url"]
            }

    except Exception as e:
        print("ERROR JIKAN:", e)

    return None
# ================================
# RAG
# ================================
def buscar_contexto(pregunta):
    datos = cargar_datos()

    for item in datos:
        if item["titulo"].lower() in pregunta.lower():
            return {
                "titulo": item["titulo"],
                "sinopsis": item["resumen"],
                "tipo": "manhwa",
                "score": "N/A",
                "imagen": ""
            }

    return buscar_en_jikan(pregunta)

# ================================
# MAPEAR TIPO
# ================================
def mapear_tipo(tipo_api):
    if tipo_api:
        tipo_api = tipo_api.lower()

        if "manga" in tipo_api:
            return "manga"
        elif "manhwa" in tipo_api:
            return "manhwa"
        elif "manhua" in tipo_api:
            return "manhua"

    return "contenido"

# ================================
# TRADUCCIÓN
# ================================
def traducir_texto(texto):
    tokens = tokenizer(texto, return_tensors="pt", padding=True, truncation=True)
    traduccion = model.generate(**tokens)
    return tokenizer.decode(traduccion[0], skip_special_tokens=True)

# ================================
# DETECTAR INTENCIÓN
# ================================
def detectar_intencion(pregunta):
    pregunta = pregunta.lower()

    if "nota" in pregunta or "valoracion" in pregunta or "puntuacion" in pregunta:
        return "valoracion"

    return "general"

# ================================
# GENERADOR FINAL
# ================================
def generar_respuesta(pregunta, contexto):
    if not contexto:
        return "No se encontró información sobre ese contenido."

    # Tipo
    tipo = mapear_tipo(contexto.get("tipo"))

    # Intención
    intencion = detectar_intencion(pregunta)

    # Traducción
    descripcion = traducir_texto(contexto.get("sinopsis", ""))

    titulo = contexto.get("titulo", "Desconocido")
    score = contexto.get("score", "N/A")
    imagen = contexto.get("imagen", "")

    # RESPUESTA CON VALORACIÓN
    if intencion == "valoracion":
        return f"""
Según nuestra información, el {tipo} "{titulo}" tiene una valoración de {score}.

Descripción:
{descripcion}

Imagen:
{imagen}
"""

    # RESPUESTA NORMAL
    return f"""
Buscando en nuestra información, el {tipo} "{titulo}" se trata de:

{descripcion}

Imagen:
{imagen}
"""