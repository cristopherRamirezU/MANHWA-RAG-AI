"""
Herramientas del agente ManhwaBot (IE1).

Funciones Python puras — sin dependencias de LangChain.
El agente (agent.py) las invoca directamente según las decisiones del LLM.
"""

import json
import os
import requests


DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "manhwas.json")
JIKAN_BASE = "https://api.jikan.moe/v4"


def _cargar_manhwas() -> list:
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _buscar_local_semantico(query: str) -> list:
    """Búsqueda semántica en BD local con fallback a coincidencia de texto."""
    try:
        from semantic_search import get_engine
        return get_engine().search(query, top_k=3, threshold=0.25)
    except Exception:
        manhwas = _cargar_manhwas()
        q = query.lower()
        return [m for m in manhwas if q in m["titulo"].lower() or m["titulo"].lower() in q]


def _buscar_jikan(query: str) -> dict | None:
    """Consulta la API Jikan de MyAnimeList."""
    try:
        resp = requests.get(
            f"{JIKAN_BASE}/manga",
            params={"q": query, "limit": 1},
            timeout=6,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            return None
        manga = data[0]
        return {
            "titulo": manga["title"],
            "tipo": manga.get("type", "N/A"),
            "resumen": manga.get("synopsis") or "Sin sinopsis disponible.",
            "score": manga.get("score", "N/A"),
            "generos": [g["name"] for g in manga.get("genres", [])],
            "imagen": manga.get("images", {}).get("jpg", {}).get("image_url", ""),
            "fuente": "Jikan/MyAnimeList",
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Herramientas públicas — invocadas por agent.py
# ---------------------------------------------------------------------------

def fn_buscar_manhwa(query: str) -> str:
    """Busca información sobre un título en BD local (semántica) y Jikan API."""
    resultados = _buscar_local_semantico(query)
    if resultados:
        r = resultados[0]
        return json.dumps({
            "fuente": "base_de_datos_local",
            "titulo": r["titulo"],
            "genero": r.get("genero", []),
            "resumen": r["resumen"],
            "score": r.get("score", "N/A"),
            "similitud_semantica": r.get("similitud", "N/A"),
        }, ensure_ascii=False, indent=2)

    resultado_jikan = _buscar_jikan(query)
    if resultado_jikan:
        return json.dumps(resultado_jikan, ensure_ascii=False, indent=2)

    return f"No se encontró información sobre '{query}'."


def fn_recomendar_por_genero(genero: str) -> str:
    """Filtra la BD local y devuelve títulos del género solicitado."""
    manhwas = _cargar_manhwas()
    genero_lower = genero.lower()
    encontrados = []
    for m in manhwas:
        generos = m.get("genero", [])
        texto = " ".join(generos).lower() if isinstance(generos, list) else str(generos).lower()
        if genero_lower in texto:
            encontrados.append(
                f"• {m['titulo']} (Score: {m.get('score', 'N/A')}) — {m['resumen'][:120]}..."
            )

    if encontrados:
        return f"Recomendaciones de género '{genero}':\n" + "\n".join(encontrados)
    return (
        f"No se encontraron títulos del género '{genero}'. "
        "Géneros disponibles: Acción, Fantasía, Drama, Romance, Terror, "
        "Deportes, Sobrenatural, Escolar, Comedia, Reencarnación, Supervivencia."
    )
