"""
Módulo de observabilidad para ManhwaBot — EP3 (IL3.1, IL3.2).

Métricas recolectadas por request (requests.jsonl):
  - timestamp       : Momento ISO 8601 de la petición
  - user_id_hash    : SHA-256[:12] del user_id para privacidad (nunca se guarda en claro)
  - pregunta_len    : Longitud en caracteres de la pregunta
  - respuesta_len   : Longitud en caracteres de la respuesta
  - latencia_ms     : Tiempo total de respuesta extremo a extremo (ms)
  - iteraciones     : Ciclos ReAct ejecutados (1–MAX_ITERATIONS)
  - herramientas    : Lista ordenada de herramientas invocadas en el ciclo
  - tokens_entrada  : Prompt tokens acumulados de todas las llamadas al LLM
  - tokens_salida   : Completion tokens acumulados
  - tokens_total    : Suma de entrada + salida
  - exito           : True si se alcanzó Final Answer; False si timeout o excepción
  - consistencia    : Similitud Jaccard respecto a respuesta anterior a la misma query
  - error           : Mensaje de excepción (None si exitoso)

Métricas por llamada a herramienta (tools.jsonl):
  - timestamp       : Momento de la llamada
  - herramienta     : Nombre de la herramienta invocada
  - input_len       : Longitud del parámetro de entrada
  - resultado_len   : Longitud del texto retornado
  - latencia_ms     : Tiempo de ejecución de la herramienta (ms)
  - exito           : True si no retornó mensaje de error de dispatcher
"""

import hashlib
import json
import os
import threading
from datetime import datetime

LOGS_DIR = os.path.join(os.path.dirname(__file__), "data", "logs")
REQUESTS_LOG = os.path.join(LOGS_DIR, "requests.jsonl")
TOOLS_LOG = os.path.join(LOGS_DIR, "tools.jsonl")

_write_lock = threading.Lock()
_consistency_cache: dict[str, str] = {}
_cache_lock = threading.Lock()

os.makedirs(LOGS_DIR, exist_ok=True)


def _append(path: str, record: dict) -> None:
    record["timestamp"] = datetime.now().isoformat()
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def _hash_user(user_id: str) -> str:
    """Anonimiza el user_id con SHA-256 truncado a 12 hex chars (IE6 — privacidad)."""
    return hashlib.sha256(user_id.encode()).hexdigest()[:12]


def compute_consistency(query: str, response: str) -> float | None:
    """
    Calcula la consistencia de respuesta mediante similitud Jaccard a nivel de palabras.

    Compara 'response' con la última respuesta registrada para la misma 'query'.
    Retorna None en la primera aparición (sin referencia previa).

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|,  donde A y B son conjuntos de tokens.
    """
    key = query.lower().strip()
    with _cache_lock:
        prev = _consistency_cache.get(key)
        _consistency_cache[key] = response

    if prev is None:
        return None

    tokens_prev = set(prev.lower().split())
    tokens_curr = set(response.lower().split())
    union = tokens_prev | tokens_curr
    if not union:
        return 1.0
    return round(len(tokens_prev & tokens_curr) / len(union), 3)


def log_request(
    user_id: str,
    pregunta: str,
    respuesta: str,
    latencia_ms: float,
    iteraciones: int,
    herramientas: list,
    tokens_entrada: int,
    tokens_salida: int,
    exito: bool,
    consistencia: float | None = None,
    error: str | None = None,
) -> None:
    """Persiste las métricas de un request completo en JSONL."""
    _append(REQUESTS_LOG, {
        "user_id_hash": _hash_user(user_id),
        "pregunta_len": len(pregunta),
        "respuesta_len": len(respuesta),
        "latencia_ms": round(latencia_ms, 2),
        "iteraciones": iteraciones,
        "herramientas": herramientas,
        "tokens_entrada": tokens_entrada,
        "tokens_salida": tokens_salida,
        "tokens_total": tokens_entrada + tokens_salida,
        "exito": exito,
        "consistencia": consistencia,
        "error": error,
    })


def log_tool_call(
    herramienta: str,
    tool_input: str,
    resultado_len: int,
    latencia_ms: float,
    exito: bool,
) -> None:
    """Persiste las métricas de una llamada individual a herramienta en JSONL."""
    _append(TOOLS_LOG, {
        "herramienta": herramienta,
        "input_len": len(tool_input),
        "resultado_len": resultado_len,
        "latencia_ms": round(latencia_ms, 2),
        "exito": exito,
    })
