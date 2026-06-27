"""
Generador de datos de demostración para el dashboard de observabilidad EP3.

Crea datos sintéticos pero realistas en:
  - backend/data/logs/requests.jsonl
  - backend/data/logs/tools.jsonl

Los datos simulan 7 días de operación con:
  - 80 requests distribuidos con variabilidad realista
  - Latencia basada en distribución log-normal (media ~1.4 s, algunos outliers)
  - 92% de tasa de éxito
  - Distribución de herramientas representativa del uso real
  - Queries repetidas para generar consistencia no perfecta
  - Algunos errores reales (timeout API, excepción en herramienta)

Uso:
    python tests/seed_metrics.py
    streamlit run dashboard/streamlit_app.py
"""

import json
import os
import random
import sys
import hashlib
from datetime import datetime, timedelta

# Añadir backend/ al path para importar constantes si es necesario
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "backend", "data", "logs")
REQUESTS_LOG = os.path.join(LOGS_DIR, "requests.jsonl")
TOOLS_LOG = os.path.join(LOGS_DIR, "tools.jsonl")

random.seed(42)

# ── Datos de simulación ─────────────────────────────────────────────────────

TOOL_PROFILES = {
    "buscar_manhwa": {
        "peso": 0.38,
        "latencia_base_ms": 280,
        "latencia_std_ms": 120,
        "tasa_exito": 0.95,
    },
    "recomendar_por_genero": {
        "peso": 0.32,
        "latencia_base_ms": 45,
        "latencia_std_ms": 20,
        "tasa_exito": 0.98,
    },
    "ver_historial": {
        "peso": 0.17,
        "latencia_base_ms": 12,
        "latencia_std_ms": 5,
        "tasa_exito": 0.99,
    },
    "guardar_preferencia": {
        "peso": 0.13,
        "latencia_base_ms": 18,
        "latencia_std_ms": 8,
        "tasa_exito": 0.97,
    },
}

QUERIES = [
    "¿De qué trata Solo Leveling?",
    "Recomiéndame manhwas de terror",
    "Busco un manhwa de acción donde el protagonista sube de nivel",
    "¿Qué es Tower of God?",
    "¿Cuál es la sinopsis de Sweet Home?",
    "Recomiéndame manhwas de romance",
    "¿Tengo algo en mi historial?",
    "Busco manhwas de fantasía",
    "Solo Leveling me encantó, le doy un 10",
    "¿Cuál es el score de Omniscient Reader?",
    "Recomiéndame manhwas de drama escolar",
    "Busco algo de vampiros o sobrenatural",
    "¿De qué trata Lookism?",
    "Guarda que vi Tower of God",
    "¿Qué manhwas de deportes hay?",
    "¿De qué trata Eleceed?",
    "Recomiéndame algo de ciclismo o deporte",
    "¿Hay algo parecido a Solo Leveling?",
    "Muéstrame mis preferencias guardadas",
    "¿Qué es Noblesse?",
]

USERS = ["user_cristopher", "user_pareja", "default", "user_test"]

ERRORS = [
    "ConnectionError: API Groq no disponible (timeout 6s)",
    "RateLimitError: 30 rpm exceeded — retry after 60s",
    None, None, None,  # La mayoría son None (sin error)
]

# ── Generadores ─────────────────────────────────────────────────────────────

def gen_latencia_ms(iteraciones: int, tools_count: int) -> float:
    """Latencia realista basada en iteraciones y herramientas usadas."""
    base = 800 + iteraciones * 350 + tools_count * 260
    noise = random.gauss(0, base * 0.25)
    return max(200, base + noise)


def gen_tokens(iteraciones: int) -> tuple[int, int]:
    """Tokens de entrada y salida estimados."""
    entrada = int(random.gauss(400 + iteraciones * 80, 60))
    salida = int(random.gauss(180 + iteraciones * 30, 40))
    return max(100, entrada), max(50, salida)


def pick_tools(n_iter: int) -> list[str]:
    """Selecciona herramientas basándose en los pesos del perfil."""
    if n_iter == 0:
        return []
    tool_names = list(TOOL_PROFILES.keys())
    weights = [TOOL_PROFILES[t]["peso"] for t in tool_names]
    count = random.randint(1, min(n_iter, 3))
    selected = random.choices(tool_names, weights=weights, k=count)
    return selected


def hash_user(user_id: str) -> str:
    return hashlib.sha256(user_id.encode()).hexdigest()[:12]


def jaccard(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    u = ta | tb
    return round(len(ta & tb) / len(u), 3) if u else 1.0


# ── Generación de datos ──────────────────────────────────────────────────────

def generate_request_record(ts: datetime, prev_responses: dict) -> dict:
    query = random.choice(QUERIES)
    user = random.choice(USERS)
    error_candidate = random.choices(ERRORS, weights=[0.04, 0.02, 0.47, 0.47, 0.0])[0]

    if error_candidate:
        # Request con error
        iteraciones = random.randint(1, 3)
        tools = pick_tools(random.randint(0, iteraciones))
        lat = gen_latencia_ms(iteraciones, len(tools)) * random.uniform(1.5, 2.5)
        tokens_in, tokens_out = gen_tokens(iteraciones)
        respuesta = "Ocurrió un error al procesar tu pregunta. Intenta de nuevo."
        return {
            "timestamp": ts.isoformat(),
            "user_id_hash": hash_user(user),
            "pregunta_len": len(query),
            "respuesta_len": len(respuesta),
            "latencia_ms": round(lat, 2),
            "iteraciones": iteraciones,
            "herramientas": tools,
            "tokens_entrada": tokens_in,
            "tokens_salida": tokens_out,
            "tokens_total": tokens_in + tokens_out,
            "exito": False,
            "consistencia": None,
            "error": error_candidate,
        }

    # Request exitoso
    iteraciones = random.choices([1, 2, 3, 4, 5, 6], weights=[15, 35, 30, 12, 5, 3])[0]
    tools = pick_tools(iteraciones)
    lat = gen_latencia_ms(iteraciones, len(tools))
    tokens_in, tokens_out = gen_tokens(iteraciones)

    # Respuesta simulada (longitud realista)
    resp_len = random.randint(120, 450)
    respuesta = "x" * resp_len  # solo la longitud importa para las métricas

    # Consistencia
    key = query.lower().strip()
    if key in prev_responses:
        # Calcular Jaccard aproximado (variación del 10-20%)
        prev_len = prev_responses[key]
        variacion = random.uniform(0.6, 0.95)
        consistencia = round(variacion, 3)
    else:
        consistencia = None

    prev_responses[key] = resp_len

    return {
        "timestamp": ts.isoformat(),
        "user_id_hash": hash_user(user),
        "pregunta_len": len(query),
        "respuesta_len": resp_len,
        "latencia_ms": round(lat, 2),
        "iteraciones": iteraciones,
        "herramientas": tools,
        "tokens_entrada": tokens_in,
        "tokens_salida": tokens_out,
        "tokens_total": tokens_in + tokens_out,
        "exito": True,
        "consistencia": consistencia,
        "error": None,
    }


def generate_tool_record(ts: datetime, tool_name: str) -> dict:
    profile = TOOL_PROFILES[tool_name]
    lat = max(5, random.gauss(profile["latencia_base_ms"], profile["latencia_std_ms"]))
    exito = random.random() < profile["tasa_exito"]
    input_len = random.randint(8, 60)
    result_len = random.randint(80, 500) if exito else random.randint(20, 80)
    return {
        "timestamp": ts.isoformat(),
        "herramienta": tool_name,
        "input_len": input_len,
        "resultado_len": result_len,
        "latencia_ms": round(lat, 2),
        "exito": exito,
    }


def main():
    os.makedirs(LOGS_DIR, exist_ok=True)

    now = datetime.now()
    start = now - timedelta(days=7)

    request_records = []
    tool_records = []
    prev_responses: dict[str, int] = {}

    # Generar timestamps distribuidos en 7 días con horarios de uso realistas
    timestamps = []
    t = start
    while t < now:
        # Mayor actividad entre 10:00 y 23:00
        if 10 <= t.hour <= 23:
            n_in_hour = random.choices([0, 1, 2, 3], weights=[0.3, 0.4, 0.2, 0.1])[0]
        else:
            n_in_hour = random.choices([0, 1], weights=[0.85, 0.15])[0]
        for _ in range(n_in_hour):
            offset_min = random.randint(0, 59)
            ts = t.replace(minute=offset_min, second=random.randint(0, 59))
            if ts < now:
                timestamps.append(ts)
        t += timedelta(hours=1)

    timestamps.sort()

    # Asegurar mínimo 60 registros para gráficos representativos
    while len(timestamps) < 60:
        ts = start + timedelta(seconds=random.randint(0, int((now - start).total_seconds())))
        timestamps.append(ts)
    timestamps.sort()

    for ts in timestamps:
        req = generate_request_record(ts, prev_responses)
        request_records.append(req)

        # Generar registros de herramientas para este request
        for tool_name in req["herramientas"]:
            tool_ts = ts + timedelta(milliseconds=random.randint(50, 300))
            tool_records.append(generate_tool_record(tool_ts, tool_name))

    # Escribir archivos JSONL
    with open(REQUESTS_LOG, "w", encoding="utf-8") as f:
        for r in request_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(TOOLS_LOG, "w", encoding="utf-8") as f:
        for r in tool_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Resumen
    exitosos = sum(1 for r in request_records if r["exito"])
    print(f"\nOK Datos de demostracion generados en:\n   {LOGS_DIR}\n")
    print(f"   requests.jsonl : {len(request_records):>4} registros")
    print(f"   tools.jsonl    : {len(tool_records):>4} registros")
    print(f"\n   Tasa de exito  : {exitosos/len(request_records)*100:.1f}%")
    lat_list = [r["latencia_ms"] for r in request_records]
    print(f"   Latencia prom. : {sum(lat_list)/len(lat_list):.0f} ms")
    print(f"\nEjecuta el dashboard:\n   streamlit run dashboard/streamlit_app.py\n")


if __name__ == "__main__":
    main()
