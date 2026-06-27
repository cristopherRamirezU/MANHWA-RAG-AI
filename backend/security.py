"""
Módulo de seguridad y uso responsable para ManhwaBot — EP3 (IL3.3).

Protocolos implementados:

1. VALIDACIÓN DE ENTRADA (OWASP LLM01 — Prompt Injection)
   - Límite de longitud (500 chars) para prevenir inyección masiva de instrucciones
   - Blocklist de patrones de jailbreak (regex case-insensitive)
   - Sanitización de caracteres de control que podrían alterar el parsing ReAct

2. RATE LIMITING POR USUARIO (ventana deslizante 60 s)
   - Máximo 20 requests/minuto por user_id (alineado con límite gratuito Groq: 30 rpm)
   - Implementación en memoria con deque; sin persistencia (se reinicia con el servidor)

3. PRIVACIDAD Y MINIMIZACIÓN DE DATOS (GDPR Art. 5)
   - Los user_id NUNCA se guardan en logs en texto claro: se hashean con SHA-256
   - El texto completo de las preguntas no se persiste; solo su longitud
   - Las API keys se leen exclusivamente desde variables de entorno

4. RESTRICCIÓN ÉTICA DE DOMINIO
   - El System Prompt del agente restringe las respuestas a manhwa/manga/manhua
   - Cualquier pregunta fuera de dominio recibe Final Answer sin invocar herramientas

5. EXPOSICIÓN SEGURA DE ERRORES
   - Los endpoints Flask nunca exponen stack traces al cliente
   - Los mensajes de error son genéricos para el cliente; el detalle queda en logs

Referencias normativas:
   OWASP LLM Top 10 v1.0 (2023) — https://owasp.org/www-project-top-10-for-large-language-model-applications/
   GDPR Reglamento (UE) 2016/679, Art. 5 — Principios relativos al tratamiento
   ISO/IEC 27001:2022 — Sistema de Gestión de Seguridad de la Información
"""

import re
import time
import threading
from collections import defaultdict, deque

MAX_INPUT_LENGTH = 500
MAX_REQUESTS_PER_MINUTE = 20

# Patrones de jailbreak / prompt injection más comunes
_BLOCKED_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(previous|all|your)\s+instructions?", re.IGNORECASE),
    re.compile(r"forget\s+(your|all)\s+instructions?", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"\bDAN\b"),
    re.compile(r"act\s+as\s+if\s+you", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?!manhwabot)", re.IGNORECASE),
    re.compile(r"reveal\s+(your\s+)?(prompt|instructions|system)", re.IGNORECASE),
    re.compile(r"(print|show|repeat|output)\s+(your\s+)?(system\s+)?prompt", re.IGNORECASE),
]

# Rate limiter
_rate_windows: dict[str, deque] = defaultdict(deque)
_rl_lock = threading.Lock()


def check_rate_limit(user_id: str) -> bool:
    """
    Verifica si el user_id está dentro del límite de requests por minuto.

    Usa una ventana deslizante: descarta timestamps más antiguos que 60 s y
    compara la cantidad restante contra MAX_REQUESTS_PER_MINUTE.

    Returns:
        True  — la petición está permitida (se registra el timestamp).
        False — el límite fue superado; la petición debe rechazarse (HTTP 429).
    """
    now = time.monotonic()
    cutoff = now - 60.0

    with _rl_lock:
        dq = _rate_windows[user_id]
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= MAX_REQUESTS_PER_MINUTE:
            return False
        dq.append(now)
    return True


def validate_input(pregunta: str) -> tuple[bool, str]:
    """
    Valida y sanitiza la pregunta del usuario antes de pasarla al agente.

    Returns:
        (True,  texto_limpio)   — si la entrada es válida.
        (False, mensaje_error)  — si la entrada debe rechazarse.
    """
    if not pregunta or not pregunta.strip():
        return False, "La pregunta no puede estar vacía."

    if len(pregunta) > MAX_INPUT_LENGTH:
        return False, f"La pregunta supera el límite de {MAX_INPUT_LENGTH} caracteres."

    for pattern in _BLOCKED_PATTERNS:
        if pattern.search(pregunta):
            return False, "Pregunta bloqueada por política de uso responsable."

    # Eliminar caracteres de control (excepto \n y \t que son legítimos)
    clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", pregunta)
    return True, clean.strip()


def sanitize_user_id(user_id: str) -> str:
    """
    Normaliza el user_id: solo alfanuméricos, guiones y guiones bajos.
    Previene path traversal al construir rutas de archivos de sesión.
    """
    clean = re.sub(r"[^a-zA-Z0-9_\-]", "", user_id)
    return clean[:64] or "default"
