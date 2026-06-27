"""
Agente ManhwaBot — ReAct por texto (IE2, IE5, IE6).

Implementa el patrón ReAct (Reasoning + Acting) mediante parsing de texto,
sin depender de function calling del modelo. Funciona con cualquier LLM de Groq.

Ciclo ReAct:
  1. LLM genera: Thought → Action → Action Input
  2. Sistema ejecuta la herramienta e inyecta: Observation
  3. LLM continúa hasta generar: Final Answer

EP3 — Instrumentación de observabilidad:
  - Latencia total del request (ms)
  - Latencia por llamada a herramienta (ms)
  - Conteo de iteraciones ReAct
  - Tokens de entrada y salida acumulados (de la API Groq)
  - Herramientas invocadas por orden
  - Consistencia de respuesta (Jaccard vs última respuesta a la misma query)
  - Registro de errores con mensaje
"""

import os
import re
import time
from dotenv import load_dotenv
from groq import Groq

from memory_manager import MemoryManager
from tools import fn_buscar_manhwa, fn_recomendar_por_genero
import observability

load_dotenv()

# ---------------------------------------------------------------------------
# Prompt del sistema — define el formato ReAct explícitamente
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Eres ManhwaBot, un asistente experto en manhwa, manga y manhua.

HERRAMIENTAS DISPONIBLES:
- buscar_manhwa     : Busca información sobre un título específico.
                      Input: nombre o descripción del título.
- recomendar_por_genero : Recomienda títulos por género o tema.
                      Input: nombre del género (Acción, Fantasía, Drama, Romance,
                      Terror, Deportes, Sobrenatural, Escolar, Comedia, Supervivencia).
- ver_historial     : Muestra el historial y calificaciones del usuario.
                      Input: "historial"
- guardar_preferencia : Guarda la calificación del usuario sobre un título.
                      Input: "Titulo: calificacion"  (ej: "Solo Leveling: 9")

USA SIEMPRE ESTE FORMATO (sin excepciones):

Thought: [razona qué necesitas hacer]
Action: [nombre exacto de la herramienta]
Action Input: [parámetro para la herramienta]

Cuando tengas suficiente información para responder:
Thought: [conclusión]
Final Answer: [respuesta completa en español para el usuario]

REGLAS IMPORTANTES:
- Empieza SIEMPRE con "Thought:"
- Usa SOLO los nombres de herramienta escritos arriba (exactos)
- Responde siempre en español
- NO escribas "Observation:" — eso lo agrega el sistema automáticamente
- Si la pregunta NO es sobre manhwa, manga o manhua, responde con Final Answer así:
  "Solo puedo ayudarte con manhwa, manga y manhua. ¿Te interesa que busque algo relacionado?"
- NUNCA respondas preguntas sobre anime, películas, series u otros temas fuera de tu especialidad
"""

# ---------------------------------------------------------------------------
# Dispatcher: nombre de herramienta → función Python
# ---------------------------------------------------------------------------

def _dispatch(tool_name: str, tool_input: str, memory: MemoryManager) -> str:
    name = tool_name.strip().lower().replace("-", "_").replace(" ", "_")
    if name == "buscar_manhwa":
        return fn_buscar_manhwa(tool_input)
    if name in ("recomendar_por_genero", "recomendar"):
        return fn_recomendar_por_genero(tool_input)
    if name in ("ver_historial", "ver_historial_usuario", "historial"):
        return memory.get_history_summary()
    if name in ("guardar_preferencia", "guardar"):
        return memory.save_preference(tool_input)
    return (
        f"Herramienta '{tool_name}' no reconocida. "
        "Usa: buscar_manhwa, recomendar_por_genero, ver_historial, guardar_preferencia."
    )


# ---------------------------------------------------------------------------
# Clase principal del agente
# ---------------------------------------------------------------------------

class ManhwaAgent:
    MODEL = "llama-3.3-70b-versatile"
    MAX_ITERATIONS = 6

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.memory = MemoryManager(user_id)
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self._response_cache: dict[str, str] = {}  # para cálculo de consistencia

    def chat(self, user_message: str) -> str:
        """
        Ejecuta el loop ReAct y devuelve la respuesta final.
        Instrumentado con métricas de observabilidad (EP3).
        """
        start_time = time.monotonic()
        tools_used: list[str] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        iterations_done = 0
        final_response: str | None = None
        error_msg: str | None = None

        try:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            # Historial de corto plazo
            history = self.memory.get_short_term_history()
            if history:
                for line in history.strip().split("\n"):
                    if line.startswith("Human:"):
                        messages.append({"role": "user", "content": line[6:].strip()})
                    elif line.startswith("AI:"):
                        messages.append({"role": "assistant", "content": line[3:].strip()})

            messages.append({"role": "user", "content": user_message})

            for iteration in range(self.MAX_ITERATIONS):
                iterations_done = iteration + 1

                response = self.client.chat.completions.create(
                    model=self.MODEL,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1024,
                    stop=["Observation:"],
                )

                # Acumular tokens de cada llamada al LLM
                if response.usage:
                    total_prompt_tokens += response.usage.prompt_tokens
                    total_completion_tokens += response.usage.completion_tokens

                text = (response.choices[0].message.content or "").strip()
                print(f"\n[Agente iter {iteration + 1}]\n{text}\n")

                # ── Respuesta final detectada ──────────────────────────
                if "Final Answer:" in text:
                    final_response = text.split("Final Answer:", 1)[1].strip()
                    self.memory.save_exchange(user_message, final_response)
                    break

                # ── Parsear Action y Action Input ──────────────────────
                action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", text)
                input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text, re.DOTALL)

                if not action_match:
                    # Modelo no siguió el formato — devolver texto tal cual
                    final_response = text
                    self.memory.save_exchange(user_message, final_response)
                    break

                action = action_match.group(1).strip()
                tool_input = input_match.group(1).strip() if input_match else ""

                # ── Ejecutar herramienta con medición de latencia ──────
                tool_start = time.monotonic()
                result = _dispatch(action, tool_input, self.memory)
                tool_latency_ms = (time.monotonic() - tool_start) * 1000

                tools_used.append(action)
                is_tool_success = not result.startswith("Herramienta '")

                observability.log_tool_call(
                    herramienta=action,
                    tool_input=tool_input,
                    resultado_len=len(result),
                    latencia_ms=tool_latency_ms,
                    exito=is_tool_success,
                )

                print(f"[Tool] {action}({tool_input!r}) → {result[:200]}")

                # ── Inyectar observación y continuar el loop ───────────
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": f"Observation: {result}"})

            if final_response is None:
                final_response = "No pude completar la respuesta tras varios intentos."
                self.memory.save_exchange(user_message, final_response)

        except Exception as exc:
            error_msg = str(exc)
            final_response = "Ocurrió un error al procesar tu pregunta. Intenta de nuevo."
            print(f"[Agente ERROR] {error_msg}")

        # ── Métricas de observabilidad ─────────────────────────────────
        latencia_ms = (time.monotonic() - start_time) * 1000
        exito = error_msg is None and final_response != "No pude completar la respuesta tras varios intentos."

        consistencia = observability.compute_consistency(user_message, final_response)

        observability.log_request(
            user_id=self.user_id,
            pregunta=user_message,
            respuesta=final_response,
            latencia_ms=latencia_ms,
            iteraciones=iterations_done,
            herramientas=tools_used,
            tokens_entrada=total_prompt_tokens,
            tokens_salida=total_completion_tokens,
            exito=exito,
            consistencia=consistencia,
            error=error_msg,
        )

        return final_response


# ---------------------------------------------------------------------------
# Caché de agentes por sesión
# ---------------------------------------------------------------------------

_agents: dict[str, ManhwaAgent] = {}


def get_agent(user_id: str = "default") -> ManhwaAgent:
    if user_id not in _agents:
        _agents[user_id] = ManhwaAgent(user_id)
    return _agents[user_id]
