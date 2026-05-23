"""
Agente ManhwaBot — ReAct por texto (IE2, IE5, IE6).

Implementa el patrón ReAct (Reasoning + Acting) mediante parsing de texto,
sin depender de function calling del modelo. Funciona con cualquier LLM de Groq.

Ciclo ReAct:
  1. LLM genera: Thought → Action → Action Input
  2. Sistema ejecuta la herramienta e inyecta: Observation
  3. LLM continúa hasta generar: Final Answer
"""

import os
import re
from dotenv import load_dotenv
from groq import Groq

from memory_manager import MemoryManager
from tools import fn_buscar_manhwa, fn_recomendar_por_genero

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

    def chat(self, user_message: str) -> str:
        """Ejecuta el loop ReAct y devuelve la respuesta final."""
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
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
                stop=["Observation:"],          # Detiene antes de que el modelo escriba la observación
            )

            text = (response.choices[0].message.content or "").strip()
            print(f"\n[Agente iter {iteration + 1}]\n{text}\n")

            # ── Respuesta final detectada ──────────────────────────────
            if "Final Answer:" in text:
                final = text.split("Final Answer:", 1)[1].strip()
                self.memory.save_exchange(user_message, final)
                return final

            # ── Parsear Action y Action Input ──────────────────────────
            action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", text)
            input_match  = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text, re.DOTALL)

            if not action_match:
                # El modelo no usó el formato — devolver texto como respuesta
                self.memory.save_exchange(user_message, text)
                return text

            action      = action_match.group(1).strip()
            tool_input  = input_match.group(1).strip() if input_match else ""

            # ── Ejecutar herramienta ───────────────────────────────────
            result = _dispatch(action, tool_input, self.memory)
            print(f"[Tool] {action}({tool_input!r}) → {result[:200]}")

            # ── Inyectar observación y continuar el loop ───────────────
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user",      "content": f"Observation: {result}"})

        output = "No pude completar la respuesta tras varios intentos."
        self.memory.save_exchange(user_message, output)
        return output


# ---------------------------------------------------------------------------
# Caché de agentes por sesión
# ---------------------------------------------------------------------------

_agents: dict[str, ManhwaAgent] = {}


def get_agent(user_id: str = "default") -> ManhwaAgent:
    if user_id not in _agents:
        _agents[user_id] = ManhwaAgent(user_id)
    return _agents[user_id]
