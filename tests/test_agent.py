"""
Pruebas de demostración del agente ManhwaBot (IE6 — toma de decisiones).

Cada escenario muestra cómo el agente ADAPTA su comportamiento según
la intención del usuario y las condiciones del entorno:

  Escenario 1 — Consulta simple:        activa BuscarManhwa
  Escenario 2 — Búsqueda semántica:     activa BuscarManhwa con query libre
  Escenario 3 — Recomendación:          activa RecomendarPorGenero
  Escenario 4 — Guardar preferencia:    activa GuardarPreferencia
  Escenario 5 — Historial:              activa VerHistorialUsuario
  Escenario 6 — Multi-paso:             combina búsqueda + guardar preferencia
  Escenario 7 — Continuidad (memoria):  usa historial de corto plazo

Uso:
    cd backend
    python ../tests/test_agent.py
"""

import sys
import os

# Permite importar desde backend/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from agent import ManhwaAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEPARATOR = "\n" + "=" * 70 + "\n"


def run_scenario(agent: ManhwaAgent, titulo: str, pregunta: str):
    print(f"{SEPARATOR}ESCENARIO: {titulo}")
    print(f"USUARIO:   {pregunta}\n")
    respuesta = agent.chat(pregunta)
    print(f"\nAGENTE:    {respuesta}")


# ---------------------------------------------------------------------------
# Escenarios
# ---------------------------------------------------------------------------

def main():
    print("\n🤖 ManhwaBot — Demostración de Toma de Decisiones del Agente")
    print("=" * 70)

    # Crear agente con user_id de prueba
    agent = ManhwaAgent(user_id="test_demo")

    # ------------------------------------------------------------------
    # Escenario 1: Consulta directa por título conocido
    # Decisión esperada: BuscarManhwa → fuente local
    # ------------------------------------------------------------------
    run_scenario(
        agent,
        "1 — Consulta directa (BD local)",
        "¿De qué trata Solo Leveling?",
    )

    # ------------------------------------------------------------------
    # Escenario 2: Búsqueda semántica — el usuario no da el título exacto
    # Decisión esperada: BuscarManhwa → búsqueda semántica → BD local o Jikan
    # ------------------------------------------------------------------
    run_scenario(
        agent,
        "2 — Búsqueda semántica (IE4)",
        "Busco un manhwa sobre un chico débil que se vuelve muy poderoso",
    )

    # ------------------------------------------------------------------
    # Escenario 3: Recomendación por género
    # Decisión esperada: RecomendarPorGenero
    # ------------------------------------------------------------------
    run_scenario(
        agent,
        "3 — Recomendación por género",
        "¿Qué manhwas de Terror puedes recomendarme?",
    )

    # ------------------------------------------------------------------
    # Escenario 4: El usuario califica un manhwa
    # Decisión esperada: GuardarPreferencia
    # ------------------------------------------------------------------
    run_scenario(
        agent,
        "4 — Guardar preferencia (memoria LP)",
        "Quiero guardar que Solo Leveling me gustó mucho, le doy un 10",
    )

    # ------------------------------------------------------------------
    # Escenario 5: Ver historial
    # Decisión esperada: VerHistorialUsuario → devuelve lo guardado antes
    # ------------------------------------------------------------------
    run_scenario(
        agent,
        "5 — Recuperar historial (memoria LP)",
        "¿Qué manhwas he guardado en mi historial?",
    )

    # ------------------------------------------------------------------
    # Escenario 6: Consulta multi-paso
    # Decisión esperada: BuscarManhwa + luego GuardarPreferencia
    # ------------------------------------------------------------------
    run_scenario(
        agent,
        "6 — Flujo multi-paso (planificación IE5)",
        "Busca información de Sweet Home y guárdalo en mi historial con nota 8",
    )

    # ------------------------------------------------------------------
    # Escenario 7: Continuidad de memoria de corto plazo
    # El agente debe recordar la conversación anterior sin que el usuario repita
    # ------------------------------------------------------------------
    run_scenario(
        agent,
        "7 — Continuidad de contexto (memoria CP)",
        "¿Y ese manhwa tiene buena calificación según MyAnimeList?",
    )

    print(SEPARATOR)
    print("✅ Demostración completada. Revisa los pasos 'Pensamiento / Acción / Observación'")
    print("   para verificar la lógica de planificación y toma de decisiones del agente.\n")


if __name__ == "__main__":
    main()
