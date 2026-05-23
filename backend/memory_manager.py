"""
Gestor de memoria de corto y largo plazo para ManhwaBot (IE3).

Sin dependencias de LangChain — implementación directa en Python puro.

Memoria de corto plazo:  ventana deslizante de las últimas k interacciones
                         almacenada en RAM durante la sesión.
Memoria de largo plazo:  archivo JSON por usuario en disco, persiste entre
                         reinicios del servidor.
"""

import json
import os
from datetime import datetime
from collections import deque


SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "data", "sessions")


class MemoryManager:
    def __init__(self, user_id: str = "default", window_size: int = 5):
        self.user_id = user_id
        self.window_size = window_size
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        self.long_term_file = os.path.join(SESSIONS_DIR, f"{user_id}.json")

        # Corto plazo: deque con límite automático (descarta el más antiguo)
        self._window: deque[dict] = deque(maxlen=window_size)

        # Largo plazo: cargado desde disco
        self.long_term = self._load_long_term()

    # ------------------------------------------------------------------
    # MEMORIA DE CORTO PLAZO
    # ------------------------------------------------------------------

    def save_exchange(self, user_input: str, agent_output: str):
        """Registra un intercambio en la ventana de corto plazo."""
        self._window.append({"human": user_input, "ai": agent_output})

    def get_short_term_history(self) -> str:
        """Devuelve el historial de corto plazo como texto plano."""
        if not self._window:
            return ""
        lines = []
        for turn in self._window:
            lines.append(f"Human: {turn['human']}")
            lines.append(f"AI: {turn['ai']}")
        return "\n".join(lines)

    def clear_short_term(self):
        """Vacía la memoria de corto plazo."""
        self._window.clear()

    # ------------------------------------------------------------------
    # MEMORIA DE LARGO PLAZO
    # ------------------------------------------------------------------

    def _load_long_term(self) -> dict:
        if os.path.exists(self.long_term_file):
            with open(self.long_term_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"preferencias": [], "historial": [], "resumen_sesiones": []}

    def _save_long_term(self):
        with open(self.long_term_file, "w", encoding="utf-8") as f:
            json.dump(self.long_term, f, ensure_ascii=False, indent=2)

    def add_to_long_term(self, categoria: str, item: dict):
        """Agrega un elemento a una categoría de la memoria de largo plazo."""
        if categoria not in self.long_term:
            self.long_term[categoria] = []
        self.long_term[categoria].append(item)
        self._save_long_term()

    def save_preference(self, titulo_y_rating: str) -> str:
        """Guarda una preferencia del usuario. Formato esperado: 'titulo: rating'."""
        try:
            partes = titulo_y_rating.split(":")
            titulo = partes[0].strip()
            rating = partes[1].strip() if len(partes) > 1 else "N/A"

            self.add_to_long_term("preferencias", {
                "titulo": titulo,
                "rating": rating,
                "fecha": datetime.now().strftime("%Y-%m-%d"),
            })
            self.add_to_long_term("historial", {
                "titulo": titulo,
                "fecha": datetime.now().strftime("%Y-%m-%d"),
            })
            return f"Preferencia guardada: '{titulo}' con calificación {rating}/10."
        except Exception as e:
            return f"Error al guardar preferencia: {e}"

    def get_history_summary(self) -> str:
        """Devuelve un resumen legible del historial y preferencias del usuario."""
        lm = self.long_term
        if not lm.get("historial") and not lm.get("preferencias"):
            return "No hay historial de lectura guardado para este usuario."

        lines = []
        if lm.get("historial"):
            lines.append("Historial de lectura (últimas 5 entradas):")
            for h in lm["historial"][-5:]:
                lines.append(f"  - {h.get('titulo', 'N/A')} ({h.get('fecha', '?')})")

        if lm.get("preferencias"):
            lines.append("\nCalificaciones guardadas:")
            for p in lm["preferencias"][-5:]:
                lines.append(f"  - {p.get('titulo', 'N/A')}: {p.get('rating', 'N/A')}/10")

        return "\n".join(lines)
