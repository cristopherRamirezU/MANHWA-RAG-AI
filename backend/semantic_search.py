"""
Motor de búsqueda semántica para recuperación de contexto (IE4).

Usa sentence-transformers con un modelo multilingüe para generar embeddings
de los manhwas y calcular similitud coseno contra la consulta del usuario.
Esto permite encontrar contenido relevante aunque la pregunta no use palabras
exactas del título o resumen.
"""

import json
import os
import numpy as np

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "manhwas.json")

# Cargado de forma lazy para no bloquear el inicio de la app
_engine: "SemanticSearchEngine | None" = None


class SemanticSearchEngine:
    MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, data_file: str = DATA_FILE):
        # Importación diferida para evitar carga pesada en arranque
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.MODEL_NAME)
        self.manhwas = self._load_data(data_file)
        self.embeddings = self._embed_corpus()

    def _load_data(self, path: str) -> list[dict]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _embed_corpus(self) -> np.ndarray:
        """Genera embeddings para todos los manhwas de la base de datos."""
        texts = [
            f"{m['titulo']} {m['resumen']} {' '.join(m.get('genero', []))}"
            for m in self.manhwas
        ]
        return self.model.encode(texts, normalize_embeddings=True)

    def search(self, query: str, top_k: int = 3, threshold: float = 0.25) -> list[dict]:
        """
        Busca los manhwas más relevantes para la consulta.

        Args:
            query: Texto de búsqueda en cualquier idioma.
            top_k: Número máximo de resultados a devolver.
            threshold: Similitud mínima para incluir un resultado (0-1).

        Returns:
            Lista de manhwas ordenados por similitud descendente.
        """
        query_emb = self.model.encode([query], normalize_embeddings=True)
        similarities = (self.embeddings @ query_emb.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                results.append({**self.manhwas[idx], "similitud": round(score, 3)})

        return results


def get_engine() -> SemanticSearchEngine:
    """Devuelve la instancia global del motor (inicialización lazy)."""
    global _engine
    if _engine is None:
        _engine = SemanticSearchEngine()
    return _engine
