from sentence_transformers import SentenceTransformer
import numpy as np
from config import EMBEDDING_MODEL


class EmbeddingModel:
    def __init__(self):
        # Load SentenceTransformer model
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        emb = self.model.encode([text], show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(emb, dtype="float32")

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed multiple text strings."""
        embs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(embs, dtype="float32")
