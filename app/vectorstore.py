import os
import json
import faiss
import numpy as np
from embeddings import EmbeddingModel
from config import FAISS_INDEX_PATH, METADATA_STORE


class VectorStore:
    def __init__(self):
        self.embedder = EmbeddingModel()

        # Load or create FAISS index
        if os.path.exists(FAISS_INDEX_PATH):
            self.index = faiss.read_index(FAISS_INDEX_PATH)
        else:
            # Will be created on first insert
            self.index = None

        # Metadata store
        if os.path.exists(METADATA_STORE):
            with open(METADATA_STORE, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []

    def add_texts(self, texts: list[str], sources: list[str]):
        """Add multiple documents (already chunked)."""

        embeddings = self.embedder.embed_texts(texts)

        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)

        # Update metadata
        for t, src in zip(texts, sources):
            self.metadata.append({"text": t, "source": src})

        self._save()

    def search(self, query: str, top_k: int = 5):
        """Search relevant chunks."""
        q_emb = self.embedder.embed_text(query)

        if self.index is None:
            return []

        distances, indices = self.index.search(q_emb, top_k)
        results = []

        for idx in indices[0]:
            if idx < len(self.metadata):
                results.append(self.metadata[idx])

        return results

    def _save(self):
        """Save FAISS index + metadata."""
        if self.index is not None:
            # Make sure directory exists
            os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
            faiss.write_index(self.index, FAISS_INDEX_PATH)

        os.makedirs(os.path.dirname(METADATA_STORE), exist_ok=True)
        with open(METADATA_STORE, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
