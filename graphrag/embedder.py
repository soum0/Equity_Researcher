from __future__ import annotations

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import Config


class ChunkEmbedder:
    def __init__(self, config: Config):
        self.config = config
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        eligible = [c for c in chunks if c.get("token_count", 0) >= self.config.MIN_CHUNK_TOKENS]
        skipped = len(chunks) - len(eligible)

        texts = [c["text"] for c in eligible]
        embeddings = []

        batch_size = 32
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
            batch = texts[i : i + batch_size]
            vecs = self.model.encode(batch, show_progress_bar=False)
            embeddings.extend(vecs.tolist())

        for chunk, vec in zip(eligible, embeddings):
            chunk["embedding"] = vec

        if skipped:
            import logging
            logging.getLogger(__name__).info(f"Skipped {skipped} chunks below MIN_CHUNK_TOKENS")

        return chunks

    def embed_query(self, query: str) -> list[float]:
        return self.model.encode(query).tolist()
