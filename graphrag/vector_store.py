from __future__ import annotations

import chromadb

from config import Config


class VectorStore:
    def __init__(self, ticker: str, config: Config):
        self.ticker = ticker
        self.config = config
        db_dir = config.CHROMA_DB_DIR.format(ticker=ticker)
        self.client = chromadb.PersistentClient(path=db_dir)
        self.collection_name = f"financial_chunks_{ticker.lower()}"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[dict]):
        # Dedup by chunk_id — last occurrence wins (handles re-runs cleanly)
        seen: dict[str, dict] = {}
        for c in chunks:
            if "embedding" in c:
                seen[c["chunk_id"]] = c
        eligible = list(seen.values())
        batch_size = 100

        for i in range(0, len(eligible), batch_size):
            batch = eligible[i : i + batch_size]
            ids = [c["chunk_id"] for c in batch]
            documents = [c["text"] for c in batch]
            embeddings = [c["embedding"] for c in batch]
            metadatas = []
            for c in batch:
                meta = {
                    "chunk_id": c.get("chunk_id", ""),
                    "ticker": c.get("ticker", ""),
                    "source_file": c.get("source_file", ""),
                    "form_type": c.get("form_type", ""),
                    "filing_date": c.get("filing_date", "") or "",
                    "section": c.get("section", ""),
                    "token_count": c.get("token_count", 0),
                }
                if c.get("sentiment_score") is not None:
                    meta["sentiment_score"] = float(c["sentiment_score"])
                metadatas.append(meta)

            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

    def similarity_search(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        filters: dict | None = None,
    ) -> list[dict]:
        kwargs: dict = {"query_embeddings": [query_embedding], "n_results": n_results}
        if filters:
            kwargs["where"] = filters

        results = self.collection.query(**kwargs)

        out: list[dict] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, text, meta, dist in zip(ids, documents, metadatas, distances):
            score = max(0.0, 1.0 - dist)
            out.append({
                "chunk_id": chunk_id,
                "text": text,
                "section": meta.get("section", ""),
                "source_file": meta.get("source_file", ""),
                "form_type": meta.get("form_type", ""),
                "filing_date": meta.get("filing_date", ""),
                "distance": round(dist, 4),
                "score": round(score, 4),
            })

        return out

    def get_chunk_by_id(self, chunk_id: str) -> dict | None:
        result = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        ids = result.get("ids", [])
        if not ids:
            return None
        return {
            "chunk_id": ids[0],
            "text": result["documents"][0],
            **result["metadatas"][0],
        }

    def collection_stats(self) -> dict:
        return {
            "total_chunks": self.collection.count(),
            "collection_name": self.collection_name,
            "embedding_model": self.config.EMBEDDING_MODEL,
        }
