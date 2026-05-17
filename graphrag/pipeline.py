from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from config import Config
from .chunker import SECChunker
from .embedder import ChunkEmbedder
from .extractor import EntityExtractor
from .graph_builder import GraphBuilder
from .retriever import HybridRetriever
from .vector_store import VectorStore


class GraphRAGPipeline:
    def __init__(self, ticker: str, config: Config | None = None):
        self.ticker = ticker.upper()
        self.config = config or Config()

    def _validate_step1(self):
        data_dir = f"{self.config.DATA_DIR}/{self.ticker}"
        required = [
            f"{data_dir}/company_profile.json",
            f"{data_dir}/sec_filings/metadata.json",
        ]
        missing = [p for p in required if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"Step 1 data missing for {self.ticker}. Run Step 1 first.\n"
                f"Missing: {missing}"
            )

    def build(self, skip_extraction: bool = False) -> dict:
        self._validate_step1()

        graphrag_dir = f"{self.config.DATA_DIR}/{self.ticker}/graphrag"
        os.makedirs(graphrag_dir, exist_ok=True)

        stats: dict = {
            "ticker": self.ticker,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "chunking": {},
            "extraction": {},
            "vector_store": {},
            "graph": {},
        }

        # ── Step 1/5: Chunk ───────────────────────────────────────────────────
        print("[1/5] Chunking documents...")
        chunks: list[dict] = []
        try:
            chunker = SECChunker(self.config)
            chunks = chunker.process_all(self.ticker)
            stats["chunking"] = {
                "total_chunks": len(chunks),
                "sec_chunks": sum(1 for c in chunks if c.get("form_type") in ("10-K", "10-Q")),
                "news_chunks": sum(1 for c in chunks if c.get("form_type") == "NEWS"),
                "financial_chunks": sum(1 for c in chunks if c.get("form_type") == "FINANCIALS"),
                "macro_chunks": sum(1 for c in chunks if c.get("form_type") == "MACRO"),
            }
        except Exception as exc:
            print(f"      WARN: Chunking failed: {exc}")
            stats["chunking"]["error"] = str(exc)

        # ── Step 2/5: Extract entities ────────────────────────────────────────
        print("[2/5] Extracting entities (Claude)...")
        entities: list[dict] = []
        relationships: list[dict] = []
        extractions: list[dict] = []

        if skip_extraction:
            entities_path = f"{graphrag_dir}/entities.json"
            rels_path = f"{graphrag_dir}/relationships.json"
            extractions_path = f"{graphrag_dir}/extractions_per_chunk.json"
            if os.path.exists(entities_path) and os.path.exists(rels_path):
                with open(entities_path) as f:
                    entities = json.load(f)
                with open(rels_path) as f:
                    relationships = json.load(f)
                if os.path.exists(extractions_path):
                    with open(extractions_path) as f:
                        extractions = json.load(f)
                print(
                    f"      Loaded {len(entities)} entities, {len(relationships)} relationships, "
                    f"{len(extractions)} chunk extractions from cache (--skip-extraction)"
                )
            else:
                print("      WARN: No cached entities found; running extraction anyway.")
                skip_extraction = False

        if not skip_extraction:
            try:
                extractor = EntityExtractor(self.config)
                entities, relationships = extractor.process_all(chunks, self.ticker)
                extractions = getattr(extractor, "_last_extractions", [])
                stats["extraction"] = {
                    "chunks_processed": len(extractions),
                    "total_entities": len(entities),
                    "total_relationships": len(relationships),
                    "unique_entity_types": _count_by_type(entities),
                }
            except Exception as exc:
                print(f"      WARN: Extraction failed: {exc}")
                stats["extraction"]["error"] = str(exc)

        # ── Step 3/5: Embed ───────────────────────────────────────────────────
        print("[3/5] Embedding chunks...")
        try:
            embedder = ChunkEmbedder(self.config)
            chunks = embedder.embed_chunks(chunks)
            embedded_count = sum(1 for c in chunks if "embedding" in c)
            stats["vector_store"]["total_vectors"] = embedded_count
            stats["vector_store"]["embedding_model"] = self.config.EMBEDDING_MODEL
            stats["vector_store"]["dimensions"] = 384
            print(f"      Model: {self.config.EMBEDDING_MODEL} (384d)  OK")
        except Exception as exc:
            print(f"      WARN: Embedding failed: {exc}")
            stats["vector_store"]["error"] = str(exc)

        # ── Step 4/5: Vector store ────────────────────────────────────────────
        print("[4/5] Building vector store (ChromaDB)...")
        try:
            vs = VectorStore(self.ticker, self.config)
            vs.add_chunks(chunks)
            col_stats = vs.collection_stats()
            print(f"      {col_stats['total_chunks']} vectors indexed  OK")
            stats["vector_store"]["indexed"] = col_stats["total_chunks"]
        except Exception as exc:
            print(f"      WARN: Vector store failed: {exc}")
            stats["vector_store"]["error"] = str(exc)

        # ── Step 5/5: Graph ───────────────────────────────────────────────────
        print("[5/5] Building knowledge graph...")
        try:
            builder = GraphBuilder(self.ticker, self.config)
            builder.build(self.ticker, entities, relationships, chunks, extractions)
            graph_stats = builder.compute_graph_stats()
            stats["graph"] = graph_stats
        except Exception as exc:
            print(f"      WARN: Graph build failed: {exc}")
            stats["graph"]["error"] = str(exc)

        # ── Report ────────────────────────────────────────────────────────────
        stats["status"] = "complete"
        report = self.generate_report(stats)
        report_path = f"{graphrag_dir}/graphrag_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        _print_summary(report)

        # ── Test retrieval ────────────────────────────────────────────────────
        print("\nTesting retrieval...")
        try:
            self.test_retrieval()
        except Exception as exc:
            print(f"  WARN: Retrieval test failed: {exc}")

        print(f"\n  Report: {report_path}")
        return report

    def load(self) -> HybridRetriever:
        graph_path = self.config.GRAPH_FILE.format(ticker=self.ticker)
        if not os.path.exists(graph_path):
            raise FileNotFoundError(
                f"GraphRAG index not found for {self.ticker}. "
                f"Run: python main.py {self.ticker} --build-graph"
            )
        return HybridRetriever(self.ticker, self.config)

    def generate_report(self, stats: dict) -> dict:
        # Always read entity/relationship counts from disk so skip-extraction mode shows real numbers
        graphrag_dir = f"{self.config.DATA_DIR}/{self.ticker}/graphrag"
        entities_on_disk: list = []
        rels_on_disk: list = []
        for path, target in [
            (f"{graphrag_dir}/entities.json", entities_on_disk),
            (f"{graphrag_dir}/relationships.json", rels_on_disk),
        ]:
            if os.path.exists(path):
                with open(path) as f:
                    target.extend(json.load(f))

        chunking = stats.get("chunking", {})
        extraction = stats.get("extraction", {})
        vs = stats.get("vector_store", {})
        graph = stats.get("graph", {})

        return {
            "ticker": self.ticker,
            "built_at": stats.get("built_at", datetime.now(timezone.utc).isoformat()),
            "status": stats.get("status", "unknown"),
            "chunking": {
                "total_chunks": chunking.get("total_chunks", 0),
                "sec_chunks": chunking.get("sec_chunks", 0),
                "news_chunks": chunking.get("news_chunks", 0),
                "financial_chunks": chunking.get("financial_chunks", 0),
                "macro_chunks": chunking.get("macro_chunks", 0),
            },
            "extraction": {
                "chunks_processed": extraction.get("chunks_processed", len(entities_on_disk)),
                "total_entities": len(entities_on_disk),
                "total_relationships": len(rels_on_disk),
                "unique_entity_types": _count_by_type(entities_on_disk),
            },
            "vector_store": {
                "total_vectors": vs.get("total_vectors", vs.get("indexed", 0)),
                "embedding_model": vs.get("embedding_model", self.config.EMBEDDING_MODEL),
                "dimensions": vs.get("dimensions", 384),
            },
            "graph": {
                "total_nodes": graph.get("total_nodes", 0),
                "total_edges": graph.get("total_edges", 0),
                "nodes_by_type": graph.get("nodes_by_type", {}),
            },
        }

    def test_retrieval(self) -> dict:
        retriever = HybridRetriever(self.ticker, self.config)
        test_queries = [
            f"What are the main risk factors for {self.ticker}?",
            "What is the revenue trend and growth outlook?",
            "What do recent news articles say about the company?",
        ]
        labels = ["Main risk factors", "Revenue trend", "Recent news sentiment"]
        results: dict = {}

        for label, query in zip(labels, test_queries):
            try:
                ctx = retriever.hybrid_retrieve(query)
                n_chunks = len(ctx.get("vector_chunks", []))
                n_entities = len(ctx.get("graph_context", {}).get("entities_found", []))
                preview = ctx.get("combined_context", "")[:200].replace("\n", " ")
                print(f'  Query: "{label}" → {n_chunks} chunks, {n_entities} entities  OK')
                results[label] = {
                    "chunks": n_chunks,
                    "entities": n_entities,
                    "preview": preview,
                }
            except Exception as exc:
                print(f'  Query: "{label}" → FAILED: {exc}')
                results[label] = {"error": str(exc)}

        return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_by_type(entities: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in entities:
        t = e.get("type", "Unknown")
        counts[t] = counts.get(t, 0) + 1
    return counts


def _print_summary(report: dict):
    chunking = report.get("chunking", {})
    extraction = report.get("extraction", {})
    vs = report.get("vector_store", {})
    graph = report.get("graph", {})

    ticker = report.get("ticker", "")
    print("\n" + "=" * 46)
    print(f"GraphRAG Build Complete -- {ticker}")
    print(f"   Chunks:        {chunking.get('total_chunks', 0)}")
    print(f"   Entities:      {extraction.get('total_entities', 0)}")
    print(f"   Relationships: {extraction.get('total_relationships', 0)}")
    print(f"   Vectors:       {vs.get('total_vectors', 0)}")
    print(f"   Graph nodes:   {graph.get('total_nodes', 0)}")
    print(f"   Graph edges:   {graph.get('total_edges', 0)}")
    print("=" * 46)
