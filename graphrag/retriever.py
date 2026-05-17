from __future__ import annotations

import difflib
import json
import os

import networkx as nx
from openai import OpenAI

from config import Config
from .embedder import ChunkEmbedder
from .graph_builder import GraphBuilder
from .vector_store import VectorStore


class HybridRetriever:
    def __init__(self, ticker: str, config: Config):
        self.ticker = ticker
        self.config = config
        self.vector_store = VectorStore(ticker, config)
        self.embedder = ChunkEmbedder(config)
        self.graph_builder = GraphBuilder(ticker, config)
        self.G: nx.DiGraph = self.graph_builder.load(ticker)
        self._client = OpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
        )

    # ── Vector search ─────────────────────────────────────────────────────────

    def vector_search(
        self,
        query: str,
        n_results: int = 8,
        section_filter: str | None = None,
    ) -> list[dict]:
        embedding = self.embedder.embed_query(query)
        filters = {"section": section_filter} if section_filter else None
        return self.vector_store.similarity_search(embedding, n_results=n_results, filters=filters)

    # ── Graph search ──────────────────────────────────────────────────────────

    def graph_search(
        self,
        entity_id: str,
        depth: int = 2,
        edge_types: list[str] | None = None,
    ) -> dict:
        if entity_id not in self.G:
            return {"root_node": None, "neighbors": [], "subgraph_summary": ""}

        root_data = dict(self.G.nodes[entity_id])
        neighbors: list[dict] = []

        # BFS up to `depth` hops
        visited = {entity_id}
        frontier = [(entity_id, 0)]

        while frontier:
            current, current_depth = frontier.pop(0)
            if current_depth >= depth:
                continue
            for _, nbr, edge_data in self.G.out_edges(current, data=True):
                etype = edge_data.get("type", "LINKED_TO")
                if edge_types and etype not in edge_types:
                    continue
                if nbr not in visited:
                    visited.add(nbr)
                    nbr_data = dict(self.G.nodes[nbr])
                    neighbors.append({
                        "node": nbr_data,
                        "edge_type": etype,
                        "depth": current_depth + 1,
                    })
                    frontier.append((nbr, current_depth + 1))

        # Build summary
        summary_parts: list[str] = []
        root_label = root_data.get("label", entity_id)
        root_type = root_data.get("type", "")
        summary_parts.append(f"{root_label} ({root_type})")
        for n in neighbors[:20]:
            node = n["node"]
            summary_parts.append(
                f"{root_label} --[{n['edge_type']}]--> {node.get('label', '')} ({node.get('type', '')})"
            )

        return {
            "root_node": root_data,
            "neighbors": neighbors,
            "subgraph_summary": ". ".join(summary_parts),
        }

    # ── Entity lookup ─────────────────────────────────────────────────────────

    def find_entity_by_label(self, label: str, node_type: str | None = None) -> list[dict]:
        all_nodes = [
            (nid, data) for nid, data in self.G.nodes(data=True)
            if node_type is None or data.get("type") == node_type
        ]
        labels = [data.get("label", nid) for nid, data in all_nodes]
        close = difflib.get_close_matches(label, labels, n=5, cutoff=0.6)
        matched: list[dict] = []
        for match_label in close:
            for nid, data in all_nodes:
                if data.get("label", nid) == match_label:
                    matched.append({"id": nid, **data})
        return matched

    # ── Hybrid retrieve ───────────────────────────────────────────────────────

    def hybrid_retrieve(
        self,
        query: str,
        n_vector_results: int = 6,
        graph_depth: int = 2,
    ) -> dict:
        # 1. Vector search
        vector_chunks = self.vector_search(query, n_results=n_vector_results)

        # 2. For each chunk, find entities linked via SOURCE_CHUNK edges
        entity_ids_found: set[str] = set()
        for chunk_result in vector_chunks:
            chunk_node_id = f"Chunk:{chunk_result['chunk_id']}"
            if chunk_node_id in self.G:
                for src, _, edge_data in self.G.in_edges(chunk_node_id, data=True):
                    if edge_data.get("type") == "SOURCE_CHUNK":
                        entity_ids_found.add(src)

        # 3. Graph search for each entity
        all_neighbors: list[dict] = []
        relationship_lines: list[str] = []
        seen_neighbor_ids: set[str] = set()

        for eid in list(entity_ids_found)[:10]:
            graph_result = self.graph_search(eid, depth=graph_depth)
            for n in graph_result["neighbors"]:
                nid = n["node"].get("id", "")
                if nid and nid not in seen_neighbor_ids and n["node"].get("type") != "Chunk":
                    seen_neighbor_ids.add(nid)
                    all_neighbors.append(n["node"])
                    root_label = self.G.nodes[eid].get("label", eid) if eid in self.G else eid
                    relationship_lines.append(
                        f"{root_label} --[{n['edge_type']}]--> {n['node'].get('label', '')}"
                    )

        # 4. Build entities_found list
        entities_found = []
        for eid in entity_ids_found:
            if eid in self.G:
                node_data = dict(self.G.nodes[eid])
                if node_data.get("type") != "Chunk":
                    entities_found.append({
                        "label": node_data.get("label", eid),
                        "type": node_data.get("type", ""),
                        "properties": node_data.get("properties", {}),
                    })

        # 5. Pull financial metrics from Company node
        company_node_id = f"Company:{self.ticker}"
        metrics_text = ""
        if company_node_id in self.G:
            metric_parts: list[str] = []
            for _, nbr, edge_data in self.G.out_edges(company_node_id, data=True):
                if edge_data.get("type") == "LINKED_TO" and self.G.nodes[nbr].get("type") == "Metric":
                    m = self.G.nodes[nbr]
                    label = m.get("label", "")
                    props = m.get("properties", {})
                    val = props.get("value", "")
                    unit = props.get("unit", "")
                    if val is not None:
                        metric_parts.append(f"{label}: {val}{unit}")
            if metric_parts:
                metrics_text = " | ".join(metric_parts[:8])

        # 6. Build combined_context
        chunks_block = "\n".join(
            f"[Source: {c.get('form_type', '')} {c.get('filing_date', '')[:7]} | "
            f"Section: {c.get('section', '')} | Score: {c.get('score', 0):.2f}]\n{c['text']}\n---"
            for c in vector_chunks
        )
        entities_line = ", ".join(
            f"{e['label']} ({e['type']})" for e in entities_found[:10]
        )
        relationships_block = "\n".join(relationship_lines[:20])

        combined_context = (
            "=== RELEVANT TEXT CHUNKS ===\n"
            f"{chunks_block}\n\n"
            "=== KNOWLEDGE GRAPH CONTEXT ===\n"
            f"Entities: {entities_line}\n"
            f"Relationships:\n{relationships_block}\n\n"
        )
        if metrics_text:
            combined_context += f"=== FINANCIAL METRICS ===\n{metrics_text}\n"

        sources = list({
            f"{c.get('form_type', '')} {c.get('filing_date', '')[:7]} §{c.get('section', '')}"
            for c in vector_chunks
        })

        return {
            "query": query,
            "vector_chunks": [
                {
                    "text": c["text"],
                    "section": c.get("section", ""),
                    "source": c.get("source_file", ""),
                    "score": c.get("score", 0.0),
                }
                for c in vector_chunks
            ],
            "graph_context": {
                "entities_found": entities_found,
                "relationships_found": [
                    {"line": line} for line in relationship_lines[:20]
                ],
                "subgraph_text": relationships_block,
            },
            "combined_context": combined_context,
            "sources": sources,
        }

    # ── Answer with citations ─────────────────────────────────────────────────

    def answer_with_citations(self, query: str, context: dict) -> dict:
        combined = context.get("combined_context", "")
        response = self._client.chat.completions.create(
            model=self.config.EXTRACTION_MODEL,
            max_tokens=1024,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial analyst. Answer based ONLY on the provided context. "
                        "Cite sources for every claim using [Source: ...] notation."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nContext:\n{combined}",
                },
            ],
        )
        answer_text = response.choices[0].message.content.strip()

        entities_referenced = [
            e["label"] for e in context.get("graph_context", {}).get("entities_found", [])
        ]

        return {
            "query": query,
            "answer": answer_text,
            "citations": context.get("sources", []),
            "entities_referenced": entities_referenced,
        }
