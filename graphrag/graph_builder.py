from __future__ import annotations

import json
import os
import pickle
from collections import Counter

import networkx as nx

from config import Config


class GraphBuilder:
    def __init__(self, ticker: str, config: Config):
        self.ticker = ticker
        self.config = config
        self.G: nx.DiGraph = nx.DiGraph()

    # ── Seed methods ──────────────────────────────────────────────────────────

    def seed_company_node(self, company_data: dict):
        node_id = f"Company:{self.ticker}"
        self.G.add_node(node_id, **{
            "id": node_id,
            "type": "Company",
            "label": company_data.get("name", self.ticker),
            "properties": {
                "ticker": company_data.get("ticker", self.ticker),
                "exchange": company_data.get("exchange", ""),
                "sector": company_data.get("sector", ""),
                "industry": company_data.get("industry", ""),
                "market_cap": company_data.get("market_cap"),
                "shares_outstanding": company_data.get("shares_outstanding"),
                "ipo_date": company_data.get("ipo_date", ""),
                "website": company_data.get("website", ""),
                "country": company_data.get("country", ""),
            },
        })

    def seed_metric_nodes(self, price_data: dict, key_metrics: dict):
        company_id = f"Company:{self.ticker}"
        basic = key_metrics.get("basic_financials", {})
        per_share = basic.get("per_share", {})
        valuation = basic.get("valuation", {})
        margins = basic.get("margins", {})
        momentum = basic.get("momentum", {})

        metrics = {
            f"Metric:{self.ticker}_pe_ratio": ("P/E Ratio", valuation.get("pe_ttm"), "x"),
            f"Metric:{self.ticker}_pb_ratio": ("P/B Ratio", valuation.get("pb_annual"), "x"),
            f"Metric:{self.ticker}_ev_ebitda": ("EV/EBITDA", valuation.get("ev_ebitda"), "x"),
            f"Metric:{self.ticker}_net_margin": ("Net Margin", margins.get("net_margin_ttm"), "%"),
            f"Metric:{self.ticker}_gross_margin": ("Gross Margin", margins.get("gross_margin_ttm"), "%"),
            f"Metric:{self.ticker}_eps_ttm": ("EPS (TTM)", per_share.get("eps_ttm"), "USD"),
            f"Metric:{self.ticker}_52w_high": ("52-Week High", momentum.get("52w_high"), "USD"),
            f"Metric:{self.ticker}_52w_low": ("52-Week Low", momentum.get("52w_low"), "USD"),
            f"Metric:{self.ticker}_beta": ("Beta", momentum.get("beta"), "x"),
        }

        # Market cap from company profile (already loaded separately) lives on the company node
        for node_id, (label, value, unit) in metrics.items():
            if value is None:
                continue
            self.G.add_node(node_id, **{
                "id": node_id,
                "type": "Metric",
                "label": label,
                "properties": {"value": value, "unit": unit},
            })
            self.G.add_edge(company_id, node_id, type="LINKED_TO",
                            properties={"auto_seeded": True})

    def seed_macro_nodes(self, macro_data: dict):
        company_id = f"Company:{self.ticker}"
        for key, ind in macro_data.get("indicators", {}).items():
            if "error" in ind:
                continue
            node_id = f"MacroIndicator:{key}"
            self.G.add_node(node_id, **{
                "id": node_id,
                "type": "MacroIndicator",
                "label": key.replace("_", " ").title(),
                "properties": {
                    "value": ind.get("latest_value"),
                    "unit": ind.get("unit", ""),
                    "trend": ind.get("trend_3m", ""),
                    "latest_date": ind.get("latest_date", ""),
                },
            })
            self.G.add_edge(node_id, company_id, type="IMPACTS",
                            properties={"auto_seeded": True})

    # ── Entity / relationship ingestion ──────────────────────────────────────

    def add_extracted_nodes(self, entities: list[dict]):
        for ent in entities:
            node_id = ent.get("id", "")
            if not node_id:
                continue
            if node_id in self.G:
                # Merge properties
                existing = self.G.nodes[node_id]
                for k, v in ent.get("properties", {}).items():
                    if v is not None and existing.get("properties", {}).get(k) is None:
                        existing.setdefault("properties", {})[k] = v
            else:
                self.G.add_node(node_id, **{
                    "id": node_id,
                    "type": ent.get("type", "Concept"),
                    "label": ent.get("label", node_id),
                    "properties": ent.get("properties", {}),
                    "source_doc": ent.get("source_chunk", ""),
                })

    def add_extracted_edges(self, relationships: list[dict]):
        import logging
        log = logging.getLogger(__name__)
        for rel in relationships:
            src = rel.get("source_id", "")
            tgt = rel.get("target_id", "")
            if not src or not tgt:
                continue
            if src not in self.G:
                log.debug(f"Skipping edge: source node not found: {src}")
                continue
            if tgt not in self.G:
                log.debug(f"Skipping edge: target node not found: {tgt}")
                continue
            edge_type = rel.get("type", "LINKED_TO")
            self.G.add_edge(src, tgt, type=edge_type,
                            properties=rel.get("properties", {}))

    def connect_chunks_to_entities(self, chunks: list[dict], extractions: list[dict]):
        chunk_map = {c["chunk_id"]: c for c in chunks}
        for extraction in extractions:
            chunk_id = extraction.get("chunk_id", "")
            entities = extraction.get("entities", [])
            if not entities:
                continue

            chunk = chunk_map.get(chunk_id)
            if chunk is None:
                continue

            ticker = chunk.get("ticker", self.ticker)
            chunk_node_id = f"Chunk:{chunk_id}"
            if chunk_node_id not in self.G:
                self.G.add_node(chunk_node_id, **{
                    "id": chunk_node_id,
                    "type": "Chunk",
                    "label": chunk_id,
                    "properties": {
                        "text_preview": chunk.get("text", "")[:200],
                        "section": chunk.get("section", ""),
                        "source_file": chunk.get("source_file", ""),
                        "token_count": chunk.get("token_count", 0),
                    },
                })

            for ent in entities:
                raw_id = ent.get("id", "")
                if not raw_id:
                    continue
                scoped_id = f"{ticker}_{raw_id}" if not raw_id.startswith(ticker) else raw_id
                if scoped_id in self.G:
                    self.G.add_edge(scoped_id, chunk_node_id, type="SOURCE_CHUNK",
                                    properties={})

    # ── Stats & persistence ───────────────────────────────────────────────────

    def compute_graph_stats(self) -> dict:
        nodes_by_type: dict[str, int] = Counter(
            data.get("type", "Unknown")
            for _, data in self.G.nodes(data=True)
        )
        edges_by_type: dict[str, int] = Counter(
            data.get("type", "LINKED_TO")
            for _, _, data in self.G.edges(data=True)
        )
        degrees = dict(self.G.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        most_connected = [
            {"id": nid, "degree": deg} for nid, deg in sorted_nodes[:10]
        ]

        try:
            undirected = self.G.to_undirected()
            components = nx.number_connected_components(undirected)
        except Exception:
            components = -1

        avg_degree = (sum(degrees.values()) / len(degrees)) if degrees else 0.0

        return {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "nodes_by_type": dict(nodes_by_type),
            "edges_by_type": dict(edges_by_type),
            "avg_degree": round(avg_degree, 2),
            "connected_components": components,
            "most_connected_nodes": most_connected,
        }

    def save(self, ticker: str):
        graphrag_dir = f"{self.config.DATA_DIR}/{ticker}/graphrag"
        os.makedirs(graphrag_dir, exist_ok=True)

        graph_path = self.config.GRAPH_FILE.format(ticker=ticker)
        with open(graph_path, "wb") as f:
            pickle.dump(self.G, f)

        stats = self.compute_graph_stats()
        with open(f"{graphrag_dir}/graph_summary.json", "w") as f:
            json.dump(stats, f, indent=2)

    def load(self, ticker: str) -> nx.DiGraph:
        graph_path = self.config.GRAPH_FILE.format(ticker=ticker)
        with open(graph_path, "rb") as f:
            self.G = pickle.load(f)
        return self.G

    def build(
        self,
        ticker: str,
        entities: list,
        relationships: list,
        chunks: list,
        extractions: list,
    ) -> nx.DiGraph:
        data_dir = f"{self.config.DATA_DIR}/{ticker}"

        # Load Step 1 data
        company_data: dict = {}
        company_path = f"{data_dir}/company_profile.json"
        if os.path.exists(company_path):
            with open(company_path) as f:
                company_data = json.load(f)

        key_metrics: dict = {}
        metrics_path = f"{data_dir}/key_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                key_metrics = json.load(f)

        price_data: dict = {}
        price_path = f"{data_dir}/price_data.json"
        if os.path.exists(price_path):
            with open(price_path) as f:
                price_data = json.load(f)

        macro_data: dict = {}
        macro_path = f"{data_dir}/macro_indicators.json"
        if os.path.exists(macro_path):
            with open(macro_path) as f:
                macro_data = json.load(f)

        self.seed_company_node(company_data)
        self.seed_metric_nodes(price_data, key_metrics)
        self.seed_macro_nodes(macro_data)
        self.add_extracted_nodes(entities)
        self.add_extracted_edges(relationships)
        self.connect_chunks_to_entities(chunks, extractions)
        self.save(ticker)

        stats = self.compute_graph_stats()
        print(f"      Nodes: {stats['total_nodes']}  |  Edges: {stats['total_edges']}  OK")

        return self.G
