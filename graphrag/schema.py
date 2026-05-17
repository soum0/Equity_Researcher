from __future__ import annotations

from typing import Optional, Literal

from pydantic import BaseModel

NodeType = Literal[
    "Company",
    "Person",
    "Segment",
    "RiskFactor",
    "Metric",
    "Concept",
    "NewsArticle",
    "MacroIndicator",
    "Competitor",
    "Geography",
    "Product",
    "Regulation",
    "Chunk",
]

EdgeType = Literal[
    "HAS_SEGMENT",
    "HAS_RISK",
    "GENERATES_REVENUE",
    "EMPLOYS",
    "COMPETES_WITH",
    "OPERATES_IN",
    "MENTIONED_IN",
    "LINKED_TO",
    "IMPACTS",
    "IMPROVED_VS_PRIOR",
    "WORSENED_VS_PRIOR",
    "SOURCE_CHUNK",
    "SAME_AS",
]


class Node(BaseModel):
    id: str
    type: NodeType
    label: str
    properties: dict = {}
    source_doc: Optional[str] = None
    source_section: Optional[str] = None
    embedding_id: Optional[str] = None


class Edge(BaseModel):
    source_id: str
    target_id: str
    type: EdgeType
    properties: dict = {}
    source_doc: Optional[str] = None


class KnowledgeGraph(BaseModel):
    ticker: str
    built_at: str
    nodes: list[Node] = []
    edges: list[Edge] = []

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return len(self.edges)

    def get_node(self, node_id: str) -> Optional[Node]:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> list[Node]:
        neighbor_ids: list[str] = []
        for edge in self.edges:
            if edge.source_id == node_id:
                if edge_type is None or edge.type == edge_type:
                    neighbor_ids.append(edge.target_id)
        return [n for n in self.nodes if n.id in neighbor_ids]
