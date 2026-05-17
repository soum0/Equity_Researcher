from __future__ import annotations

import json
import os
from datetime import datetime, timezone


class PipelineCheckpointer:
    """
    Saves and loads pipeline state after each node.
    Enables resume from checkpoint if pipeline is interrupted.
    """

    CHECKPOINT_ORDER = [
        "01_ingest",
        "02_build_graph",
        "03_fundamental",
        "04_risk",
        "05_sentiment",
        "06_lead_analyst",
        "07_pdf",
    ]

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.checkpoint_dir = f"data/{ticker}/pipeline/checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _path(self, node_name: str) -> str:
        return os.path.join(self.checkpoint_dir, f"{node_name}.json")

    def save(self, node_name: str, state: dict):
        """Save current state after node completes."""
        payload = dict(state)
        payload["_checkpoint_node"] = node_name
        payload["_saved_at"] = datetime.now(timezone.utc).isoformat()
        with open(self._path(node_name), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    def load(self, node_name: str) -> dict | None:
        """Load checkpoint for a specific node. Returns None if missing."""
        path = self._path(node_name)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_latest_checkpoint(self) -> str | None:
        """Find the latest completed node by checking checkpoint files in order."""
        latest = None
        for name in self.CHECKPOINT_ORDER:
            if os.path.exists(self._path(name)):
                latest = name
        return latest

    def clear_from(self, node_name: str):
        """Delete checkpoints from node_name onwards (inclusive)."""
        found = False
        for name in self.CHECKPOINT_ORDER:
            if name == node_name:
                found = True
            if found:
                path = self._path(name)
                if os.path.exists(path):
                    os.remove(path)

    def save_full_state(self, state: dict):
        """Save complete final state to data/{ticker}/pipeline/state.json."""
        out_dir = os.path.dirname(self.checkpoint_dir)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "state.json")
        payload = dict(state)
        payload["_saved_at"] = datetime.now(timezone.utc).isoformat()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    def should_skip_node(self, node_name: str, refresh: bool) -> bool:
        """Return True if node should be skipped (checkpointed and not refresh)."""
        if refresh:
            return False
        return os.path.exists(self._path(node_name))
