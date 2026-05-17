from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

from openai import OpenAI
from tqdm import tqdm

from config import Config

log = logging.getLogger(__name__)

PRIORITY_SECTIONS = {"Risk Factors", "MD&A", "Business Overview", "News"}

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    return _client


EXTRACTION_SYSTEM_PROMPT = """You are a financial document analyst. Extract entities and relationships from the provided text chunks from SEC filings or financial news articles.

You will receive one or more numbered chunks. Return a JSON array — one object per chunk — in this exact format:
[
  {
    "chunk_index": 0,
    "entities": [
      {
        "id": "short_unique_snake_case_id",
        "type": "Company|Person|Segment|RiskFactor|Metric|Concept|Geography|Product|Regulation|Competitor",
        "label": "Human readable name",
        "properties": {
          "description": "one sentence description",
          "value": "numeric value if applicable",
          "unit": "USD|%|x|null",
          "sentiment": "positive|negative|neutral|null"
        }
      }
    ],
    "relationships": [
      {
        "source_id": "entity_id_from_above",
        "target_id": "entity_id_from_above",
        "type": "HAS_RISK|HAS_SEGMENT|GENERATES_REVENUE|EMPLOYS|COMPETES_WITH|OPERATES_IN|MENTIONED_IN|LINKED_TO|IMPACTS",
        "properties": {
          "confidence": 0.9,
          "description": "why these are related"
        }
      }
    ]
  }
]

Rules:
- Only extract entities explicitly mentioned in the text
- Minimum confidence 0.7 for relationships
- Use consistent IDs: type prefix + snake_case label e.g. "risk_supply_chain", "segment_iphone", "geo_china"
- Maximum 15 entities and 20 relationships per chunk
- Return empty lists if nothing significant found
- NEVER invent information not present in the text
- Return ONLY the JSON array, no other text"""


def _parse_json_safe(raw: str) -> list | dict:
    """Parse JSON with a regex-based repair fallback."""
    # Strip markdown fences
    text = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    text = re.sub(r"\n?```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try extracting the outermost JSON array or object
        for pattern in (r"\[.*\]", r"\{.*\}"):
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return []


class EntityExtractor:
    def __init__(self, config: Config):
        self.config = config
        self.client = _get_client()
        self.model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")

    def extract_from_chunk(self, chunk: dict) -> dict:
        """Extract from a single chunk. Used as a fallback."""
        results = self._call_api([chunk])
        if results:
            r = results[0]
            return {
                "chunk_id": chunk["chunk_id"],
                "entities": r.get("entities", []),
                "relationships": r.get("relationships", []),
                "extraction_success": True,
            }
        return {
            "chunk_id": chunk["chunk_id"],
            "entities": [],
            "relationships": [],
            "extraction_success": False,
        }

    def _call_api(self, chunks_batch: list[dict]) -> list[dict]:
        """Send up to EXTRACTION_BATCH_SIZE chunks in a single API call."""
        import time

        numbered = "\n\n".join(
            f"--- Chunk {i} ---\n{c['text'][:600]}"
            for i, c in enumerate(chunks_batch)
        )

        def _do_call(batch_text: str, n_chunks: int) -> list[dict]:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=4000,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Extract entities and relationships from these {n_chunks} chunk(s):\n\n"
                            f"{batch_text}"
                        ),
                    },
                ],
            )
            extracted_text = response.choices[0].message.content
            parsed = _parse_json_safe(extracted_text)
            if isinstance(parsed, dict):
                parsed = [parsed]
            if not isinstance(parsed, list):
                return []
            return parsed

        try:
            return _do_call(numbered, len(chunks_batch))

        except Exception as e:
            if "413" in str(e) or "429" in str(e):
                log.warning(
                    f"Rate/size error for batch of {len(chunks_batch)} chunks: {e}. "
                    f"Retrying individually after 5s..."
                )
                time.sleep(5)
                # Retry one chunk at a time
                results = []
                for i, chunk in enumerate(chunks_batch):
                    try:
                        single = f"--- Chunk {i} ---\n{chunk['text'][:600]}"
                        chunk_results = _do_call(single, 1)
                        results.extend(chunk_results)
                    except Exception as inner_e:
                        log.warning(f"Single-chunk retry failed for chunk {i}: {inner_e}")
                return results
            else:
                log.warning(f"API call failed for batch of {len(chunks_batch)} chunks: {e}")
                return []

    def extract_batch(self, chunks: list[dict]) -> list[dict]:
        """
        Filter to priority sections only, cap each section at MAX_CHUNKS_PER_SECTION
        (evenly sampled), then call the API in batches of EXTRACTION_BATCH_SIZE chunks.
        """
        # Group priority chunks by section
        by_section: dict[str, list[dict]] = {}
        for c in chunks:
            sec = c.get("section", "")
            if sec in PRIORITY_SECTIONS:
                by_section.setdefault(sec, []).append(c)

        # Evenly sample each section down to the cap
        cap = self.config.MAX_CHUNKS_PER_SECTION
        to_process: list[dict] = []
        for sec, sec_chunks in by_section.items():
            if len(sec_chunks) <= cap:
                sampled = sec_chunks
            else:
                step = len(sec_chunks) / cap
                sampled = [sec_chunks[int(i * step)] for i in range(cap)]
            to_process.extend(sampled)

        batch_size = self.config.EXTRACTION_BATCH_SIZE
        batches = [to_process[i : i + batch_size] for i in range(0, len(to_process), batch_size)]
        n_batches = len(batches)
        workers = min(4, n_batches)
        est_minutes = (n_batches * 15) / 60 / workers
        print(f"      Processing {len(to_process)} chunks "
              f"({n_batches} API calls, {workers} parallel workers, ~{est_minutes:.1f} min estimated)...")

        # results_map preserves order: batch_index -> list[extraction_result]
        results_map: dict[int, list[dict]] = {}

        with tqdm(total=len(to_process), desc="Extracting entities", unit="chunk") as pbar:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_to_idx = {
                    pool.submit(self._call_api, batch): (idx, batch)
                    for idx, batch in enumerate(batches)
                }
                for future in as_completed(future_to_idx):
                    idx, batch = future_to_idx[future]
                    api_results = future.result() or []
                    batch_extractions: list[dict] = []
                    for j, chunk in enumerate(batch):
                        chunk_result = next(
                            (r for r in api_results if r.get("chunk_index") == j),
                            api_results[j] if j < len(api_results) else {},
                        )
                        batch_extractions.append({
                            "chunk_id": chunk["chunk_id"],
                            "entities": chunk_result.get("entities", []),
                            "relationships": chunk_result.get("relationships", []),
                            "extraction_success": bool(chunk_result),
                        })
                    results_map[idx] = batch_extractions
                    pbar.update(len(batch))

        # Flatten in original batch order
        results: list[dict] = []
        for idx in range(n_batches):
            results.extend(results_map.get(idx, []))
        return results

    def deduplicate_entities(
        self, all_extractions: list[dict], ticker: str
    ) -> tuple[list, list]:
        entity_map: dict[str, dict] = {}
        all_relationships: list[dict] = []
        same_as_edges: list[dict] = []

        for extraction in all_extractions:
            chunk_id = extraction["chunk_id"]

            for raw_ent in extraction.get("entities", []):
                raw_id = raw_ent.get("id", "")
                if not raw_id:
                    continue
                scoped_id = f"{ticker}_{raw_id}"
                if scoped_id in entity_map:
                    existing = entity_map[scoped_id]
                    for k, v in raw_ent.get("properties", {}).items():
                        if v is not None and existing.get("properties", {}).get(k) is None:
                            existing.setdefault("properties", {})[k] = v
                else:
                    ent = dict(raw_ent)
                    ent["id"] = scoped_id
                    ent["source_chunk"] = chunk_id
                    entity_map[scoped_id] = ent

            for raw_rel in extraction.get("relationships", []):
                confidence = raw_rel.get("properties", {}).get("confidence", 1.0)
                if confidence < 0.7:
                    continue
                rel = dict(raw_rel)
                src = raw_rel.get("source_id", "")
                tgt = raw_rel.get("target_id", "")
                if src:
                    rel["source_id"] = f"{ticker}_{src}"
                if tgt:
                    rel["target_id"] = f"{ticker}_{tgt}"
                rel["source_chunk"] = chunk_id
                all_relationships.append(rel)

        # Fuzzy dedup: SAME_AS edges for near-identical labels within same type
        unique_entities = list(entity_map.values())
        by_type: dict[str, list[dict]] = {}
        for ent in unique_entities:
            by_type.setdefault(ent.get("type", ""), []).append(ent)

        for type_group in by_type.values():
            for i, a in enumerate(type_group):
                for b in type_group[i + 1:]:
                    ratio = SequenceMatcher(
                        None,
                        a.get("label", "").lower(),
                        b.get("label", "").lower(),
                    ).ratio()
                    if ratio > 0.8:
                        same_as_edges.append({
                            "source_id": a["id"],
                            "target_id": b["id"],
                            "type": "SAME_AS",
                            "properties": {"similarity": round(ratio, 3)},
                        })

        all_relationships.extend(same_as_edges)
        return unique_entities, all_relationships

    def process_all(self, chunks: list[dict], ticker: str) -> tuple[list, list]:
        all_extractions = self.extract_batch(chunks)
        entities, relationships = self.deduplicate_entities(all_extractions, ticker)

        graphrag_dir = f"{self.config.DATA_DIR}/{ticker}/graphrag"
        os.makedirs(graphrag_dir, exist_ok=True)

        with open(f"{graphrag_dir}/entities.json", "w") as f:
            json.dump(entities, f, indent=2)
        with open(f"{graphrag_dir}/relationships.json", "w") as f:
            json.dump(relationships, f, indent=2)
        # Save raw per-chunk results so SOURCE_CHUNK edges can be rebuilt in --skip-extraction mode
        with open(f"{graphrag_dir}/extractions_per_chunk.json", "w") as f:
            json.dump(all_extractions, f, indent=2)

        print(f"      Entities found:      {len(entities)}")
        print(f"      Relationships found: {len(relationships)}  OK")

        self._last_extractions = all_extractions
        return entities, relationships
