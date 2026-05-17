from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from openai import APIStatusError, OpenAI

from config import Config
from graphrag.pipeline import GraphRAGPipeline
from ingestion.utils import setup_logger

MAX_PROMPT_CHARS = 8000    # ~2000 tokens, safe for all Groq models
MAX_CONTEXT_CHARS = 4000   # cap per retrieve() combined_context
MAX_USER_MSG_CHARS = 12000  # ~3000 tokens, leaves room for system prompt


class BaseAgent(ABC):
    """Abstract base class for all analysis agents."""

    def __init__(self, ticker: str, config: Config | None = None):
        self.ticker = ticker.upper()
        self.config = config or Config()
        self.logger = setup_logger(self.__class__.__name__)
        self.retriever = GraphRAGPipeline(ticker, self.config).load()
        self.llm_client = OpenAI(
            base_url=self.config.AGENT_BASE_URL,
            api_key=self.config.AGENT_API_KEY,
        )
        self.citations: list[str] = []

    def retrieve(self, query: str, section_filter: str | None = None) -> dict:
        """
        Wrapper around HybridRetriever. Accumulates citations into self.citations.
        Uses section_filter for targeted vector search when provided.
        """
        if section_filter:
            chunks = self.retriever.vector_search(
                query,
                n_results=self.config.AGENT_RETRIEVAL_K,
                section_filter=section_filter,
            )
            chunks_block = "\n".join(
                f"[Source: {c.get('form_type', '')} {c.get('filing_date', '')[:7]} | "
                f"Section: {c.get('section', '')} | Score: {c.get('score', 0):.2f}]\n{c['text']}\n---"
                for c in chunks
            )
            sources = list({
                f"{c.get('form_type', '')} {c.get('filing_date', '')[:7]} §{c.get('section', '')}"
                for c in chunks
            })
            ctx = {
                "query": query,
                "vector_chunks": chunks,
                "graph_context": {"entities_found": [], "relationships_found": [], "subgraph_text": ""},
                "combined_context": f"=== RELEVANT TEXT CHUNKS ===\n{chunks_block}\n",
                "sources": sources,
            }
        else:
            ctx = self.retriever.hybrid_retrieve(
                query,
                n_vector_results=self.config.AGENT_RETRIEVAL_K,
                graph_depth=self.config.AGENT_GRAPH_DEPTH,
            )

        for source in ctx.get("sources", []):
            if source and source not in self.citations:
                self.citations.append(source)

        combined = ctx.get("combined_context", "")
        if len(combined) > MAX_CONTEXT_CHARS:
            ctx["combined_context"] = combined[:MAX_CONTEXT_CHARS] + "\n\n[Context truncated for length]"

        return ctx

    def llm_call(
        self,
        system_prompt: str,
        user_message: str,
        expect_json: bool = True,
    ) -> str | dict:
        """Call the LLM via OpenRouter with optional JSON parsing."""
        if expect_json:
            system_prompt = system_prompt + "\n\nReturn ONLY valid JSON. No markdown, no explanation."

        # Hard truncate before any token budget check
        if len(user_message) > MAX_USER_MSG_CHARS:
            user_message = user_message[:MAX_USER_MSG_CHARS] + \
                           "\n\n[Context truncated to fit token limit]"
            self.logger.warning(f"user_message truncated to {MAX_USER_MSG_CHARS} chars")

        # Truncate further if combined prompt still exceeds budget
        if len(system_prompt) + len(user_message) > MAX_PROMPT_CHARS * 4:
            allowed_user_chars = MAX_PROMPT_CHARS * 4 - len(system_prompt) - 200
            if allowed_user_chars > 500:
                user_message = user_message[:allowed_user_chars] + "\n\n[Context truncated for length]"
                self.logger.warning(f"Prompt truncated to {allowed_user_chars} chars")

        time.sleep(0.5)

        def _make_call(msg: str) -> object:
            return self.llm_client.chat.completions.create(
                model=self.config.AGENT_MODEL,
                max_tokens=self.config.AGENT_MAX_TOKENS,
                temperature=self.config.AGENT_TEMPERATURE,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": msg},
                ],
            )

        start = time.time()
        try:
            response = _make_call(user_message)
        except APIStatusError as e:
            if e.status_code in (413, 429):
                match = re.search(r'try again in ([\d.]+)s', str(e))
                wait = float(match.group(1)) + 1.0 if match else 10.0
                self.logger.warning(f"Rate limit hit ({e.status_code}), sleeping {wait:.1f}s then retrying...")
                time.sleep(wait)
                response = _make_call(user_message[:8000])
            else:
                raise
        latency = time.time() - start

        usage = response.usage
        self.logger.info(
            f"LLM call | model={self.config.AGENT_MODEL} | "
            f"prompt_tokens={usage.prompt_tokens} | "
            f"completion_tokens={usage.completion_tokens} | "
            f"latency={latency:.1f}s"
        )

        raw = response.choices[0].message.content.strip()

        if not expect_json:
            return raw

        # Repair common LLM non-JSON values before parsing
        cleaned = re.sub(r':\s*INSUFFICIENT DATA', ': null', raw)
        cleaned = re.sub(r':\s*N/A(?=[,\n\}])', ': null', cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        self.logger.warning(f"Failed to parse LLM response as JSON. Raw preview: {raw[:300]}")
        return {}

    def save_result(self, result: dict, filename: str):
        """Save result to data/{ticker}/analysis/{filename} with metadata."""
        result["_metadata"] = {
            "ticker": self.ticker,
            "agent": self.__class__.__name__,
            "model": self.config.AGENT_MODEL,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "citations": self.citations,
        }
        out_dir = f"{self.config.DATA_DIR}/{self.ticker}/analysis"
        os.makedirs(out_dir, exist_ok=True)
        path = f"{out_dir}/{filename}"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        self.logger.info(f"Saved: {path}")

    def load_result(self, filename: str) -> dict | None:
        """Load previously saved result. Returns None if file doesn't exist."""
        path = f"{self.config.DATA_DIR}/{self.ticker}/analysis/{filename}"
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @abstractmethod
    def analyze(self, refresh: bool = False) -> dict:
        """Each agent implements this. Returns structured analysis dict."""
