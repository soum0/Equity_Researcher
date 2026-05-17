# AI Financial Analysis Pipeline

A multi-agent, multi-step system that ingests real financial data, builds a GraphRAG knowledge graph, and runs four specialized AI agents to produce a full equity research report — complete with PDF investment memo.

Built entirely with free-tier APIs and a free LLM (OpenRouter). No paid model required.

---

## What It Does

Given any stock ticker (e.g. `AAPL`), the pipeline:

1. **Ingests** SEC filings, live price data, news articles, and macro indicators
2. **Builds a GraphRAG index** — semantic vector search + a NetworkX knowledge graph with entity relationships extracted by an LLM
3. **Runs 4 specialized agents** — Fundamental, Risk, Sentiment, and Lead Analyst
4. **Produces a final investment memo** as Markdown + PDF with a BUY / HOLD / SELL recommendation, price target, scorecard, and cited sources

```
Ticker Input
    │
    ▼
┌─────────────────────────────┐
│  Step 1 — Data Ingestion    │  SEC filings · Price data · News · Macro
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Step 2 — GraphRAG Build    │  Chunk → Extract entities → Embed → Graph
└────────────┬────────────────┘
             │
    ┌────────┴──────────────────────────────┐
    ▼           ▼                           ▼
┌────────┐ ┌────────┐              ┌──────────────┐
│ Fund.  │ │  Risk  │              │  Sentiment   │
│ Agent  │ │ Agent  │              │    Agent     │
└────┬───┘ └───┬────┘              └──────┬───────┘
     │         │     (run in parallel)    │
     └─────────┴──────────────────────────┘
                          │
                          │  all three outputs feed in
                          ▼
               ┌─────────────────────┐
               │   Lead Analyst      │  Disagree resolution + scoring
               │      Agent          │  + bull/bear case writing
               └──────────┬──────────┘
                          │
                          ▼
              Investment Memo (.md + .pdf)
```

---

## Sample Output (AAPL)

```
RECOMMENDATION:  HOLD
Current Price:   $300.23   |   Price Target: $309.24   |   Return: +3.0%

SCORECARD
  Fundamental Agent   →  7.2 / 10   BUY_LEANING
  Risk Agent          →  8.0 / 10   CAUTION
  Sentiment Agent     →  +0.19      POSITIVE
  Composite Score     →  0.51       HOLD  (Confidence: 45%)

SOURCES CITED: 23
```

---

## Project Structure

```
finance_agent/
├── main.py                         # CLI entry point for all steps
├── config.py                       # All settings and API keys
├── requirements.txt                # Full dependency list
├── requirements_streamlit.txt      # Streamlit frontend deps
│
├── ingestion/                      # Step 1 — Data fetchers
│   ├── pipeline.py                 # Orchestrates all fetchers
│   ├── sec_fetcher.py              # SEC EDGAR 10-K / 10-Q downloader
│   ├── price_fetcher.py            # Yahoo Finance price history + ratios
│   ├── company_fetcher.py          # Finnhub company profile + metrics
│   ├── news_fetcher.py             # Finnhub + NewsAPI headlines
│   ├── macro_fetcher.py            # FRED macro indicators
│   └── utils.py                    # Logging, file I/O helpers
│
├── graphrag/                       # Step 2 — Knowledge graph
│   ├── pipeline.py                 # GraphRAGPipeline orchestrator
│   ├── chunker.py                  # Section-aware SEC chunker (tiktoken)
│   ├── extractor.py                # LLM entity/relationship extraction
│   ├── graph_builder.py            # NetworkX DiGraph builder
│   ├── embedder.py                 # Sentence-transformers local embeddings
│   ├── vector_store.py             # ChromaDB persistent vector store
│   ├── retriever.py                # Hybrid retriever (vector + graph)
│   └── schema.py                   # Pydantic Node / Edge / KnowledgeGraph
│
├── agents/                         # Steps 3–6 — Analysis agents
│   ├── base_agent.py               # Abstract base (retrieve + llm_call + save)
│   ├── fundamental_agent.py        # 4-pass fundamental analysis
│   ├── risk_agent.py               # 4-pass risk assessment
│   ├── sentiment_agent.py          # Batch news sentiment + theme analysis
│   ├── lead_analyst_agent.py       # Final synthesis + investment memo
│   └── prompts/                    # System + user prompts per agent
│
└── data/
    └── {TICKER}/
        ├── company_profile.json
        ├── price_data.json
        ├── key_metrics.json
        ├── news_articles.json
        ├── macro_indicators.json
        ├── sec_filings/
        │   ├── 10_K_2024-11-01.txt
        │   └── metadata.json
        ├── graphrag/
        │   ├── chunks.json               # 489 text chunks with metadata
        │   ├── entities.json             # Deduplicated entity nodes
        │   ├── relationships.json        # Entity relationships
        │   ├── extractions_per_chunk.json
        │   ├── graph.pkl                 # Serialized NetworkX DiGraph
        │   └── graphrag_report.json
        └── analysis/
            ├── fundamental_analysis.json
            ├── risk_analysis.json
            ├── sentiment_analysis.json
            ├── investment_memo.json
            ├── investment_memo.md
            └── investment_memo.pdf
```

---

## Setup

### 1. Clone & create virtual environment

```bash
git clone https://github.com/soum0/Equity_Researcher.git
cd finance-agent
python3 -m venv env
source env/bin/activate          # Windows: env\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads the `all-MiniLM-L6-v2` embedding model (~80 MB, cached to `~/.cache/huggingface`).

### 3. Get API keys

| Key | Where to get it | Free tier |
|---|---|---|
| `FINNHUB_API_KEY` | [finnhub.io](https://finnhub.io) | 60 calls/min |
| `NEWSAPI_KEY` | [newsapi.org](https://newsapi.org) | 100 calls/day |
| `FRED_API_KEY` | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) | Unlimited |
| `OPENROUTER_API_KEY` | [openrouter.ai](https://openrouter.ai) | Free models available |
| `SEC_USER_AGENT` | No sign-up — just set your name and email | Free |

### 4. Configure `.env`

```bash
cp .env.example .env    # or create manually
```

```env
FINNHUB_API_KEY     = your_finnhub_key
NEWSAPI_KEY         = your_newsapi_key
FRED_API_KEY        = your_fred_key
SEC_USER_AGENT      = Your Name your@email.com

# OpenRouter (free LLM)
OPENROUTER_API_KEY  = sk-or-v1-...
OPENROUTER_MODEL    = google/gemini-2.0-flash-exp:free   # recommended (fastest free)
# Other fast free options:
# meta-llama/llama-3.2-3b-instruct:free   (~8s per batch)
# mistralai/mistral-7b-instruct:free      (~16s per batch)
# meta-llama/llama-3.3-70b-instruct       (~60s per batch, highest quality)

# Optional tuning
NEWS_LOOKBACK_DAYS  = 30
MAX_SEC_FILINGS     = 3
```

### 5. Validate setup

```bash
python main.py AAPL --validate
```

Expected output:
```
  [FINNHUB_API_KEY]     SET
  [NEWSAPI_KEY]         SET
  [FRED_API_KEY]        SET
  [SEC_USER_AGENT]      SET
  [OPENROUTER_API_KEY]  SET
```

---

## Running the Pipeline

### Step 1 — Ingest data

```bash
python main.py AAPL
```

Downloads and saves all financial data for the ticker. Runs all five fetchers in sequence.

**What gets downloaded:**

| Fetcher | Source | Data |
|---|---|---|
| `company` | Finnhub | Name, sector, market cap, IPO date |
| `price` | Yahoo Finance | 2 years of OHLCV, key ratios, P/E, D/E, margins |
| `news` | Finnhub + NewsAPI | Last 30 days of headlines and summaries |
| `sec` | SEC EDGAR | Last 3 annual (10-K) and quarterly (10-Q) filings |
| `macro` | FRED | Fed Funds Rate, CPI, GDP, unemployment, 10Y yield, VIX |

**Latency breakdown — Step 1:**

| Fetcher | Time | Why |
|---|---|---|
| SEC EDGAR | 15–30s | Downloads 3–6 full filing documents (each 200–300 KB) at 0.1s rate-limit delay per request |
| Price data | 3–5s | Single Yahoo Finance request, parses 2 years of OHLCV |
| Company profile | 1–2s | Finnhub REST call |
| News articles | 2–5s | Two API calls (Finnhub + NewsAPI), returns up to 100 articles |
| Macro (FRED) | 3–8s | 6 time-series requests at 0.2s delay each |
| **Total Step 1** | **~30–60s** | |

**Output:**
```
data/AAPL/
├── company_profile.json    (~1 KB)
├── price_data.json         (~150 KB)
├── key_metrics.json        (~20 KB)
├── news_articles.json      (~200 KB, 90–100 articles)
├── macro_indicators.json   (~30 KB)
└── sec_filings/
    ├── 10_K_2025-10-31.txt (~220 KB)
    ├── 10_K_2024-11-01.txt (~218 KB)
    ├── 10_K_2023-11-03.txt (~217 KB)
    ├── 10_Q_*.txt
    └── metadata.json
```

---

### Step 2 — Build GraphRAG index

```bash
python main.py AAPL --build-graph
```

Builds the knowledge graph index from Step 1 data. Runs 5 sub-steps:

```
[1/5] Chunking documents...
[2/5] Extracting entities (LLM)...
[3/5] Embedding chunks...
[4/5] Building vector store (ChromaDB)...
[5/5] Building knowledge graph...
```

**Latency breakdown — Step 2:**

| Sub-step | Time | Why |
|---|---|---|
| Chunking | ~2s | Pure Python + tiktoken tokenizer. Splits 6 SEC filings (~1.3M chars) and 96 news articles into 489 token-bounded chunks |
| Entity extraction | **60–240s** | The bottleneck — see table below |
| Embedding | ~15s | Local `all-MiniLM-L6-v2` model encodes 489 chunks in batches of 32 on CPU |
| ChromaDB index | ~3s | Upserts 489 embedding vectors to persistent local store |
| Graph build | ~1s | Builds NetworkX DiGraph: seeds Company/Metric/Macro nodes, adds extracted entities + edges |
| **Total (fresh)** | **~80–270s** | Dominated by LLM extraction |
| **Total (--skip-extraction)** | **~20s** | Loads cached entities.json, skips LLM calls entirely |

**Entity extraction latency by model:**

| Model | Cost | Seconds/batch | 16 batches × 4 workers | Recommended? |
|---|---|---|---|---|
| `google/gemini-2.0-flash-exp:free` | Free | ~1s | **~4s total** | Best for speed |
| `meta-llama/llama-3.2-3b-instruct:free` | Free | ~2s | **~8s total** | Good balance |
| `mistralai/mistral-7b-instruct:free` | Free | ~4s | **~16s total** | Good quality |
| `meta-llama/llama-3.3-70b-instruct` | Free* | ~15s | **~60s total** | Best quality |

> *Free on OpenRouter with rate limits. The pipeline uses 4 parallel `ThreadPoolExecutor` workers to minimize wall time regardless of model.

**Why entity extraction is the slow step:**
- The pipeline samples 20 chunks from each priority section (Risk Factors, MD&A, Business Overview, News) = 79 chunks total
- These are grouped into batches of 5 chunks per API call = 16 API calls
- Each API call sends ~2,500 tokens of SEC text and receives back structured JSON with entities and relationships
- Network round-trip + LLM inference time is the fundamental limit — not Python code

**Output:**
```
data/AAPL/graphrag/
├── chunks.json                  (489 chunks, ~4 MB)
├── entities.json                (~153 deduplicated entities)
├── relationships.json           (~209 relationships)
├── extractions_per_chunk.json   (raw per-chunk results, enables --skip-extraction)
├── graph.pkl                    (NetworkX DiGraph, ~2 MB)
├── graph_summary.json           (node/edge counts by type)
└── graphrag_report.json
```

**Graph statistics (AAPL example):**
```
Total nodes:  ~248  (Company, Metric, MacroIndicator, RiskFactor, Segment, Chunk, ...)
Total edges:  ~600  (HAS_RISK, IMPACTS, LINKED_TO, SOURCE_CHUNK, ...)
Avg degree:   ~4.8
```

**Re-run without re-calling the LLM (fast mode):**
```bash
python main.py AAPL --build-graph --skip-extraction
# Loads entities.json + relationships.json from disk
# Skips all LLM API calls
# Total time: ~20s
```

---

### Step 3 — Fundamental Analysis Agent

```bash
python main.py AAPL --analyze-fundamentals
```

Runs 4 focused retrieval + LLM analysis passes, then synthesizes into a single verdict.

| Pass | Query focus | Retrieval method |
|---|---|---|
| 1 — Business Quality | Revenue breakdown, segments, growth drivers | Hybrid (vector + graph) |
| 2 — Financial Health | Margins, cash flow, debt, valuation ratios | Vector (MD&A section) + structured JSON |
| 3 — Management Quality | Executive team, insider transactions, guidance accuracy | Hybrid + company_profile.json |
| 4 — Competitive Position | Market share, moat, competitor threats, AI disruption risk | Vector (Business Overview) + hybrid |
| Synthesis | Combines all 4 passes into overall score + verdict | LLM synthesis call |

**Latency: ~60–90s** (5 LLM calls × ~12–15s each)

**Output saved:** `data/AAPL/analysis/fundamental_analysis.json`

**Sample terminal output:**
```
╭──────────────────────────────────────────────────────────╮
│  FUNDAMENTAL ANALYSIS — AAPL                              │
│  Generated: 2026-05-17 14:32:01 UTC                       │
╰──────────────────────────────────────────────────────────╯

OVERALL SCORE:  7.2 / 10  (STRONG)
VERDICT:        BUY_LEANING  (Confidence: 72%)

STRENGTHS                              WEAKNESSES
✓ wide moat                            ⚠ China revenue concentration
✓ diversified revenue streams          ⚠ smartphone market saturation
✓ high free cash flow generation       ⚠ regulatory headwinds in EU

KEY METRICS  P/E: 35.8x | Net Margin: 27.1% | FCF: $93.0B | D/E: 1.76x
SOURCES CITED: 18
```

Results are cached for 24 hours. Force refresh with:
```bash
python main.py AAPL --analyze-fundamentals --refresh
```

---

### Step 4 — Risk Agent

```bash
python main.py AAPL --analyze-risk
```

Four passes focused on risk dimensions:

| Pass | Focus |
|---|---|
| 1 — Company-specific risks | 10-K Risk Factors section — concentration, supply chain, regulatory |
| 2 — Macro risks | Interest rates, inflation, FX, recession impact via FRED data |
| 3 — Risk interconnections | How risks compound (e.g. China risk → revenue → margins) |
| 4 — Bear case | Worst-case 12-month scenario with probability estimate |
| Synthesis | Risk severity score (0–10), top 5 risks ranked |

**Latency: ~60–90s** (5 LLM calls)

**Output saved:** `data/AAPL/analysis/risk_analysis.json`

---

### Step 5 — Sentiment Agent

```bash
python main.py AAPL --analyze-sentiment
```

Processes news articles in batches of 10, computes source-weighted sentiment scores with recency decay, identifies dominant themes, and produces a market mood verdict.

**Source weight system:** Bloomberg / Reuters / WSJ = 1.0 · CNBC / Fortune = 0.8 · Generic blogs = 0.4

**Latency: ~30–60s** (varies with number of articles, typically 90–100 for 30-day lookback)

**Output saved:** `data/AAPL/analysis/sentiment_analysis.json`

---

### Step 6 — Lead Analyst Agent (Investment Memo)

```bash
python main.py AAPL --lead-analyst
```

Synthesizes all three prior analyses into the final investment recommendation:

1. **Disagreement resolution** — detects where Fundamental, Risk, and Sentiment agents conflict and adjudicates
2. **Bull/bear case writing** — formulates the strongest argument for each side
3. **Quantitative scoring** — weighted composite score (pure Python math, no LLM)
4. **Investment memo** — full equity research report with price target

**Composite scoring formula:**
```
Composite = 0.45 × fundamental_score
          + 0.35 × (1 − risk_score/10)
          + 0.20 × sentiment_normalized
```

**Latency: ~45–60s** (3 LLM calls)

**Output saved:**
```
data/AAPL/analysis/
├── investment_memo.json   (structured data)
├── investment_memo.md     (Markdown report)
└── investment_memo.pdf    (print-ready PDF via ReportLab)
```

---

### Run everything in one command

```bash
# Step 1 + Step 2 together
python main.py AAPL --full

# Step 1, then build graph, then full agent pipeline
python main.py AAPL --full
python main.py AAPL --analyze-fundamentals
python main.py AAPL --analyze-risk
python main.py AAPL --analyze-sentiment
python main.py AAPL --lead-analyst
```

---

### Query the knowledge graph directly

```bash
python main.py AAPL --query "What are the main risk factors connected to China revenue?"
```

Returns combined context: matched text chunks + graph entity traversal + financial metrics.

---

## Total End-to-End Latency

| Step | Fresh run | Cached / Skip |
|---|---|---|
| Step 1 — Ingestion | 30–60s | N/A (always fetches live) |
| Step 2 — GraphRAG (Gemini Flash) | ~85s | ~20s (`--skip-extraction`) |
| Step 2 — GraphRAG (Llama 70B) | ~240s | ~20s (`--skip-extraction`) |
| Step 3 — Fundamental Agent | 60–90s | ~1s (24h cache) |
| Step 4 — Risk Agent | 60–90s | ~1s (24h cache) |
| Step 5 — Sentiment Agent | 30–60s | ~1s (24h cache) |
| Step 6 — Lead Analyst | 45–60s | ~1s (24h cache) |
| **Total (Gemini Flash model)** | **~5–6 min** | **~25s second run** |
| **Total (Llama 70B model)** | **~8–10 min** | **~25s second run** |

> The dominant cost is always LLM API round-trips. Everything else (chunking, embedding, graph build, PDF generation) runs in pure Python locally and takes under 20 seconds combined.

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Data ingestion** | Finnhub, Yahoo Finance, FRED, SEC EDGAR | Free APIs with broad coverage |
| **Chunking** | tiktoken (`cl100k_base`) | Token-accurate splitting respects SEC section boundaries |
| **Entity extraction** | OpenRouter (any free LLM) | Cost-zero entity/relationship extraction from SEC text |
| **Embeddings** | `sentence-transformers` · `all-MiniLM-L6-v2` | 384-dim, 80MB, fully local — zero API cost at query time |
| **Vector store** | ChromaDB (persistent) | Local, no server needed, cosine similarity search |
| **Knowledge graph** | NetworkX `DiGraph` | Lightweight directed graph, serialized to `.pkl` |
| **Hybrid retrieval** | Vector search + graph traversal (BFS, depth=2) | Answers relational questions standard RAG cannot |
| **Agents** | OpenRouter (any free LLM) | 4 specialized agents with structured JSON outputs |
| **Output** | ReportLab (PDF) + Markdown | Recruiter-ready investment memo |
| **Parallelism** | `ThreadPoolExecutor` (4 workers) | 4× speedup on extraction API calls |

---

## Configuration Reference

All settings live in `config.py` and are overridable via `.env`.

| Setting | Default | Description |
|---|---|---|
| `OPENROUTER_MODEL` | `meta-llama/llama-3.3-70b-instruct` | LLM for extraction and agents |
| `EXTRACTION_BATCH_SIZE` | `5` | Chunks per extraction API call |
| `MAX_CHUNKS_PER_SECTION` | `20` | Cap per priority section (controls extraction time) |
| `CHUNK_SIZE_TOKENS` | `512` | Target tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | `64` | Overlap between adjacent chunks |
| `MIN_CHUNK_TOKENS` | `100` | Skip chunks smaller than this |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformers model |
| `MAX_SEC_FILINGS` | `3` | Number of SEC filings to download |
| `NEWS_LOOKBACK_DAYS` | `30` | News article lookback window |
| `AGENT_RETRIEVAL_K` | `8` | Vector search results per agent query |
| `AGENT_GRAPH_DEPTH` | `2` | Graph traversal depth for hybrid retrieval |
| `AGENT_MAX_TOKENS` | `4000` | Max tokens per agent LLM call |

---

## CLI Reference

```bash
python main.py AAPL                          # Step 1: ingest data
python main.py AAPL --build-graph            # Step 2: build GraphRAG index
python main.py AAPL --build-graph --skip-extraction  # Step 2: rebuild graph, skip LLM
python main.py AAPL --full                   # Step 1 + Step 2 together
python main.py AAPL --analyze-fundamentals   # Step 3: fundamental analysis
python main.py AAPL --analyze-risk           # Step 4: risk analysis
python main.py AAPL --analyze-sentiment      # Step 5: sentiment analysis
python main.py AAPL --lead-analyst           # Step 6: investment memo
python main.py AAPL --query "What risks..."  # Query GraphRAG index directly
python main.py AAPL --validate               # Check all API keys are set
python main.py AAPL --only price             # Run only one Step 1 fetcher
python main.py AAPL --skip macro             # Skip one Step 1 fetcher
python main.py AAPL --analyze-fundamentals --refresh  # Ignore 24h cache
```

---

## Data Flow

```
SEC EDGAR ──────────────────────────────────┐
Yahoo Finance ───────────────────────────── │ ingestion/
Finnhub (company + news) ────────────────── │ pipeline.py
FRED (macro) ────────────────────────────── │
                                            ▼
                              data/{ticker}/*.json
                              data/{ticker}/sec_filings/*.txt
                                            │
                                            ▼
                              graphrag/chunker.py
                              489 chunks (section-aware, token-bounded)
                                            │
                                            ▼
                              graphrag/extractor.py
                              LLM → 153 entities, 209 relationships
                              (4 parallel workers, 16 API calls)
                                            │
                          ┌─────────────────┴──────────────────┐
                          ▼                                      ▼
              graphrag/embedder.py                  graphrag/graph_builder.py
              all-MiniLM-L6-v2 (local)              NetworkX DiGraph
              489 × 384-dim vectors                 ~248 nodes, ~600 edges
                          │                                      │
                          ▼                                      │
              graphrag/vector_store.py                           │
              ChromaDB (persistent)                              │
                          │                                      │
                          └─────────────────┬──────────────────┘
                                            ▼
                              graphrag/retriever.py
                              HybridRetriever
                              (vector search + graph BFS)
                                            │
                          ┌─────────────────┴──────────────────────────────┐
                          ▼           ▼              ▼               ▼
                   Fundamental     Risk         Sentiment        Lead Analyst
                     Agent        Agent           Agent            Agent
                   (4 passes)   (4 passes)   (batch articles)   (3 passes)
                          │           │              │               │
                          └───────────┴──────────────┴───────────────┘
                                            │
                                            ▼
                                  investment_memo.md
                                  investment_memo.pdf
```

---

## GraphRAG vs Standard RAG

Standard RAG retrieves text chunks by similarity. GraphRAG adds a knowledge graph layer on top:

| Question | Standard RAG | GraphRAG |
|---|---|---|
| "What are Apple's risk factors?" | Returns relevant chunks | Returns chunks + graph shows RiskFactor → Company edges |
| "How does China risk connect to iPhone revenue?" | May miss the link | Traverses: geo_china → IMPACTS → segment_iphone → GENERATES_REVENUE → Metric |
| "Which risks worsened year-over-year?" | Cannot compare across filings | Graph edges `WORSENED_VS_PRIOR` connect Metric nodes across years |
| "What news is linked to the Services segment?" | Keyword match only | Graph path: NewsArticle → MENTIONED_IN → Concept → LINKED_TO → Segment |

---

## Extending to Another Ticker

```bash
python main.py TSLA
python main.py TSLA --build-graph
python main.py TSLA --analyze-fundamentals
```

Each ticker gets its own isolated `data/{TICKER}/` directory. No changes to code needed.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'dotenv'`**
```bash
# Make sure you're in the virtual environment
source env/bin/activate
pip install -r requirements.txt
```

**Extraction takes very long**
```bash
# Switch to a faster free model in .env:
OPENROUTER_MODEL = google/gemini-2.0-flash-exp:free

# Or skip extraction entirely after first run:
python main.py AAPL --build-graph --skip-extraction
```

**`FileNotFoundError: Step 1 data missing`**
```bash
# Run Step 1 first
python main.py AAPL
```

**`FileNotFoundError: GraphRAG index not found`**
```bash
# Run Step 2 first
python main.py AAPL --build-graph
```

**Graph has 169 nodes instead of ~248**
```bash
# extractions_per_chunk.json missing — run fresh extraction once:
python main.py AAPL --build-graph   # (no --skip-extraction)
```

**Rate limit errors from OpenRouter**
Retry after 30 seconds. The pipeline automatically handles 429s with exponential backoff. Switching to a different free model often resolves persistent rate limits.

---

## License

MIT — free to use, modify, and distribute.
