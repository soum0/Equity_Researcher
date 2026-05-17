import os

# Load secrets: Streamlit Cloud first, then .env for local dev
from dotenv import load_dotenv
load_dotenv()

def _get(key: str, default: str = "") -> str:
    # Try st.secrets first (Streamlit Cloud / local secrets.toml)
    try:
        import streamlit as st
        val = st.secrets.get(key)
        if val is not None:
            return str(val)
    except Exception:
        pass
    return os.getenv(key, default) or default


class Config:
    # API Keys
    FINNHUB_API_KEY = _get("FINNHUB_API_KEY")
    NEWSAPI_KEY     = _get("NEWSAPI_KEY")
    FRED_API_KEY    = _get("FRED_API_KEY")
    SEC_USER_AGENT  = _get("SEC_USER_AGENT", "FinancialPipeline contact@example.com")

    # Ingestion settings
    NEWS_LOOKBACK_DAYS = int(os.getenv("NEWS_LOOKBACK_DAYS", 30))
    MAX_SEC_FILINGS = int(os.getenv("MAX_SEC_FILINGS", 3))
    PRICE_HISTORY_PERIOD = "2y"
    # On Render, /data is the persistent disk mount path; locally use data/
    DATA_DIR = "/data" if os.path.exists("/data") else "data"

    # FRED series to fetch
    FRED_SERIES = {
        "federal_funds_rate": "FEDFUNDS",
        "cpi_inflation": "CPIAUCSL",
        "gdp_growth": "A191RL1Q225SBEA",
        "unemployment_rate": "UNRATE",
        "10y_treasury_yield": "GS10",
        "vix_index": "VIXCLS",
    }

    # SEC EDGAR base URLs
    SEC_BASE_URL = "https://data.sec.gov"
    SEC_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"

    # Rate limiting
    FINNHUB_DELAY_SECONDS = 0.5
    SEC_DELAY_SECONDS = 0.1
    FRED_DELAY_SECONDS = 0.2

    # ── GraphRAG Settings ──────────────────────────────────────────────────
    OPENROUTER_API_KEY  = _get("OPENROUTER_API_KEY")
    OPENROUTER_MODEL    = _get("OPENROUTER_MODEL", "llama-3.1-8b-instant")
    OPENROUTER_BASE_URL = _get("OPENROUTER_BASE_URL", "https://api.groq.com/openai/v1")

    CHUNK_SIZE_TOKENS = 512
    CHUNK_OVERLAP_TOKENS = 64
    MIN_CHUNK_TOKENS = 100
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    # Template strings — format with ticker at runtime
    CHROMA_DB_DIR = DATA_DIR + "/{ticker}/graphrag/chroma"
    GRAPH_FILE    = DATA_DIR + "/{ticker}/graphrag/graph.pkl"

    EXTRACTION_MODEL = _get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
    EXTRACTION_BATCH_SIZE = 3
    MAX_CHUNKS_PER_SECTION = 20   # cap per priority section to limit API calls

    INDEX_SEC_FILINGS = True
    INDEX_NEWS = True
    INDEX_FINANCIALS = True
    INDEX_MACRO = True

    # ── Agent Settings ─────────────────────────────────────────────────────────
    AGENT_BASE_URL = _get("OPENROUTER_BASE_URL", "https://api.groq.com/openai/v1")
    AGENT_API_KEY  = _get("OPENROUTER_API_KEY")
    AGENT_MODEL    = _get("OPENROUTER_MODEL", "llama-3.3-70b-versatile")
    AGENT_MAX_TOKENS  = 1500
    AGENT_TEMPERATURE = 0.1
    AGENT_RETRIEVAL_K = 4
    AGENT_GRAPH_DEPTH = 2
