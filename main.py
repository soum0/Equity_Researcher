"""
Usage:
  python main.py AAPL --pipeline          # Full LangGraph pipeline (Steps 1-7)
  python main.py AAPL --pipeline --refresh # Force re-run all steps
  python main.py AAPL --resume            # Resume from last checkpoint
  python main.py AAPL --pdf               # Just regenerate PDF

  python main.py AAPL                          # Step 1 only: ingest data
  python main.py AAPL --build-graph            # Step 2: build GraphRAG index
  python main.py AAPL --full                   # Step 1 + Step 2 together
  python main.py AAPL --analyze-fundamentals   # Step 3: fundamental analysis
  python main.py AAPL --analyze-risk           # Step 4: risk analysis
  python main.py AAPL --analyze-sentiment      # Step 5: sentiment analysis
  python main.py AAPL --analyze-all            # Steps 3 + 4 + 5 together
  python main.py AAPL --generate-memo          # Step 6: Lead Analyst memo
  python main.py AAPL --full-analysis          # Steps 3 + 4 + 5 + 6 together
  python main.py AAPL --refresh                # Force re-run (ignore cache)
  python main.py AAPL --query "What are risks" # Test retrieval
  python main.py AAPL --validate               # Check .env keys are set
  python main.py AAPL --only price             # Run only one Step 1 fetcher
  python main.py AAPL --skip macro             # Skip one Step 1 fetcher
  python main.py AAPL --build-graph --skip-extraction  # Rebuild graph from cached entities
"""
import argparse
import os
import sys

from config import Config
from ingestion.company_fetcher import CompanyFetcher
from ingestion.macro_fetcher import MacroFetcher
from ingestion.news_fetcher import NewsFetcher
from ingestion.pipeline import DataIngestionPipeline
from ingestion.price_fetcher import PriceFetcher
from ingestion.sec_fetcher import SECFetcher
from ingestion.utils import ensure_dir, setup_logger

FETCHER_KEYS = ["price", "company", "news", "sec", "macro"]

BANNER = """\
+----------------------------------------------+
|  AI Financial Pipeline -- Data Ingestion     |
+----------------------------------------------+"""


def validate_env(config: Config) -> bool:
    checks = [
        ("FINNHUB_API_KEY", config.FINNHUB_API_KEY),
        ("NEWSAPI_KEY", config.NEWSAPI_KEY),
        ("FRED_API_KEY", config.FRED_API_KEY),
        ("SEC_USER_AGENT", config.SEC_USER_AGENT),
    ]
    all_ok = True
    for name, val in checks:
        if val:
            print(f"  [{name}]  SET")
        else:
            print(f"  [{name}]  MISSING")
            all_ok = False
    return all_ok


def run_single_fetcher(fetcher_key: str, ticker: str, config: Config):
    ensure_dir(f"{config.DATA_DIR}/{ticker}")

    if fetcher_key == "price":
        result = PriceFetcher(ticker, config).fetch_all()
        print(f"Price fetcher done. Data points: {result.get('price_history', {}).get('data_points', 0)}")

    elif fetcher_key == "company":
        result = CompanyFetcher(ticker, config).fetch_all()
        print(f"Company fetcher done. Name: {result.get('name')}")

    elif fetcher_key == "news":
        company_data = {}
        try:
            company_data = CompanyFetcher(ticker, config).fetch_all()
        except Exception:
            pass
        company_name = company_data.get("name") or ticker
        result = NewsFetcher(ticker, company_name, config).fetch_all()
        print(f"News fetcher done. Articles: {result.get('total_articles', 0)}")

    elif fetcher_key == "sec":
        result = SECFetcher(ticker, config).fetch_all()
        print(f"SEC fetcher done. Filings: {len(result.get('filings', []))}")

    elif fetcher_key == "macro":
        result = MacroFetcher(config).fetch_all(ticker=ticker)
        n = len([k for k, v in result.get("indicators", {}).items() if "error" not in v])
        print(f"Macro fetcher done. Series: {n}")

    else:
        print(f"Unknown fetcher key: {fetcher_key}. Valid: {FETCHER_KEYS}")
        sys.exit(1)


GRAPHRAG_BANNER = """\
+----------------------------------------------+
|  AI Financial Pipeline -- GraphRAG Build     |
+----------------------------------------------+"""


def _run_build_graph(ticker: str, config: Config, skip_extraction: bool = False):
    from graphrag.pipeline import GraphRAGPipeline
    print(GRAPHRAG_BANNER)
    print(f"\nTicker: {ticker}")
    print(f"Source: data/{ticker}/\n")
    pipeline = GraphRAGPipeline(ticker, config)
    pipeline.build(skip_extraction=skip_extraction)


def _run_query(ticker: str, query: str, config: Config):
    from graphrag.pipeline import GraphRAGPipeline
    print(f"\nQuerying GraphRAG index for {ticker}...")
    print(f"Query: {query}\n")
    retriever = GraphRAGPipeline(ticker, config).load()
    ctx = retriever.hybrid_retrieve(query)
    print(ctx["combined_context"])
    sources = ctx.get("sources", [])
    if sources:
        print("Sources:", ", ".join(sources))


def main():
    parser = argparse.ArgumentParser(
        description="AI Financial Pipeline — Data Ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("ticker", type=str, help="Stock ticker symbol (e.g. AAPL)")
    parser.add_argument(
        "--only",
        choices=FETCHER_KEYS,
        metavar="FETCHER",
        help=f"Run only one fetcher: {FETCHER_KEYS}",
    )
    parser.add_argument(
        "--skip",
        choices=FETCHER_KEYS,
        metavar="FETCHER",
        help=f"Skip one fetcher: {FETCHER_KEYS}",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Check that all required .env keys are set",
    )
    parser.add_argument(
        "--build-graph",
        action="store_true",
        help="Step 2: build GraphRAG knowledge graph index",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run Step 1 (ingestion) then Step 2 (GraphRAG) together",
    )
    parser.add_argument(
        "--query",
        type=str,
        metavar="QUERY",
        help="Run a hybrid retrieval query against an existing GraphRAG index",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Rebuild graph from cached entities.json without re-calling Claude",
    )
    parser.add_argument(
        "--analyze-fundamentals",
        action="store_true",
        help="Step 3: run Fundamental Analysis Agent",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-run analysis, ignoring cached results",
    )
    parser.add_argument(
        "--analyze-risk",
        action="store_true",
        help="Step 4: run Risk Analysis Agent",
    )
    parser.add_argument(
        "--analyze-all",
        action="store_true",
        help="Steps 3 + 4 + 5: run Fundamental, Risk, and Sentiment Analysis",
    )
    parser.add_argument(
        "--analyze-sentiment",
        action="store_true",
        help="Step 5: run Sentiment Analysis Agent",
    )
    parser.add_argument(
        "--generate-memo",
        action="store_true",
        help="Step 6: run Lead Analyst Agent and generate investment memo",
    )
    parser.add_argument(
        "--full-analysis",
        action="store_true",
        help="Steps 3+4+5+6: run all analysis agents end-to-end",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Step 7: run full LangGraph pipeline (Steps 1-7) with checkpointing",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume pipeline from last checkpoint",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Regenerate PDF from existing investment_memo.md and .json",
    )
    args = parser.parse_args()

    ticker = args.ticker.upper()
    config = Config()
    logger = setup_logger("main")

    print(BANNER)
    print(f"\nTicker: {ticker}")
    print(f"Output: {config.DATA_DIR}/{ticker}/\n")

    if args.validate:
        print("Validating environment variables:")
        ok = validate_env(config)
        if config.OPENROUTER_API_KEY:
            print(f"  [OPENROUTER_API_KEY]  SET")
        else:
            print(f"  [OPENROUTER_API_KEY]  MISSING")
            ok = False
        sys.exit(0 if ok else 1)

    if any([args.analyze_all, args.analyze_fundamentals, args.analyze_risk, args.analyze_sentiment,
            args.generate_memo, args.full_analysis]):
        if not os.getenv("OPENROUTER_API_KEY"):
            print("ERROR: OPENROUTER_API_KEY not set in .env")
            sys.exit(1)

    if args.pipeline or args.resume:
        from pipeline.graph import run_pipeline
        if args.resume:
            from pipeline.checkpointer import PipelineCheckpointer
            latest = PipelineCheckpointer(ticker).get_latest_checkpoint()
            print(f"Resuming from checkpoint: {latest or 'beginning'}")
        run_pipeline(ticker, refresh=args.refresh)
        return

    if args.pdf:
        import json as _json
        from pipeline.pdf_generator import PDFGenerator
        md_path   = f"{config.DATA_DIR}/{ticker}/analysis/investment_memo.md"
        json_path = f"{config.DATA_DIR}/{ticker}/analysis/investment_memo.json"
        with open(md_path, encoding="utf-8") as f:
            memo_md = f.read()
        with open(json_path, encoding="utf-8") as f:
            memo_json = _json.load(f)
        pdf_path = PDFGenerator(ticker).generate(memo_md, memo_json)
        print(f"PDF generated: {pdf_path}")
        print(f"Open with: open {pdf_path}")
        return

    if args.full_analysis:
        from agents.fundamental_agent import FundamentalAgent
        from agents.lead_analyst_agent import LeadAnalystAgent
        from agents.risk_agent import RiskAgent
        from agents.sentiment_agent import SentimentAgent
        FundamentalAgent(ticker, config).analyze(refresh=args.refresh)
        RiskAgent(ticker, config).analyze(refresh=args.refresh)
        SentimentAgent(ticker, config).analyze(refresh=args.refresh)
        LeadAnalystAgent(ticker, config).analyze(refresh=args.refresh)
        return

    if args.generate_memo:
        from agents.lead_analyst_agent import LeadAnalystAgent
        LeadAnalystAgent(ticker, config).analyze(refresh=args.refresh)
        return

    if args.analyze_all:
        from agents.fundamental_agent import FundamentalAgent
        from agents.risk_agent import RiskAgent
        from agents.sentiment_agent import SentimentAgent
        FundamentalAgent(ticker, config).analyze(refresh=args.refresh)
        RiskAgent(ticker, config).analyze(refresh=args.refresh)
        SentimentAgent(ticker, config).analyze(refresh=args.refresh)
        return

    if args.analyze_fundamentals:
        from agents.fundamental_agent import FundamentalAgent
        FundamentalAgent(ticker, config).analyze(refresh=args.refresh)
        return

    if args.analyze_risk:
        from agents.risk_agent import RiskAgent
        RiskAgent(ticker, config).analyze(refresh=args.refresh)
        return

    if args.analyze_sentiment:
        from agents.sentiment_agent import SentimentAgent
        SentimentAgent(ticker, config).analyze(refresh=args.refresh)
        return

    if args.query:
        _run_query(ticker, args.query, config)
        return

    if args.build_graph:
        _run_build_graph(ticker, config, skip_extraction=args.skip_extraction)
        return

    if args.full:
        pipeline = DataIngestionPipeline(ticker, config)
        pipeline.run()
        print()
        _run_build_graph(ticker, config, skip_extraction=args.skip_extraction)
        return

    if args.only:
        run_single_fetcher(args.only, ticker, config)
        return

    if args.skip:
        skipped = args.skip
        logger.info(f"Skipping fetcher: {skipped}")

        ensure_dir(f"{config.DATA_DIR}/{ticker}")
        results: dict = {}
        errors: list[str] = []
        company_name = ticker

        step = 1
        for key in FETCHER_KEYS:
            if key == skipped:
                print(f"[{step}/5] {key.upper()} — SKIPPED")
                step += 1
                continue

            print(f"[{step}/5] Fetching {key}...")
            try:
                if key == "price":
                    data = PriceFetcher(ticker, config).fetch_all()
                    results["price"] = data
                elif key == "company":
                    data = CompanyFetcher(ticker, config).fetch_all()
                    results["company"] = data
                    company_name = data.get("name") or ticker
                elif key == "news":
                    data = NewsFetcher(ticker, company_name, config).fetch_all()
                    results["news"] = data
                elif key == "sec":
                    data = SECFetcher(ticker, config).fetch_all()
                    results["sec"] = data
                elif key == "macro":
                    data = MacroFetcher(config).fetch_all(ticker=ticker)
                    results["macro"] = data
                print(f"[{step}/5] {key.upper()} — OK")
            except Exception as exc:
                errors.append(f"{key}: {exc}")
                results[key] = {}
                print(f"[{step}/5] {key.upper()} — FAILED: {exc}")
                logger.error(f"[{ticker}] {key} failed: {exc}")
            step += 1

        pipeline = DataIngestionPipeline(ticker, config)
        report = pipeline.generate_report(results, errors, company_name)
        from ingestion.utils import save_json
        save_json(report, f"{config.DATA_DIR}/{ticker}/ingestion_report.json")
        pipeline._print_summary(report)
        return

    # Full pipeline
    pipeline = DataIngestionPipeline(ticker, config)
    pipeline.run()


if __name__ == "__main__":
    main()
