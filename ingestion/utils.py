import json
import logging
import os
import time
from html import unescape
from pathlib import Path


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(logs_dir / f"{name}.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
    logger = setup_logger("retry")
    last_exc = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            wait = base_delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {exc}. Retrying in {wait}s...")
            time.sleep(wait)
    raise last_exc


def rate_limit(seconds: float):
    time.sleep(seconds)


def save_json(data: dict | list, filepath: str):
    logger = setup_logger("utils")
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved: {filepath}")


def load_json(filepath: str) -> dict | list | None:
    path = Path(filepath)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def sanitize_text(text: str) -> str:
    if not text:
        return ""
    text = unescape(text)
    text = text.replace("\x00", "")
    text = " ".join(text.split())
    return text.strip()
