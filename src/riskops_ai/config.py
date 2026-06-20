from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / os.getenv("DATA_DIR", "data")
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
POLICY_DIR = DATA_DIR / "policies"
INDEX_DIR = DATA_DIR / "index"

@dataclass(frozen=True)
class LLMSettings:
    provider: str = os.getenv("LLM_PROVIDER", "offline").lower().strip()
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


def ensure_dirs() -> None:
    for path in [BRONZE_DIR, SILVER_DIR, GOLD_DIR, POLICY_DIR, INDEX_DIR]:
        path.mkdir(parents=True, exist_ok=True)
