from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from riskops_ai.config import GOLD_DIR
from riskops_ai.pipelines.generate_synthetic_data import generate_all
from riskops_ai.pipelines.build_marts_duckdb import build_all


def test_build_pipeline_creates_gold_marts():
    generate_all()
    counts = build_all()
    assert counts["gold_portfolio_summary"] > 0
    assert (GOLD_DIR / "gold_data_quality_report.parquet").exists()
