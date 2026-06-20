from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from riskops_ai.agent import RiskOpsAgent
from riskops_ai.pipelines.generate_synthetic_data import generate_all
from riskops_ai.pipelines.build_marts_duckdb import build_all
from riskops_ai.rag.policy_index import build_index


def test_agent_returns_answer():
    generate_all()
    build_all()
    build_index()
    response = RiskOpsAgent().ask("Why did approval rate drop this month?")
    assert response.answer
    assert response.tools_used
    assert response.citations
