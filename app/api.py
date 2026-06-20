from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pydantic import BaseModel, Field
from fastapi import FastAPI

from riskops_ai.agent import RiskOpsAgent
from riskops_ai.tools import data_quality_diagnostics, portfolio_summary

app = FastAPI(title="RiskOps AI", version="0.1.0")
agent = RiskOpsAgent()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["Why did approval rate drop this month?"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "riskops-ai"}


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    return agent.ask(request.question).__dict__


@app.get("/portfolio")
def portfolio() -> dict:
    return {"rows": portfolio_summary()}


@app.get("/quality")
def quality() -> dict:
    return {"rows": data_quality_diagnostics(only_failures=False)}
