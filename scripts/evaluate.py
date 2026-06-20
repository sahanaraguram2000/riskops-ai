from __future__ import annotations

import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from riskops_ai.agent import RiskOpsAgent

QUESTIONS = [
    "Why did approval rate drop this month?",
    "Which risk segment has the highest 30+ DPD?",
    "Show data quality issues in repayment data.",
    "Summarize portfolio risk with policy citations.",
]


def simple_metrics(answer: str, citations: list[str]) -> dict:
    return {
        "answer_chars": len(answer),
        "has_policy_citation": int(bool(citations)),
        "mentions_action": int("action" in answer.lower() or "recommended" in answer.lower()),
    }


def main() -> None:
    agent = RiskOpsAgent()
    try:
        import mlflow
        mlflow.set_experiment("riskops-ai-eval")
        use_mlflow = True
    except Exception:
        mlflow = None
        use_mlflow = False

    for question in QUESTIONS:
        result = agent.ask(question)
        metrics = simple_metrics(result.answer, result.citations)
        print("\nQUESTION:", question)
        print("ANSWER:", result.answer[:500], "...")
        print("METRICS:", metrics, "latency_ms=", result.latency_ms)
        if use_mlflow:
            with mlflow.start_run(run_name=question[:40]):
                mlflow.log_param("question", question)
                mlflow.log_param("tools_used", ",".join(result.tools_used))
                mlflow.log_metric("latency_ms", result.latency_ms)
                for k, v in metrics.items():
                    mlflow.log_metric(k, v)
                mlflow.log_text(result.answer, "answer.txt")


if __name__ == "__main__":
    main()
