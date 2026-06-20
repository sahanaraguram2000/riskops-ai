from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from riskops_ai.llm import get_llm, LLMClient
from riskops_ai.tools import answer_question_with_tools

SYSTEM_PROMPT = """
You are RiskOps AI, a careful analyst for NBFC credit-risk data.
Answer using the provided tool evidence and policy context only.
When evidence is incomplete, say what is missing.
Always provide practical next steps.
Do not claim this is real customer data; it is a synthetic portfolio demo.
""".strip()


@dataclass
class AgentResponse:
    question: str
    answer: str
    tools_used: list[str]
    evidence: dict[str, Any]
    citations: list[str]
    latency_ms: int


class RiskOpsAgent:
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or get_llm()

    def ask(self, question: str) -> AgentResponse:
        started = time.perf_counter()
        tool_payload = answer_question_with_tools(question)
        evidence = tool_payload["evidence"]
        citations = [c["source"] for c in evidence.get("policy_context", [])]

        prompt = f"""
{SYSTEM_PROMPT}

User question:
{question}

Tool evidence JSON:
{json.dumps(evidence, default=str, indent=2)[:12000]}

Write a concise executive answer with:
1. Direct finding
2. Evidence from metrics
3. Policy citation names
4. Recommended action
""".strip()
        answer = self.llm.complete(prompt)
        if citations:
            answer = answer.rstrip() + "\n\nPolicy sources: " + ", ".join(citations)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return AgentResponse(
            question=question,
            answer=answer,
            tools_used=tool_payload["tools_used"],
            evidence=evidence,
            citations=citations,
            latency_ms=latency_ms,
        )


def ask(question: str) -> dict[str, Any]:
    response = RiskOpsAgent().ask(question)
    return response.__dict__
