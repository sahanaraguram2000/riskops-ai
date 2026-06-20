from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

import requests

from riskops_ai.config import LLMSettings


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


def _extract_user_question(prompt: str) -> str:
    match = re.search(r"User question:\s*(.*?)\n\nTool evidence JSON:", prompt, flags=re.S)
    if match:
        return match.group(1).strip()
    return prompt[:500]


def _extract_evidence(prompt: str) -> dict[str, Any]:
    match = re.search(r"Tool evidence JSON:\s*(\{.*?\})\s*\n\nWrite a concise", prompt, flags=re.S)
    if not match:
        return {}
    raw = match.group(1)
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return str(value).replace(" 00:00:00", "")


def _policy_sources(evidence: dict[str, Any]) -> str:
    sources = []
    for chunk in evidence.get("policy_context", [])[:3]:
        source = chunk.get("source")
        if source and source not in sources:
            sources.append(source)
    return ", ".join(sources) if sources else "No policy source retrieved"


@dataclass
class OfflineLLM:
    """Deterministic zero-cost fallback.

    It creates evidence-aware responses from the routed tools. This keeps the
    public Streamlit deployment free while avoiding the same generic response
    for every sample question.
    """

    def complete(self, prompt: str) -> str:
        question = _extract_user_question(prompt).lower()
        evidence = _extract_evidence(prompt)

        if "incident" in question or "anomaly" in question:
            return self._incident_answer(evidence)
        if any(x in question for x in ["quality", "dq", "null", "duplicate", "schema", "failure"]):
            return self._dq_answer(evidence)
        if any(x in question for x in ["dpd", "delinquency", "default", "overdue", "repayment", "30+"]):
            return self._delinquency_answer(evidence)
        if any(x in question for x in ["approval", "approve", "drop", "funnel", "rejected"]):
            return self._approval_answer(evidence)
        return self._portfolio_answer(evidence)

    def _approval_answer(self, evidence: dict[str, Any]) -> str:
        rows = evidence.get("approval_drop_analysis", [])
        if not rows:
            return "No approval-rate drop was found in the latest available synthetic portfolio window. Recommended action: verify the latest application month and rerun the approval funnel mart."
        top = rows[0]
        return (
            f"**Direct finding:** The largest approval-rate decline is in **{_fmt(top.get('channel'))} / {_fmt(top.get('employment_type'))}**, "
            f"where approval rate moved from **{_fmt(top.get('previous_approval_rate_pct'))}%** to **{_fmt(top.get('latest_approval_rate_pct'))}%** "
            f"in **{_fmt(top.get('latest_month'))}**, a drop of **{_fmt(top.get('approval_drop_pp'))} percentage points**.\n\n"
            f"**Evidence:** The same segment's average risk score increased from **{_fmt(top.get('previous_avg_risk_score'))}** to **{_fmt(top.get('latest_avg_risk_score'))}** "
            f"across **{_fmt(top.get('latest_applications'))} latest applications**, which suggests a mix-shift toward riskier applicants rather than only a reporting issue.\n\n"
            f"**Recommended action:** Compare Partner/Branch sourcing mix, bureau-score distribution, employment-type mix, and cutoff changes month over month. Validate the approval mart before making a policy decision.\n\n"
            f"**Policy context:** {_policy_sources(evidence)}"
        )

    def _delinquency_answer(self, evidence: dict[str, Any]) -> str:
        rows = evidence.get("delinquency_analysis", [])
        if not rows:
            return "No delinquency rows were returned from the DPD mart. Recommended action: check repayment enrichment and DPD bucket logic."
        top = rows[0]
        return (
            f"**Direct finding:** The highest 30+ DPD hotspot is **{_fmt(top.get('product'))} / {_fmt(top.get('channel'))} / {_fmt(top.get('risk_band'))}** "
            f"for due month **{_fmt(top.get('due_month'))}**.\n\n"
            f"**Evidence:** This segment shows **{_fmt(top.get('avg_30_plus_dpd_rate_pct'))}% average 30+ DPD rate** and "
            f"**₹{_fmt(top.get('amount_30_plus_due'))}** amount due in 30+ DPD buckets across **{_fmt(top.get('instalments'))} instalments**.\n\n"
            f"**Recommended action:** Prioritize collections for 30+ DPD accounts, review repeat delinquency, and check whether this is a small-count spike or a broad segment deterioration.\n\n"
            f"**Policy context:** {_policy_sources(evidence)}"
        )

    def _dq_answer(self, evidence: dict[str, Any]) -> str:
        rows = evidence.get("data_quality_diagnostics", [])
        if not rows:
            return "No failed data-quality checks were found. Recommended action: still review freshness and schema drift before using the marts for decisioning."
        critical = [r for r in rows if str(r.get("severity", "")).upper() == "CRITICAL"]
        top = critical[0] if critical else rows[0]
        return (
            f"**Direct finding:** The data-quality layer found **{len(rows)} failed checks**. The highest-priority issue is "
            f"**{_fmt(top.get('table_name'))}.{_fmt(top.get('rule_name'))}** with severity **{_fmt(top.get('severity'))}**.\n\n"
            f"**Evidence:** Metric value is **{_fmt(top.get('metric_value'))}**. Details: {_fmt(top.get('details'))}.\n\n"
            f"**Recommended action:** Assign an owner, validate upstream ingestion for the impacted table, rerun the failed rule after fixing the source/mart, and block risk reporting if the failed rule affects approval, repayment, or bureau fields.\n\n"
            f"**Policy context:** {_policy_sources(evidence)}"
        )

    def _portfolio_answer(self, evidence: dict[str, Any]) -> str:
        rows = evidence.get("portfolio_summary", [])
        if not rows:
            return "No portfolio-summary rows were returned. Recommended action: rebuild the gold portfolio mart."
        total_loans = sum(float(r.get("loans", 0) or 0) for r in rows)
        total_principal = sum(float(r.get("principal_disbursed", 0) or 0) for r in rows)
        top = max(rows, key=lambda r: float(r.get("principal_disbursed", 0) or 0))
        return (
            f"**Direct finding:** The latest portfolio view covers **{int(total_loans)} loans** in the displayed gold mart slice, "
            f"with about **₹{round(total_principal, 2)}** disbursed principal.\n\n"
            f"**Evidence:** The largest displayed segment is **{_fmt(top.get('product'))} / {_fmt(top.get('channel'))} / {_fmt(top.get('risk_band'))}**, "
            f"with **₹{_fmt(top.get('principal_disbursed'))}** principal and average risk score **{_fmt(top.get('avg_risk_score'))}**.\n\n"
            f"**Recommended action:** Review portfolio concentration by product, channel, and risk band; combine this with approval-drop and DPD hotspot analysis before making credit-policy changes.\n\n"
            f"**Policy context:** {_policy_sources(evidence)}"
        )

    def _incident_answer(self, evidence: dict[str, Any]) -> str:
        approval_rows = evidence.get("approval_drop_analysis", [])
        dq_rows = evidence.get("data_quality_diagnostics", [])
        top_approval = approval_rows[0] if approval_rows else {}
        top_dq = dq_rows[0] if dq_rows else {}
        return (
            "**Incident summary:** Approval anomaly detected in the latest synthetic portfolio window.\n\n"
            f"**Impacted area:** {_fmt(top_approval.get('channel', 'Approval funnel'))} / {_fmt(top_approval.get('employment_type', 'Applicant mix'))}.\n\n"
            f"**Observed movement:** Approval rate changed from **{_fmt(top_approval.get('previous_approval_rate_pct'))}%** to **{_fmt(top_approval.get('latest_approval_rate_pct'))}%**, "
            f"a **{_fmt(top_approval.get('approval_drop_pp'))} pp** decline. Average risk score moved from **{_fmt(top_approval.get('previous_avg_risk_score'))}** "
            f"to **{_fmt(top_approval.get('latest_avg_risk_score'))}**.\n\n"
            f"**DQ cross-check:** {_fmt(top_dq.get('table_name', 'No critical DQ rule returned'))} / {_fmt(top_dq.get('rule_name', 'NA'))} "
            f"is the top failed rule to validate before final RCA.\n\n"
            "**Recommended owner action:** Risk analytics should validate applicant mix and cutoff changes; data engineering should validate freshness, schema, and primary/foreign-key checks; business should review sourcing-channel quality.\n\n"
            f"**Policy context:** {_policy_sources(evidence)}"
        )


@dataclass
class OllamaLLM:
    base_url: str
    model: str
    timeout_seconds: int = 45

    def complete(self, prompt: str) -> str:
        url = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        try:
            response = requests.post(url, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
            return response.json().get("response", "").strip() or OfflineLLM().complete(prompt)
        except Exception:
            return OfflineLLM().complete(prompt)


@dataclass
class GeminiLLM:
    api_key: str
    model: str
    timeout_seconds: int = 45

    def complete(self, prompt: str) -> str:
        if not self.api_key:
            return OfflineLLM().complete(prompt)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            response = requests.post(url, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception:
            return OfflineLLM().complete(prompt)


def get_llm(settings: LLMSettings | None = None) -> LLMClient:
    settings = settings or LLMSettings()
    if settings.provider == "ollama":
        return OllamaLLM(settings.ollama_base_url, settings.ollama_model)
    if settings.provider == "gemini":
        return GeminiLLM(settings.gemini_api_key, settings.gemini_model)
    return OfflineLLM()
