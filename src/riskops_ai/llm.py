from __future__ import annotations

import json
import requests
from dataclasses import dataclass
from typing import Protocol

from riskops_ai.config import LLMSettings


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


@dataclass
class OfflineLLM:
    """Deterministic zero-cost fallback.

    This makes the project demo-able even without API keys or local GPU/LLM.
    """
    def complete(self, prompt: str) -> str:
        lower = prompt.lower()
        if "approval" in lower and "drop" in lower:
            return (
                "The approval-rate decline is most likely driven by a mix shift toward higher-risk Partner-channel applications, "
                "especially self-employed and small-business applicants in the latest month. The evidence shows higher average risk score "
                "and policy rules requiring manual review or decline for weak bureau, recent delinquencies, and high enquiry counts. "
                "Recommended action: validate upstream Partner sourcing, compare bureau-score distribution month-over-month, and review cutoff changes."
            )
        if "data quality" in lower or "dq" in lower or "quality" in lower:
            return (
                "The data-quality diagnostics found issues that should be triaged before using the affected marts for risk decisions. "
                "Prioritize failed primary-key, repayment schedule, null-rate, and approval-anomaly checks. Each failed rule should have an owner, "
                "impact assessment, and fix SLA."
            )
        if "dpd" in lower or "delinquency" in lower:
            return (
                "Delinquency risk is concentrated in higher-risk bands and later repayment buckets. Focus collections on 30+ DPD accounts, "
                "repeat delinquency, high outstanding principal, and failed promise-to-pay cases."
            )
        return (
            "Based on the retrieved policy context and analytical marts, the portfolio should be reviewed by risk band, channel, product, "
            "and employment type. Check approval movement, DPD buckets, and data-quality failures before drawing final business conclusions."
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
