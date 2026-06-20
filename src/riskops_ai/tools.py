from __future__ import annotations

from dataclasses import asdict
from typing import Any

import duckdb

from riskops_ai.config import GOLD_DIR
from riskops_ai.rag.policy_index import PolicyRetriever


def _parquet_path(name: str) -> str:
    return str((GOLD_DIR / f"{name}.parquet").resolve()).replace("'", "''")


def run_sql(sql: str) -> list[dict[str, Any]]:
    con = duckdb.connect(database=":memory:")
    try:
        return con.execute(sql).df().to_dict(orient="records")
    finally:
        con.close()


def portfolio_summary(limit: int = 12) -> list[dict[str, Any]]:
    sql = f"""
        SELECT *
        FROM read_parquet('{_parquet_path('gold_portfolio_summary')}')
        ORDER BY disbursed_month DESC, principal_disbursed DESC
        LIMIT {int(limit)}
    """
    return run_sql(sql)


def approval_drop_analysis() -> list[dict[str, Any]]:
    sql = f"""
        WITH base AS (
            SELECT *
            FROM read_parquet('{_parquet_path('gold_approval_funnel')}')
        ), monthly AS (
            SELECT application_month, channel, employment_type,
                   SUM(applications) AS applications,
                   SUM(approved_applications) AS approvals,
                   ROUND(100.0 * SUM(approved_applications) / SUM(applications), 2) AS approval_rate_pct,
                   ROUND(AVG(avg_risk_score), 2) AS avg_risk_score
            FROM base
            GROUP BY 1,2,3
        ), ranked AS (
            SELECT *, DENSE_RANK() OVER (ORDER BY application_month DESC) AS rn
            FROM monthly
        ), latest AS (
            SELECT * FROM ranked WHERE rn = 1
        ), previous AS (
            SELECT * FROM ranked WHERE rn = 2
        )
        SELECT
            l.application_month AS latest_month,
            l.channel,
            l.employment_type,
            p.approval_rate_pct AS previous_approval_rate_pct,
            l.approval_rate_pct AS latest_approval_rate_pct,
            ROUND(p.approval_rate_pct - l.approval_rate_pct, 2) AS approval_drop_pp,
            p.avg_risk_score AS previous_avg_risk_score,
            l.avg_risk_score AS latest_avg_risk_score,
            l.applications AS latest_applications
        FROM latest l
        JOIN previous p USING(channel, employment_type)
        WHERE p.approval_rate_pct - l.approval_rate_pct > 0
        ORDER BY approval_drop_pp DESC, latest_applications DESC
        LIMIT 10
    """
    return run_sql(sql)


def delinquency_analysis() -> list[dict[str, Any]]:
    sql = f"""
        SELECT
            due_month,
            product,
            channel,
            risk_band,
            ROUND(SUM(CASE WHEN dpd_bucket IN ('31-60 Moderate','61-90 Severe','90+ Critical') THEN amount_due ELSE 0 END), 2) AS amount_30_plus_due,
            ROUND(AVG(thirty_plus_dpd_rate_pct), 2) AS avg_30_plus_dpd_rate_pct,
            SUM(scheduled_instalments) AS instalments
        FROM read_parquet('{_parquet_path('gold_dpd_bucket_analysis')}')
        GROUP BY 1,2,3,4
        ORDER BY avg_30_plus_dpd_rate_pct DESC, amount_30_plus_due DESC
        LIMIT 10
    """
    return run_sql(sql)


def data_quality_diagnostics(only_failures: bool = True) -> list[dict[str, Any]]:
    where = "WHERE status = 'FAIL'" if only_failures else ""
    sql = f"""
        SELECT table_name, rule_name, status, severity, metric_value, details
        FROM read_parquet('{_parquet_path('gold_data_quality_report')}')
        {where}
        ORDER BY CASE severity WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 ELSE 3 END, table_name, rule_name
    """
    return run_sql(sql)


def retrieve_policy_context(question: str, k: int = 3) -> list[dict[str, Any]]:
    retriever = PolicyRetriever()
    return [asdict(chunk) for chunk in retriever.retrieve(question, k=k)]


def answer_question_with_tools(question: str) -> dict[str, Any]:
    """Simple keyword router for the demo agent.

    The original version used an if/elif chain, so many questions triggered only one
    tool and the offline response felt repetitive. This router can call multiple
    tools for one question, which makes incident/anomaly/data-quality questions
    much more realistic in the free Streamlit deployment.
    """
    q = question.lower()
    tools_used: list[str] = []
    evidence: dict[str, Any] = {}

    asks_approval = any(x in q for x in ["approval", "approve", "drop", "funnel", "rejected", "anomaly"])
    asks_quality = any(x in q for x in ["quality", "dq", "null", "duplicate", "schema", "incident", "anomaly", "failure"])
    asks_delinquency = any(x in q for x in ["dpd", "delinquency", "default", "overdue", "repayment", "30+"])
    asks_portfolio = any(x in q for x in ["portfolio", "summary", "overall", "risk", "health", "citation"])

    if asks_approval:
        tools_used.append("approval_drop_analysis")
        evidence["approval_drop_analysis"] = approval_drop_analysis()

    # Run DQ before delinquency for questions like "data quality issues in repayment data".
    if asks_quality:
        tools_used.append("data_quality_diagnostics")
        evidence["data_quality_diagnostics"] = data_quality_diagnostics()

    if asks_delinquency:
        tools_used.append("delinquency_analysis")
        evidence["delinquency_analysis"] = delinquency_analysis()

    if asks_portfolio or not evidence:
        tools_used.append("portfolio_summary")
        evidence["portfolio_summary"] = portfolio_summary()

    tools_used.append("retrieve_policy_context")
    evidence["policy_context"] = retrieve_policy_context(question)
    return {"tools_used": tools_used, "evidence": evidence}
