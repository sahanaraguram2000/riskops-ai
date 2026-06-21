from __future__ import annotations

import sys
from pathlib import Path

# Allows `streamlit run app/streamlit_app.py` from repo root without installation.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import plotly.express as px
import streamlit as st

from riskops_ai.agent import RiskOpsAgent
from riskops_ai.tools import (
    approval_drop_analysis,
    data_quality_diagnostics,
    delinquency_analysis,
    portfolio_summary,
)

st.set_page_config(page_title="RiskOps AI", layout="wide")
st.title("RiskOps AI: Agentic RAG for Credit Risk + Data Quality")
# st.caption("Zero-cost local/web demo using synthetic NBFC data. Default LLM mode is offline and deterministic.")


def ensure_demo_artifacts() -> None:
    """Build demo data/index on first web launch if generated files were not committed."""
    from riskops_ai.config import GOLD_DIR, INDEX_DIR

    required_files = [
        GOLD_DIR / "gold_portfolio_summary.parquet",
        GOLD_DIR / "gold_approval_funnel.parquet",
        GOLD_DIR / "gold_data_quality_report.parquet",
        INDEX_DIR / "policy_tfidf_index.pkl",
    ]
    if all(path.exists() for path in required_files):
        return

    with st.spinner("Preparing synthetic demo data for first launch..."):
        from riskops_ai.pipelines.generate_synthetic_data import generate_all
        from riskops_ai.pipelines.build_marts_duckdb import build_all
        from riskops_ai.rag.policy_index import build_index

        generate_all()
        build_all()
        build_index()


ensure_demo_artifacts()

examples = [
    "Why did approval rate drop this month?",
    "Which segment has the highest 30+ DPD?",
    "Show data quality issues in repayment data.",
    "Summarize portfolio risk with policy citations.",
    "Generate an incident summary for the approval anomaly.",
]


def load_selected_question() -> None:
    st.session_state.question = st.session_state.selected_question


if "question" not in st.session_state:
    st.session_state.question = examples[0]

with st.sidebar:
    st.header("Demo questions")
    st.radio(
        "Pick one",
        examples,
        key="selected_question",
        on_change=load_selected_question,
    )
    st.caption("The selected sample question is used only in the Agent tab.")
    st.divider()
    st.markdown(
        "**App layout**\n\n"
        "- **Ask RiskOps Agent:** natural-language Q&A with tools and evidence.\n"
        "- **Portfolio Monitoring Dashboard:** fixed portfolio, DQ, approval, and delinquency views."
    )

agent_tab, dashboard_tab = st.tabs(["Ask RiskOps Agent", "Portfolio Monitoring Dashboard"])

with agent_tab:
    st.subheader("Ask the RiskOps Agent")
    st.write(
        "Ask a credit-risk, approval, delinquency, or data-quality question. "
        "Only the answer and its supporting evidence appear here; fixed monitoring charts are kept in the dashboard tab."
    )

    question = st.text_area("Question", key="question", height=90)

    if st.button("Run agent", type="primary"):
        agent = RiskOpsAgent()
        response = agent.ask(question)

        st.subheader("Answer")
        st.write(response.answer)
        st.caption(f"Tools used: {', '.join(response.tools_used)} | Latency: {response.latency_ms} ms")

        with st.expander("Evidence JSON"):
            st.json(response.evidence)

with dashboard_tab:
    st.subheader("Portfolio Monitoring Dashboard")
    st.write(
        "These are fixed monitoring views built from the Gold marts. "
        "They are intentionally separated from the Q&A flow so the agent answer is not mixed with unrelated charts."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Latest approval drop drivers")
        approval_df = pd.DataFrame(approval_drop_analysis())
        st.dataframe(approval_df, use_container_width=True)
        if not approval_df.empty:
            fig = px.bar(
                approval_df.head(8),
                x="approval_drop_pp",
                y="channel",
                color="employment_type",
                orientation="h",
                title="Approval drop by channel and employment type",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Data-quality failures")
        dq_df = pd.DataFrame(data_quality_diagnostics())
        st.dataframe(dq_df, use_container_width=True)
        if not dq_df.empty:
            fig = px.histogram(dq_df, x="severity", title="Failed DQ checks by severity")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Portfolio summary")
    portfolio_df = pd.DataFrame(portfolio_summary(30))
    st.dataframe(portfolio_df, use_container_width=True)

    st.markdown("### Delinquency hotspots")
    dpd_df = pd.DataFrame(delinquency_analysis())
    st.dataframe(dpd_df, use_container_width=True)
