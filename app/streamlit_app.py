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
from riskops_ai.tools import portfolio_summary, data_quality_diagnostics, approval_drop_analysis, delinquency_analysis

st.set_page_config(page_title="RiskOps AI", layout="wide")
st.title("RiskOps AI: Agentic RAG for Credit Risk + Data Quality")
st.caption("Zero-cost local demo using synthetic NBFC data. Default LLM mode is offline and deterministic.")


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
    st.caption("Changing the sample now updates the question box, so each run uses the selected query.")

agent = RiskOpsAgent()
question = st.text_area("Ask the RiskOps agent", key="question", height=90)

if st.button("Run agent", type="primary"):
    response = agent.ask(question)
    st.subheader("Answer")
    st.write(response.answer)
    st.caption(f"Tools used: {', '.join(response.tools_used)} | Latency: {response.latency_ms} ms")

    with st.expander("Evidence JSON"):
        st.json(response.evidence)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Latest approval drop drivers")
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
    st.subheader("Data-quality failures")
    dq_df = pd.DataFrame(data_quality_diagnostics())
    st.dataframe(dq_df, use_container_width=True)
    if not dq_df.empty:
        fig = px.histogram(dq_df, x="severity", title="Failed DQ checks by severity")
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Portfolio summary")
portfolio_df = pd.DataFrame(portfolio_summary(30))
st.dataframe(portfolio_df, use_container_width=True)

st.subheader("Delinquency hotspots")
dpd_df = pd.DataFrame(delinquency_analysis())
st.dataframe(dpd_df, use_container_width=True)
