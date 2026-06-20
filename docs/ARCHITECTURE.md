# Architecture

## Main components

1. **Synthetic Data Generator**
   - Creates realistic NBFC-like customer, loan, repayment, bureau, and collections data.
   - Injects controlled anomalies so the agent has meaningful incidents to explain.

2. **Bronze/Silver/Gold Pipeline**
   - Bronze: raw CSVs
   - Silver: typed, cleaned Parquet tables
   - Gold: risk marts and data-quality reports

3. **Policy RAG**
   - Chunks local policy markdown files.
   - Builds a local TF-IDF index that works without paid embeddings.
   - Can be replaced with Chroma/FAISS later.

4. **RiskOps Agent**
   - Routes user questions to tools.
   - Uses SQL analytics, RAG, DQ checks, and incident summarization.
   - Supports offline, Ollama, and Gemini providers.

5. **Serving Layer**
   - FastAPI for programmatic usage.
   - Streamlit for demos.

6. **LLMOps Hooks**
   - Evaluation script logs to MLflow when available.
   - Captures latency, retrieval quality, and citation presence.

## Why local-first?

A recruiter or interviewer should be able to clone the repo, run it, and see your work without needing corporate accounts or cloud credits.
