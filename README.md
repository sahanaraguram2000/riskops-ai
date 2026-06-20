# RiskOps AI: Agentic RAG Platform for Credit Risk Analytics & Data Quality

A **zero-cost, local-first GenAI/Data Engineering project** designed for a Data/Analytics Engineer profile. It simulates an NBFC lending analytics platform where business/risk users can ask natural-language questions about portfolio risk, approval drops, delinquency, and data-quality failures.

This repo is intentionally built without requiring Snowflake, Databricks, Azure, Power BI, OpenAI, or any corporate account. It runs locally with free/open-source tools and has optional adapters for Ollama, Gemini, MLflow, PySpark, and Chroma.

## What this demonstrates

- **Agentic RAG** over credit policy documents
- **Tool-calling agent** for SQL analytics, policy retrieval, and data-quality diagnostics
- **Lakehouse-style Bronze/Silver/Gold architecture** using local CSV/Parquet
- **Credit risk analytics**: approval rate, DPD buckets, delinquency, vintage analysis, risk segments
- **Data quality monitoring**: null spikes, duplicate keys, schema drift, FK failures, outliers
- **LLMOps hooks**: local MLflow tracing/evaluation script
- **Production-style API and UI**: FastAPI + Streamlit
- **Free LLM options**: offline deterministic mode, Ollama local model, or Gemini free-tier key

## Architecture

```text
Synthetic NBFC data
    ↓
Bronze CSV files
    ↓
DuckDB/PySpark-compatible transformations
    ↓
Silver Parquet tables
    ↓
Gold risk marts + DQ reports
    ↓
Policy RAG index
    ↓
RiskOps Agent
    ├── SQL analytics tool
    ├── Policy retrieval tool
    ├── Data-quality diagnostics tool
    └── Incident-summary tool
    ↓
FastAPI / Streamlit / MLflow
```

## Cost

Default mode costs **₹0**.

| Requirement | Default free option |
|---|---|
| Warehouse | DuckDB |
| Lakehouse tables | Parquet, optional Delta Lake |
| Vector search | TF-IDF local retriever, optional Chroma |
| LLM | Offline deterministic mode, optional Ollama |
| Dashboard | Streamlit |
| API | FastAPI |
| Observability | MLflow local |
| Deployment | Local Docker / Streamlit Community Cloud |

## Quick start

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install base dependencies
pip install -r requirements.txt

# 3. Create data, marts, policy index, and sample evaluation files
python scripts/bootstrap.py

# 4. Run Streamlit app
streamlit run app/streamlit_app.py
```

Open the Streamlit URL and ask:

```text
Why did approval rate drop this month?
Which risk segment has the highest 30+ DPD?
Show data quality issues in repayment data.
Summarize portfolio risk with policy citations.
Generate an incident summary for the approval anomaly.
```

## Run FastAPI

```bash
uvicorn app.api:app --reload --port 8000
```

Then test:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Why did approval rate drop this month?"}'
```

## Optional: use Ollama local LLM

Install Ollama and pull a small free model:

```bash
ollama pull llama3.2:3b
```

Then create `.env` from `.env.example` and set:

```bash
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b
```

Run the app again. If Ollama is not running, the project falls back to offline mode.

## Optional: use Gemini API free tier

Create `.env`:

```bash
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-1.5-flash
```

Use only synthetic data with external APIs.

## Optional: run MLflow evaluation

```bash
mlflow ui --port 5000
python scripts/evaluate.py
```

The evaluation script logs response length, answerability, citation presence, latency, and simple retrieval relevance.


## Deploy on Streamlit Community Cloud

This repo is ready for a free Streamlit Community Cloud deployment. The app automatically creates demo data and the policy index on first launch if generated marts are not committed to Git.

1. Push this folder to a GitHub repository.
2. Go to Streamlit Community Cloud and create a new app from that repo.
3. Set the main file path to:

```text
app/streamlit_app.py
```

4. Deploy.

For zero-cost deployment, keep the default offline LLM mode. For Gemini, add secrets/environment variables in Streamlit Cloud settings:

```toml
LLM_PROVIDER = "gemini"
GEMINI_API_KEY = "your_key_here"
GEMINI_MODEL = "gemini-1.5-flash"
```

Do not use Ollama on Streamlit Community Cloud; Ollama is intended for local/Docker use because it needs a running local model server.

## Optional: Docker

```bash
docker compose up --build
```

Services:

- Streamlit app: http://localhost:8501
- FastAPI: http://localhost:8000
- MLflow UI: http://localhost:5000

## Repo layout

```text
riskops-ai/
├── app/
│   ├── api.py
│   └── streamlit_app.py
├── data/
│   ├── policies/
│   └── contracts/
├── docs/
├── scripts/
├── src/riskops_ai/
│   ├── agent.py
│   ├── config.py
│   ├── llm.py
│   ├── tools.py
│   ├── pipelines/
│   └── rag/
├── tests/
├── docker-compose.yml
├── Dockerfile
├── Makefile
└── requirements.txt
```

## Resume bullets

Use these only after you run/customize the project:

- Built a **production-style Agentic RAG RiskOps platform** using **Python, DuckDB, Parquet, LangGraph-compatible agent patterns, FastAPI, Streamlit, and MLflow**, enabling natural-language investigation of credit risk, delinquency trends, approval anomalies, and data-quality failures.
- Designed a **Bronze/Silver/Gold lakehouse-style pipeline** for synthetic NBFC loan, repayment, bureau, and collections data, creating curated marts for DPD analysis, approval funnels, vintage curves, and portfolio risk segmentation.
- Implemented tool-calling workflows for **SQL analytics, policy retrieval, data-quality diagnostics, and incident-summary generation**, with offline/Ollama/Gemini LLM provider support.
- Added **LLMOps-style evaluation and observability hooks** including prompt logging, retrieval context inspection, latency tracking, citation checks, and MLflow experiment logging.

## What to improve before interviews

1. Add your own architecture diagram screenshot.
2. Record a 90-second Loom demo.
3. Add screenshots from Streamlit and MLflow into `docs/screenshots/`.
4. Customize the synthetic data story to match a fintech/NBFC company.
5. Push to GitHub with a clean README and pinned demo questions.

## Disclaimer

This is a portfolio project using synthetic data. It is not a real credit decisioning system and should not be used for lending decisions.
