<!-- README.md -->

# Ad-Work

**Agentic ad optimization pipeline** that combines time-series forecasting, 
contextual bandit optimization, and LLM reasoning to generate explainable 
advertising recommendations.

> Upload your campaign data → Get demand forecasts → Receive bid and budget 
> recommendations with plain-English reasoning attached to every decision.

**[Live Demo](https://ad-work-lrikjcynfa8feptsuauuoz.streamlit.app)**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![CI](https://github.com/{{YOUR_GITHUB_USERNAME}}/ad-work/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/License-MIT-green)

---
---

## The Problem

Digital advertisers manage campaigns across Google, Meta, and Amazon — each with different metrics, bid strategies, and budget constraints. Decisions are manual, reactive, and rarely explained. Ad-Work automates the full loop: monitor → forecast → optimize → explain.

## Architecture
| Forecaster | 
| Prophet + statsmodels |
|---|
| CTR Model |
| LightGBM on Criteo data |
|---|
| Bandit |
| Thompson Sampling |
|---|
| Budget Allocation |
| Cross-campaign reallocation |
|---|
| LLM |
| LangGraph Agent |
|---|
|DuckDB|
| campaigns · daily_metrics · forecasts · recommendations |
|---|
| Streamlit Dahsboard|
|Overview · Forecasts · Recommendations · Competitors|


## ML Components

### 1. Demand Forecasting
- **Prophet** (primary) with Google Trends as external regressor
- **Exponential Smoothing** fallback for environments where Prophet won't install
- Backtest validation with train/test split — median MAPE ~12% on clicks
- 14-day forecast horizon with 80% confidence intervals

### 2. CTR Prediction
- **LightGBM** trained on the [Criteo Click Logs](https://huggingface.co/datasets/criteo/criteo-click-logs) dataset (1M+ samples)
- 13 numerical + 26 categorical features, high-cardinality encoding
- **0.79 AUC-ROC**, calibration error < 0.02
- Full preprocessing pipeline with `CriteoPreprocessor` (fit/transform pattern)

### 3. Bid Optimization
- **Thompson Sampling** (Beta and Normal arms) for bid multiplier selection
- Bid response model with diminishing returns (impression ~ bid^0.5)
- Converges ~40% faster than epsilon-greedy (shown via regret curves)
- Per-campaign recommendations with confidence levels

### 4. Budget Allocation
- Cross-campaign reallocation using posterior ROAS beliefs
- Shifts budget from underperformers to high-ROAS campaigns
- Integrates forecast signals for forward-looking allocation

### 5. Competitor Intelligence
- TF-IDF vectorization + K-Means clustering of competitor ad copy
- PCA projection for visual cluster exploration
- Auto-labelled strategy clusters (Price, Features, Lifestyle, Urgency, Comparison)
- Extensible: upload your own competitor data via CSV

### 6. LLM Agent (LangGraph)
- **StateGraph** with conditional routing — critical portfolios skip forecasts and get urgent recommendations
- 3 LLM calls per run: performance analysis → forecast insights → synthesis
- LLM-agnostic: Groq (Llama 3.3 70B, free) ↔ OpenAI (GPT-4o-mini) via env var
- Graceful degradation: falls back to template recommendations if LLM fails

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Database | DuckDB | Embedded analytical DB, zero config, SQL interface |
| Forecasting | Prophet + statsmodels | Industry standard + lightweight fallback |
| CTR Model | LightGBM | Fast, handles categoricals natively |
| Optimization | Thompson Sampling (NumPy) | No library needed, provably optimal exploration |
| Agent | LangGraph | Stateful multi-step agent with conditional routing |
| LLM | Groq / OpenAI | Abstraction layer, swap with one env var |
| Dashboard | Streamlit + Plotly | Python-native, interactive, free hosting |
| CI | GitHub Actions + ruff | Lint + test on every push |

## Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
git clone https://github.com/{{YOUR_GITHUB_USERNAME}}/ad-work.git
cd ad-work
uv sync
```

### Create a .env file:
```python
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_key_here
OPENAI_API_KEY=
DUCKDB_PATH=data/adwork.duckdb
LOG_LEVEL=INFO
```

### Run:
```bash 
# Seed demo data (10 campaigns × 90 days)
uv run python scripts/seed_demo.py

# Generate forecasts
uv run python scripts/run_forecasts.py

# Run bid + budget optimization
uv run python scripts/run_optimization.py

# Run the LLM agent
uv run python scripts/run_agent.py

### Launch dashboard
uv run streamlit run dashboard/app.py
```

### Train CTR Model (optional):
```bash
# Quick — synthetic data
uv run python scripts/train_ctr.py --synthetic

### Full — real Criteo data (downloads ~4GB)
uv run python scripts/download_criteo.py
uv run python scripts/train_ctr.py --sample-size 1000000
```

### Tests:
```bash
uv run pytest tests/ -v
```

## Upload your own data
Ad-Work accepts CSV exports from Google Ads, Meta Ads, and Amazon Ads. The system auto-detects the platform format based on column names.

- Go to 📤 Upload Data in the dashboard
- Upload your CSV (exported from any ad platform)
- The system maps columns to the internal schema automatically
- All ML pipelines run on your data
