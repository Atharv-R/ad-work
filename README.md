<!-- README.md -->

# Ad-Work

**Agentic ad optimization pipeline** that combines time-series forecasting, 
contextual bandit optimization, and LLM reasoning to generate explainable 
advertising recommendations.

> Upload your campaign data → Get demand forecasts → Receive bid and budget 
> recommendations with plain-English reasoning attached to every decision.

🔗 **[Live Demo](https://ad-work-lrikjcynfa8feptsuauuoz.streamlit.app/)** (app built, currently adding functionalities)

---

## Architecture

Campaign Data → Demand Forecasting (Prophet/TFT)
↓
Bid Optimization (Contextual Bandit)
↓
LLM Agent (Groq/OpenAI) synthesizes insights
↓
Dashboard with explainable recommendations

### ML Components
- **Demand Forecasting:** Prophet for time-series prediction with seasonality detection
- **CTR Prediction:** LightGBM trained on Criteo Click Logs (real ad interaction data)
- **Bid Optimization:** Thompson Sampling contextual bandit with exploration/exploitation balancing
- **NLP Intelligence:** Competitor ad monitoring via Meta Ad Library + sentence-transformers

### Engineering Decisions
- **LLM-agnostic design:** Abstraction layer swaps between Groq (free) and OpenAI (production) via environment variable
- **Embedded analytics DB:** DuckDB for fast analytical queries with zero infrastructure
- **Schema-first data:** Pydantic models validate all data flowing through the pipeline

---

## Quick Start

### Clone
git clone https://github.com/YOUR_USERNAME/ad-work.git
cd ad-work

### Setup
make setup

### Add your Groq API key to .env
### Get one free at https://console.groq.com

### Run
make run

## Project Status

Phase 0: Foundation & project structure
Phase 1: Data ingestion & sample campaigns
Phase 2: Demand forecasting (Prophet)
Phase 3: CTR model on Criteo data
Phase 4: Bandit optimization
Phase 5: LLM agent pipeline
Phase 6: Competitor intelligence
Phase 7: Polish & deploy

## Tech Stack

Component | Technology | Cost
-----------------------------
Language | Python 3.11+ | Free
Database | DuckDB | Free
Dashboard | Streamlit | Free
Forecasting | Prophet / Temporal Fusion Transformer | Free
Optimization | Thompson Sampling (custom) | Free
LLM | Groq (Llama 3.3 70B) / OpenAI (GPT-4o-mini) | Free
NLP | sentence-transformers | Free
Hosting | Streamlit Cloud | Free
CI/CD | GitHub Actions | Free

