# tests/test_agent.py

"""Test the LangGraph agent pipeline."""

import os
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    """
    Give each test a truly fresh database.
    
    The key fix: we must update the settings reference inside
    connection.py, not just adwork.config. Otherwise get_db()
    reads the stale path from its module-level import.
    """
    import adwork.config
    import adwork.db.connection as conn_module

    # 1. Close any existing connection
    conn_module.close_db()

    # 2. Build new settings pointing at a fresh tmp file
    db_path = str(tmp_path / "test_agent.duckdb")
    new_settings = adwork.config.Settings(
        duckdb_path=db_path,
        groq_api_key=os.environ.get("GROQ_API_KEY", "test"),
        llm_provider="groq",
    )

    # 3. Patch BOTH the config module AND the connection module's reference
    adwork.config.settings = new_settings
    conn_module.settings = new_settings       # ← this is the critical line

    # 4. Seed minimal data
    db = conn_module.get_db()

    db.execute("""
        INSERT INTO campaigns (campaign_id, campaign_name, platform, status, daily_budget)
        VALUES ('test_001', 'Test Google Campaign', 'google', 'active', 100.0)
    """)

    rng = np.random.default_rng(42)
    for i in range(60):
        d = date(2025, 4, 1) + timedelta(days=i)
        impr = int(rng.integers(3000, 8000))
        clicks = int(rng.integers(100, 400))
        conv = int(rng.integers(3, 20))
        spend = round(float(rng.uniform(80, 300)), 2)
        rev = round(float(rng.uniform(150, 900)), 2)
        ctr = round(clicks / impr, 6) if impr > 0 else 0
        cpc = round(spend / clicks, 4) if clicks > 0 else 0
        roas = round(rev / spend, 4) if spend > 0 else 0

        db.execute("""
            INSERT INTO daily_metrics
            (campaign_id, date, impressions, clicks, conversions, spend, revenue, ctr, cpc, roas)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ['test_001', d, impr, clicks, conv, spend, rev, ctr, cpc, roas])

    yield

    conn_module.close_db()


def test_tools_gather_data():
    from adwork.agent.tools import gather_campaign_data

    data = gather_campaign_data()

    assert "campaigns" in data
    assert data["n_campaigns"] >= 1
    assert "kpis" in data
    assert data["kpis"]["total_spend"] > 0


def test_tools_format_for_llm():
    from adwork.agent.tools import format_campaigns_for_llm, gather_campaign_data

    data = gather_campaign_data()
    text = format_campaigns_for_llm(data)

    assert "Test Google Campaign" in text
    assert "ROAS" in text
    assert len(text) > 50


def test_tools_bid_optimization():
    from adwork.agent.tools import run_bid_optimization

    result = run_bid_optimization()

    assert "bid_recommendations" in result
    assert isinstance(result["bid_recommendations"], list)


def test_tools_budget_optimization():
    from adwork.agent.tools import run_budget_optimization

    result = run_budget_optimization()

    assert "shifts" in result
    assert isinstance(result["shifts"], list)


def test_graph_builds():
    from adwork.agent.graph import build_optimization_graph

    graph = build_optimization_graph()
    assert graph is not None


def test_graph_runs_without_llm():
    """
    The graph should handle LLM failures gracefully and still produce output.
    """
    from adwork.agent.graph import build_optimization_graph
    from adwork.agent.llm_client import reset_client

    original_key = os.environ.get("GROQ_API_KEY", "")
    os.environ["GROQ_API_KEY"] = "invalid_key_for_testing"
    reset_client()

    # ── Create a mock LLM client that always raises ──
    failing_client = MagicMock()
    failing_client.complete.side_effect = Exception("LLM unavailable")
    failing_client.complete_json.side_effect = Exception("LLM unavailable")
    failing_client.complete_pydantic.side_effect = Exception("LLM unavailable")
    failing_client.provider_name = "mock"
    failing_client.model_name = "mock-fail"

    # ── Patch everywhere get_llm_client is looked up ──
    with patch("adwork.agent.graph.get_llm_client", return_value=failing_client), \
         patch("adwork.agent.llm_client.get_llm_client", return_value=failing_client):

        graph = build_optimization_graph()

        result = graph.invoke({
            "agent_log": [],
            "errors": [],
            "campaign_data": {},
            "forecast_data": {},
            "performance_analysis": {},
            "forecast_insights": {},
            "portfolio_health": "normal",
            "bid_recommendations": [],
            "budget_shifts": [],
            "final_recommendations": [],
            "daily_summary": "",
            "run_timestamp": "",
            "llm_provider": "",
        })

        # Graph still completes
        assert "agent_log" in result
        assert len(result["agent_log"]) > 0
        # LLM failures were recorded
        assert len(result["errors"]) > 0