# src/adwork/pipeline/daily_loop.py

"""
Daily Optimization Loop
=======================
Clean entry point that builds and runs the LangGraph agent.
"""

from datetime import datetime
from loguru import logger


def run_daily_optimization() -> dict:
    """
    Execute the full daily optimization agent loop.

    Returns the final agent state with all results.
    """
    from adwork.agent.graph import build_optimization_graph

    logger.info("Starting daily optimization loop...")

    graph = build_optimization_graph()

    initial_state = {
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
        "run_timestamp": datetime.now().isoformat(),
        "llm_provider": "",
    }

    result = graph.invoke(initial_state)

    n_recs = len(result.get("final_recommendations", []))
    health = result.get("portfolio_health", "unknown")
    n_errors = len(result.get("errors", []))

    logger.info(
        f"Daily loop complete: {n_recs} recommendations, "
        f"health={health}, {n_errors} errors"
    )

    return dict(result)