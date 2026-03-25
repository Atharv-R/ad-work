# src/adwork/agent/graph.py

"""
LangGraph Optimization Agent
=============================
A stateful agent pipeline that:
1. Gathers campaign data from the database
2. Uses LLM to analyze performance and flag issues
3. Routes based on portfolio health (conditional edge)
4. Runs Thompson Sampling optimization
5. Uses LLM to synthesize everything into explainable recommendations
6. Stores results

Graph structure:

    gather_data → analyze_performance → route_by_health
        ├── "critical" → run_optimization → synthesize_critical → store_results → END
        └── "normal"   → check_forecasts → run_optimization → synthesize → store_results → END

The critical path skips forecast analysis (act now, analyze later).
The normal path does full analysis including demand forecasts.
"""

from __future__ import annotations

import json
from typing import TypedDict, Annotated
from datetime import datetime
import operator

from langgraph.graph import StateGraph, END
from loguru import logger

from adwork.agent.llm_client import get_llm_client
from adwork.agent.prompts import (
    SYSTEM_PROMPT,
    PERFORMANCE_ANALYSIS_PROMPT,
    FORECAST_ANALYSIS_PROMPT,
    SYNTHESIS_PROMPT,
    SYNTHESIS_CRITICAL_PROMPT,
)
from adwork.agent.tools import (
    gather_campaign_data,
    gather_forecast_data,
    run_bid_optimization,
    run_budget_optimization,
    format_campaigns_for_llm,
)
from adwork.db.queries import (          
    clear_recommendations,
    store_recommendations,
)

# ─── State Definition ────────────────────────────────

class AgentState(TypedDict):
    """
    Typed state flowing through the LangGraph pipeline.

    Fields with Annotated[list, operator.add] are ACCUMULATED —
    each node appends to them. All other fields are REPLACED.
    """
    # Accumulated across nodes
    agent_log: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]

    # Data layer (set by gather_data)
    campaign_data: dict
    forecast_data: dict

    # LLM analysis layer
    performance_analysis: dict
    forecast_insights: dict
    portfolio_health: str

    # Optimization layer
    bid_recommendations: list[dict]
    budget_shifts: list[dict]

    # Output layer
    final_recommendations: list[dict]
    daily_summary: str

    # Metadata
    run_timestamp: str
    llm_provider: str


# ─── Node Functions ───────────────────────────────────

def gather_data_node(state: AgentState) -> dict:
    """Node 1: Gather all campaign data from the database."""
    logger.info("Agent: Gathering campaign data...")

    campaign_data = gather_campaign_data()
    n = campaign_data.get("n_campaigns", 0)

    return {
        "campaign_data": campaign_data,
        "agent_log": [f"📊 Gathered data for {n} campaigns"],
        "run_timestamp": datetime.now().isoformat(),
    }


def analyze_performance_node(state: AgentState) -> dict:
    """Node 2: LLM analyzes campaign performance and rates portfolio health."""
    logger.info("Agent: Analyzing performance with LLM...")

    campaign_data = state.get("campaign_data", {})
    if not campaign_data or campaign_data.get("error"):
        return {
            "performance_analysis": {},
            "portfolio_health": "normal",
            "errors": ["No campaign data available for analysis"],
            "agent_log": ["⚠️ No data — skipping performance analysis"],
        }

    campaign_summary = format_campaigns_for_llm(campaign_data)

    try:
        llm = get_llm_client()
        prompt = PERFORMANCE_ANALYSIS_PROMPT.format(campaign_summary=campaign_summary)

        response = llm.complete_json(messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])

        health = response.get("portfolio_health", "normal")
        n_under = len(response.get("underperformers", []))

        return {
            "performance_analysis": response,
            "portfolio_health": health,
            "llm_provider": llm.provider_name,
            "agent_log": [
                f"🔍 Performance analyzed: portfolio is {health.upper()}, "
                f"{n_under} underperformers flagged"
            ],
        }

    except Exception as e:
        logger.error(f"LLM analysis failed: {e}")
        return {
            "performance_analysis": {},
            "portfolio_health": "normal",
            "errors": [f"LLM performance analysis failed: {str(e)}"],
            "agent_log": ["⚠️ LLM analysis failed — using default routing"],
        }


def check_forecasts_node(state: AgentState) -> dict:
    """Node 3 (normal path): LLM reviews demand forecasts."""
    logger.info("Agent: Checking forecasts with LLM...")

    forecast_data = gather_forecast_data()
    campaign_data = state.get("campaign_data", {})
    campaign_summary = format_campaigns_for_llm(campaign_data) if campaign_data else "No data"

    # Build forecast summary text
    fc_lines = []
    for fc in forecast_data.get("forecasts", {}).values():
        fc_lines.append(f"  {fc['campaign_id']}/{fc['metric']}: avg predicted={fc['avg_predicted']}")
    for t in forecast_data.get("trends", []):
        fc_lines.append(f"  Trend '{t['keyword']}': interest={t['recent_interest']}, change={t['change_pct']:+.1f}%")

    forecast_text = "\n".join(fc_lines) if fc_lines else "No forecast data available."

    try:
        llm = get_llm_client()
        prompt = FORECAST_ANALYSIS_PROMPT.format(
            campaign_summary=campaign_summary,
            forecast_summary=forecast_text,
        )

        response = llm.complete_json(messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])

        n_opps = len(response.get("opportunities", []))

        return {
            "forecast_data": forecast_data,
            "forecast_insights": response,
            "agent_log": [f"📈 Forecast reviewed: {n_opps} opportunities identified"],
        }

    except Exception as e:
        logger.error(f"Forecast analysis failed: {e}")
        return {
            "forecast_data": forecast_data,
            "forecast_insights": {},
            "errors": [f"Forecast analysis failed: {str(e)}"],
            "agent_log": ["⚠️ Forecast analysis failed — continuing without"],
        }


def run_optimization_node(state: AgentState) -> dict:
    """Node 4: Run Thompson Sampling bid optimization + budget allocation."""
    logger.info("Agent: Running optimization...")

    try:
        bid_result = run_bid_optimization()
        budget_result = run_budget_optimization()

        n_bids = bid_result.get("n_recommendations", 0)
        n_shifts = budget_result.get("n_shifts", 0)

        return {
            "bid_recommendations": bid_result.get("bid_recommendations", []),
            "budget_shifts": budget_result.get("shifts", []),
            "agent_log": [
                f"🎯 Optimization complete: {n_bids} bid recs + {n_shifts} budget shifts"
            ],
        }

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return {
            "bid_recommendations": [],
            "budget_shifts": [],
            "errors": [f"Optimization failed: {str(e)}"],
            "agent_log": ["⚠️ Optimization failed"],
        }


def synthesize_node(state: AgentState) -> dict:
    """Node 5 (normal path): LLM synthesizes everything into final recommendations."""
    logger.info("Agent: Synthesizing recommendations...")

    perf = state.get("performance_analysis", {})
    forecast = state.get("forecast_insights", {})
    bids = state.get("bid_recommendations", [])
    shifts = state.get("budget_shifts", [])

    # Summarize bid recs for prompt
    bid_lines = []
    for b in bids[:10]:
        bid_lines.append(
            f"  {b['campaign_name']}: {b['recommended_action']} "
            f"{abs(b['recommended_change_pct']):.0f}% "
            f"(ROAS {b['current_roas']:.1f}x → {b['expected_roas_after']:.1f}x, "
            f"confidence: {b['confidence']})"
        )
    bid_text = "\n".join(bid_lines) if bid_lines else "No bid recommendations."

    shift_lines = []
    for s in shifts:
        shift_lines.append(
            f"  Move ${s['amount']:.0f}/day: {s['from_campaign_name']} → {s['to_campaign_name']}"
        )
    shift_text = "\n".join(shift_lines) if shift_lines else "No budget shifts."

    try:
        llm = get_llm_client()
        prompt = SYNTHESIS_PROMPT.format(
            performance_analysis=json.dumps(perf, indent=2)[:1500],
            forecast_insights=json.dumps(forecast, indent=2)[:1000],
            bid_recommendations=bid_text,
            budget_shifts=shift_text,
        )

        response = llm.complete_json(messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])

        recs = response.get("recommendations", [])
        summary = response.get("daily_summary", "No summary generated.")

        return {
            "final_recommendations": recs,
            "daily_summary": summary,
            "agent_log": [f"✅ Synthesized {len(recs)} final recommendations"],
        }

    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        # Fallback: use raw bid recommendations as final
        fallback_recs = [
            {
                "campaign_id": b["campaign_id"],
                "campaign_name": b["campaign_name"],
                "action_type": "bid_adjustment",
                "action_summary": f"{b['recommended_action']} bid {abs(b['recommended_change_pct']):.0f}%",
                "reasoning": b["reasoning"],
                "confidence": b["confidence"],
                "priority": i + 1,
            }
            for i, b in enumerate(bids[:8])
        ]
        return {
            "final_recommendations": fallback_recs,
            "daily_summary": "Agent synthesis failed — showing raw optimization results.",
            "errors": [f"Synthesis failed: {str(e)}"],
            "agent_log": ["⚠️ LLM synthesis failed — using fallback recommendations"],
        }


def synthesize_critical_node(state: AgentState) -> dict:
    """Node 5 (critical path): Urgent synthesis skipping forecast details."""
    logger.info("Agent: Generating URGENT recommendations...")

    perf = state.get("performance_analysis", {})
    bids = state.get("bid_recommendations", [])
    shifts = state.get("budget_shifts", [])

    bid_text = "\n".join(
        f"  {b['campaign_name']}: {b['recommended_action']} {abs(b['recommended_change_pct']):.0f}%"
        for b in bids[:10]
    ) or "No bid recommendations."

    shift_text = "\n".join(
        f"  Move ${s['amount']:.0f}/day: {s['from_campaign_name']} → {s['to_campaign_name']}"
        for s in shifts
    ) or "No budget shifts."

    try:
        llm = get_llm_client()
        prompt = SYNTHESIS_CRITICAL_PROMPT.format(
            performance_analysis=json.dumps(perf, indent=2)[:1500],
            bid_recommendations=bid_text,
            budget_shifts=shift_text,
        )

        response = llm.complete_json(messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])

        recs = response.get("recommendations", [])
        summary = response.get("daily_summary", "URGENT: See recommendations.")

        return {
            "final_recommendations": recs,
            "daily_summary": summary,
            "agent_log": [f"🚨 URGENT: Generated {len(recs)} critical recommendations"],
        }

    except Exception as e:
        logger.error(f"Critical synthesis failed: {e}")
        return {
            "final_recommendations": [],
            "daily_summary": "Critical synthesis failed. Review campaigns manually.",
            "errors": [f"Critical synthesis failed: {str(e)}"],
            "agent_log": ["⚠️ Critical synthesis failed"],
        }


def store_results_node(state: AgentState) -> dict:
    """Node 6: Store final recommendations in the database."""
    logger.info("Agent: Storing results...")

    recs = state.get("final_recommendations", [])
    summary = state.get("daily_summary", "")

    # Store recommendations in DuckDB
    clear_recommendations()

    recs_to_store = []
    for rec in recs:
        recs_to_store.append({
            "campaign_id": rec.get("campaign_id", ""),
            "action_type": rec.get("action_type", "bid_adjustment"),
            "action_detail": {"action_summary": rec.get("action_summary", "")},
            "reasoning": rec.get("reasoning", ""),
            "confidence": rec.get("confidence", "medium"),
        })

    n_stored = store_recommendations(recs_to_store) if recs_to_store else 0

    # Save full agent output to JSON
    from pathlib import Path
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    agent_output = {
        "run_timestamp": state.get("run_timestamp", ""),
        "llm_provider": state.get("llm_provider", "unknown"),
        "portfolio_health": state.get("portfolio_health", "normal"),
        "performance_analysis": state.get("performance_analysis", {}),
        "forecast_insights": state.get("forecast_insights", {}),
        "n_bid_recommendations": len(state.get("bid_recommendations", [])),
        "n_budget_shifts": len(state.get("budget_shifts", [])),
        "final_recommendations": recs,
        "daily_summary": summary,
        "agent_log": state.get("agent_log", []),
        "errors": state.get("errors", []),
    }

    with open(output_dir / "agent_output.json", "w") as f:
        json.dump(agent_output, f, indent=2, default=str)

    return {
        "agent_log": [f"💾 Stored {n_stored} recommendations + agent output saved"],
    }


# ─── Routing Function ────────────────────────────────

def route_by_health(state: AgentState) -> str:
    """
    Conditional edge: route based on portfolio health.

    Critical path skips forecast analysis — act now, analyze later.
    Normal path does full analysis including demand forecasts.
    """
    health = state.get("portfolio_health", "normal")
    logger.info(f"Agent: Routing by health = {health}")
    return health


# ─── Graph Builder ────────────────────────────────────

# Replace the graph builder with this corrected version:

def _route_after_optimization(state: AgentState) -> str:
    """Route after optimization to the correct synthesis node."""
    return state.get("portfolio_health", "normal")


def build_optimization_graph():
    """
    Build and compile the LangGraph optimization pipeline.

    Graph:
        gather_data → analyze_performance → route_by_health
            ├── "critical" → run_optimization_critical → synthesize_critical → store → END
            └── "normal"   → check_forecasts → run_optimization_normal → synthesize → store → END
    """
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("gather_data", gather_data_node)
    graph.add_node("analyze_performance", analyze_performance_node)
    graph.add_node("check_forecasts", check_forecasts_node)
    graph.add_node("run_optimization_normal", run_optimization_node)
    graph.add_node("run_optimization_critical", run_optimization_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("synthesize_critical", synthesize_critical_node)
    graph.add_node("store_results", store_results_node)

    # Entry
    graph.set_entry_point("gather_data")

    # Flow
    graph.add_edge("gather_data", "analyze_performance")

    # Conditional: health-based routing
    graph.add_conditional_edges(
        "analyze_performance",
        route_by_health,
        {
            "critical": "run_optimization_critical",
            "normal": "check_forecasts",
        },
    )

    # Normal path
    graph.add_edge("check_forecasts", "run_optimization_normal")
    graph.add_edge("run_optimization_normal", "synthesize")
    graph.add_edge("synthesize", "store_results")

    # Critical path
    graph.add_edge("run_optimization_critical", "synthesize_critical")
    graph.add_edge("synthesize_critical", "store_results")

    # End
    graph.add_edge("store_results", END)

    return graph.compile()