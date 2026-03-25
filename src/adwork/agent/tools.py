# src/adwork/agent/tools.py

"""
Agent Tools
===========
Data retrieval and action functions called by the LangGraph nodes.
These are deterministic — the graph decides WHEN to call them,
the LLM decides WHAT TO DO with the results.
"""

import pandas as pd
from datetime import date, timedelta
from loguru import logger

from adwork.db.connection import get_db
from adwork.db.queries import (
    get_date_range,
    get_kpis,
    get_all_campaigns,
    get_campaign_summary,
    get_daily_metrics_for_campaign,
    get_stored_forecasts,
    get_trends,
    store_recommendations,
    clear_recommendations,
)


def gather_campaign_data() -> dict:
    """
    Gather all campaign data needed for the agent's analysis.
    Returns a structured dict summarizing all campaigns.
    """
    date_range = get_date_range()
    if not date_range:
        return {"error": "No data available"}

    start, end = date_range
    campaigns_df = get_all_campaigns()
    kpis = get_kpis(start, end)
    summary = get_campaign_summary(start, end)

    # Per-campaign detailed metrics (last 30 days + trend)
    campaign_details = []
    for _, camp in campaigns_df.iterrows():
        cid = camp["campaign_id"]
        daily = get_daily_metrics_for_campaign(cid)

        if daily.empty:
            continue

        daily = daily.sort_values("date")
        recent_14 = daily.tail(14)
        prior_14 = daily.iloc[-28:-14] if len(daily) >= 28 else pd.DataFrame()

        r_spend = recent_14["spend"].sum()
        r_rev = recent_14["revenue"].sum()
        r_conv = recent_14["conversions"].sum()
        r_clicks = recent_14["clicks"].sum()
        r_roas = r_rev / r_spend if r_spend > 0 else 0

        trend = "stable"
        trend_pct = 0.0
        if not prior_14.empty:
            p_spend = prior_14["spend"].sum()
            p_rev = prior_14["revenue"].sum()
            p_roas = p_rev / p_spend if p_spend > 0 else 0
            if p_roas > 0:
                trend_pct = ((r_roas - p_roas) / p_roas) * 100
                if trend_pct > 5:
                    trend = "improving"
                elif trend_pct < -5:
                    trend = "declining"

        campaign_details.append({
            "campaign_id": cid,
            "campaign_name": camp["campaign_name"],
            "platform": camp["platform"],
            "spend_14d": round(r_spend, 2),
            "revenue_14d": round(r_rev, 2),
            "conversions_14d": int(r_conv),
            "clicks_14d": int(r_clicks),
            "roas_14d": round(r_roas, 2),
            "trend": trend,
            "trend_pct": round(trend_pct, 1),
        })

    return {
        "date_range": {"start": str(start), "end": str(end)},
        "kpis": kpis,
        "campaigns": campaign_details,
        "n_campaigns": len(campaign_details),
    }


def gather_forecast_data() -> dict:
    """Gather stored forecasts and trends for the agent."""
    forecasts = get_stored_forecasts()
    trends = get_trends()

    forecast_summary = {}
    if not forecasts.empty:
        for cid in forecasts["campaign_id"].unique():
            camp_fc = forecasts[forecasts["campaign_id"] == cid]
            for metric in camp_fc["metric"].unique():
                metric_fc = camp_fc[camp_fc["metric"] == metric]
                forecast_summary[f"{cid}_{metric}"] = {
                    "campaign_id": cid,
                    "metric": metric,
                    "avg_predicted": round(metric_fc["predicted_value"].mean(), 1),
                    "forecast_days": len(metric_fc),
                }

    trend_summary = []
    if not trends.empty:
        for kw in trends["keyword"].unique():
            kw_data = trends[trends["keyword"] == kw].sort_values("date")
            if len(kw_data) >= 14:
                recent = kw_data.tail(7)["interest"].mean()
                prior = kw_data.iloc[-14:-7]["interest"].mean()
                change = ((recent - prior) / (prior + 1)) * 100
                trend_summary.append({
                    "keyword": kw,
                    "recent_interest": round(recent, 1),
                    "change_pct": round(change, 1),
                })

    return {
        "forecasts": forecast_summary,
        "n_forecasts": len(forecast_summary),
        "trends": trend_summary,
    }


def run_bid_optimization() -> dict:
    """Execute the Thompson Sampling bid recommender."""
    from adwork.optimization.bid_recommender import BidRecommender

    campaigns = get_all_campaigns()
    conn = get_db()
    all_metrics = conn.execute(
        "SELECT * FROM daily_metrics ORDER BY campaign_id, date"
    ).df()
    forecasts = get_stored_forecasts()

    recommender = BidRecommender()
    recs = recommender.generate_recommendations(campaigns, all_metrics, forecasts)

    return {
        "bid_recommendations": [r.model_dump() for r in recs],
        "n_recommendations": len(recs),
    }


def run_budget_optimization() -> dict:
    """Execute the budget allocator."""
    from adwork.optimization.budget_allocator import BudgetAllocator

    date_range = get_date_range()
    if not date_range:
        return {"shifts": [], "allocation": {}}

    summary = get_campaign_summary(date_range[0], date_range[1])
    allocator = BudgetAllocator()
    result = allocator.allocate(summary)

    return {
        "current_allocation": result["current_allocation"],
        "recommended_allocation": result["recommended_allocation"],
        "shifts": [s.model_dump() for s in result["shifts"]],
        "n_shifts": len(result["shifts"]),
    }


def format_campaigns_for_llm(campaign_data: dict) -> str:
    """Create a concise text summary of campaigns for LLM prompts."""
    lines = [
        f"Portfolio: {campaign_data['n_campaigns']} campaigns | "
        f"Total Spend: ${campaign_data['kpis']['total_spend']:,.0f} | "
        f"ROAS: {campaign_data['kpis']['overall_roas']:.1f}x | "
        f"Conversions: {campaign_data['kpis']['total_conversions']:,}",
        "",
        "Campaigns (last 14 days):",
    ]

    for c in campaign_data["campaigns"]:
        emoji = "📈" if c["trend"] == "improving" else "📉" if c["trend"] == "declining" else "➡️"
        lines.append(
            f"  {emoji} {c['campaign_name']} ({c['platform']}): "
            f"ROAS={c['roas_14d']:.1f}x, Spend=${c['spend_14d']:,.0f}, "
            f"Conv={c['conversions_14d']}, Trend={c['trend']} ({c['trend_pct']:+.1f}%)"
        )

    return "\n".join(lines)