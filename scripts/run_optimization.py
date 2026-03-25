# scripts/run_optimization.py

"""
Run the bid optimization pipeline.

Generates bid recommendations + budget allocation + regret simulation.

Usage:
    uv run python scripts/run_optimization.py

Windows:
    .venv\\Scripts\\python scripts\\run_optimization.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loguru import logger


def main():
    from adwork.db.queries import (
        get_all_campaigns,
        get_campaign_summary,
        get_date_range,
        get_stored_forecasts,
        store_recommendations,
        clear_recommendations,
    )
    from adwork.db.connection import get_db
    from adwork.optimization.bid_recommender import BidRecommender
    from adwork.optimization.budget_allocator import BudgetAllocator
    from adwork.models.bandit import run_bandit_simulation
    import pandas as pd

    print("=" * 60)
    print("  Ad-Work Optimization Pipeline")
    print("=" * 60)
    print()

    date_range = get_date_range()
    if not date_range:
        print("No data loaded. Run seed_demo.py first.")
        return

    start, end = date_range

    # ── Step 1: Bid Recommendations ──
    print("[1/3] Generating bid recommendations...")

    campaigns = get_all_campaigns()
    conn = get_db()
    all_metrics = conn.execute(
        "SELECT * FROM daily_metrics ORDER BY campaign_id, date"
    ).df()
    forecasts = get_stored_forecasts()

    recommender = BidRecommender()
    bid_recs = recommender.generate_recommendations(campaigns, all_metrics, forecasts)

    print(f"       Generated {len(bid_recs)} bid recommendations\n")

    for rec in bid_recs:
        emoji = "⬆️" if rec.recommended_action == "increase" else "⬇️" if rec.recommended_action == "decrease" else "➡️"
        print(f"  {emoji} {rec.campaign_name}")
        print(f"     {rec.recommended_action.upper()} {abs(rec.recommended_change_pct):.0f}% | "
              f"ROAS: {rec.current_roas:.1f}x → {rec.expected_roas_after:.1f}x | "
              f"Confidence: {rec.confidence.upper()}")
        print()

    # ── Step 2: Budget Allocation ──
    print("[2/3] Running budget allocation...")

    summary = get_campaign_summary(start, end)
    allocator = BudgetAllocator()
    allocation = allocator.allocate(summary)

    shifts = allocation["shifts"]
    print(f"       Generated {len(shifts)} budget shift recommendations\n")
    for shift in shifts:
        print(f"  💰 Move ${shift.amount:.0f}/day: {shift.from_campaign_name} → {shift.to_campaign_name}")
    print()

    # ── Step 3: Bandit Simulation ──
    print("[3/3] Running bandit simulation (validation)...")

    # Use the first Google campaign for the demo simulation
    demo_campaign = campaigns[campaigns["platform"] == "google"].iloc[0]
    demo_metrics = all_metrics[all_metrics["campaign_id"] == demo_campaign["campaign_id"]]

    recent = demo_metrics.tail(30)
    total_impr = recent["impressions"].sum()
    total_clicks = recent["clicks"].sum()
    total_conv = recent["conversions"].sum()
    total_spend = recent["spend"].sum()
    total_rev = recent["revenue"].sum()

    sim_results = run_bandit_simulation(
        base_metrics={
            "base_impressions": total_impr / len(recent),
            "base_ctr": total_clicks / total_impr if total_impr > 0 else 0.03,
            "base_cpc": total_spend / total_clicks if total_clicks > 0 else 1.5,
            "base_conv_rate": total_conv / total_clicks if total_clicks > 0 else 0.03,
            "base_aov": total_rev / total_conv if total_conv > 0 else 200,
        },
        n_rounds=90,
    )

    print()
    print("  Simulation Results (90 rounds):")
    print("  " + "-" * 50)
    for s in sim_results["strategies"]:
        final_regret = s.cumulative_regret[-1] if s.cumulative_regret else 0
        print(f"    {s.strategy:<30} Reward: ${s.total_reward:>10,.0f}  Regret: ${final_regret:>8,.0f}")
    print(f"    {'Oracle':<30} Reward: ${sim_results['oracle']['total_reward']:>10,.0f}")
    print()

    # ── Store recommendations ──
    clear_recommendations()
    recs_to_store = []

    for rec in bid_recs:
        recs_to_store.append({
            "campaign_id": rec.campaign_id,
            "action_type": "bid_adjustment",
            "action_detail": {
                "action": rec.recommended_action,
                "change_pct": rec.recommended_change_pct,
                "multiplier": rec.recommended_multiplier,
                "expected_roas": rec.expected_roas_after,
            },
            "reasoning": rec.reasoning,
            "confidence": rec.confidence,
        })

    for shift in shifts:
        recs_to_store.append({
            "campaign_id": shift.to_campaign,
            "action_type": "budget_reallocation",
            "action_detail": {
                "from": shift.from_campaign,
                "to": shift.to_campaign,
                "amount": shift.amount,
            },
            "reasoning": shift.reasoning,
            "confidence": shift.confidence,
        })

    stored = store_recommendations(recs_to_store)

    # Save simulation results for dashboard
    sim_path = Path("data/processed")
    sim_path.mkdir(parents=True, exist_ok=True)

    sim_serializable = {
        "strategies": [s.model_dump() for s in sim_results["strategies"]],
        "oracle": sim_results["oracle"],
        "n_rounds": sim_results["n_rounds"],
        "ts_beliefs": sim_results["ts_beliefs"],
    }
    with open(sim_path / "simulation_results.json", "w") as f:
        json.dump(sim_serializable, f, indent=2)

    print(f"  Stored {stored} recommendations in database")
    print(f"  Simulation results saved to data/processed/simulation_results.json")
    print()
    print("=" * 60)
    print("  Optimization complete! Run dashboard to see results.")
    print("=" * 60)


if __name__ == "__main__":
    main()