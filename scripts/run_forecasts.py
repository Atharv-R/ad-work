# scripts/run_forecasts.py

"""
Generate forecasts for all campaigns.

Run with:
    uv run python scripts/run_forecasts.py

    Or on Windows:
    .venv\\Scripts\\python scripts\\run_forecasts.py

Options:
    --campaign CAMPAIGN_ID    Forecast a single campaign
    --metrics clicks,spend    Which metrics to forecast (comma-separated)
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loguru import logger


def run_forecasts(campaign_filter: str | None = None, metrics: list[str] | None = None):
    """Generate and store forecasts."""
    from adwork.db.queries import (
        get_all_campaigns,
        get_daily_metrics_for_campaign,
        get_trends,
        store_forecast_results,
    )
    from adwork.models.forecaster import DemandForecaster

    if metrics is None:
        metrics = ["clicks", "conversions"]

    campaigns = get_all_campaigns()

    if campaign_filter:
        campaigns = campaigns[campaigns["campaign_id"] == campaign_filter]
        if campaigns.empty:
            logger.error(f"Campaign '{campaign_filter}' not found")
            return

    # Load trends data once
    trends_df = get_trends()

    forecaster = DemandForecaster()
    results_summary = []

    total = len(campaigns) * len(metrics)
    done = 0

    for _, campaign in campaigns.iterrows():
        cid = campaign["campaign_id"]
        cname = campaign["campaign_name"]

        # Get historical metrics
        hist = get_daily_metrics_for_campaign(cid)
        if len(hist) < 30:
            logger.warning(f"Skipping {cname}: only {len(hist)} days (need 30+)")
            done += len(metrics)
            continue

        # Get matching trends for this campaign
        campaign_trends = _match_trends(cname, trends_df)

        for metric in metrics:
            done += 1
            logger.info(f"[{done}/{total}] Forecasting {metric} for {cname}")

            try:
                result = forecaster.run(
                    historical_df=hist,
                    metric=metric,
                    campaign_id=cid,
                    trends_df=campaign_trends,
                )

                # Store forecast in DB
                rows = store_forecast_results(cid, metric, result["forecast"])

                results_summary.append({
                    "campaign": cname,
                    "metric": metric,
                    "engine": result["engine"],
                    "mape": result["evaluation"]["mape"],
                    "coverage": result["evaluation"]["coverage"],
                    "forecast_rows": rows,
                })

            except Exception as e:
                logger.error(f"Failed {cname}/{metric}: {e}")
                results_summary.append({
                    "campaign": cname,
                    "metric": metric,
                    "engine": "error",
                    "mape": None,
                    "coverage": None,
                    "forecast_rows": 0,
                })

    # Print summary
    print()
    print("=" * 70)
    print("  Forecast Generation Summary")
    print("=" * 70)
    print(f"  {'Campaign':<40} {'Metric':<12} {'MAPE':>8} {'Coverage':>10}")
    print("-" * 70)

    for r in results_summary:
        mape_str = f"{r['mape']:.1f}%" if r["mape"] is not None else "ERROR"
        cov_str = f"{r['coverage']:.0%}" if r["coverage"] is not None else "—"
        print(f"  {r['campaign']:<40} {r['metric']:<12} {mape_str:>8} {cov_str:>10}")

    print("=" * 70)


def _match_trends(campaign_name: str, trends_df) -> pd.DataFrame | None:
    """
    Match a campaign to relevant Google Trends keywords
    based on campaign name.
    """
    import pandas as pd

    if trends_df is None or trends_df.empty:
        return None

    name_lower = campaign_name.lower()

    keyword_matches = {
        "laptop": ["laptop"],
        "headphone": ["headphones"],
        "monitor": ["computer monitor"],
        "earbud": ["wireless earbuds"],
        "keyboard": ["gaming keyboard"],
    }

    matched_keywords = []
    for trigger, keywords in keyword_matches.items():
        if trigger in name_lower:
            matched_keywords.extend(keywords)

    # Default: use all trends for generic campaigns (shopping, brand, etc.)
    if not matched_keywords:
        matched_keywords = trends_df["keyword"].unique().tolist()[:3]

    filtered = trends_df[trends_df["keyword"].isin(matched_keywords)]
    return filtered if not filtered.empty else None


if __name__ == "__main__":
    import pandas as pd

    parser = argparse.ArgumentParser(description="Generate Ad-Work forecasts")
    parser.add_argument("--campaign", type=str, default=None, help="Single campaign ID")
    parser.add_argument("--metrics", type=str, default="clicks,conversions",
                        help="Comma-separated metrics to forecast")
    args = parser.parse_args()

    metrics_list = [m.strip() for m in args.metrics.split(",")]
    run_forecasts(campaign_filter=args.campaign, metrics=metrics_list)