# scripts/seed_demo.py

"""
Seed the Ad-Work demo database with realistic sample data.

Run with:
    uv run python scripts/seed_demo.py

    Or on Windows if uv run doesn't work:
    .venv\Scripts\python scripts\seed_demo.py

This generates:
1. 10 campaigns across Google, Meta, and Amazon
2. 90 days of daily performance metrics with realistic patterns
3. Google Trends data for product categories
4. Saves CSVs to data/sample_campaigns/ as format examples
"""

import sys
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from datetime import date, timedelta
from loguru import logger

from adwork.db.connection import get_db, reset_db


# ─── Configuration ───────────────────────────────────────────

SEED = 42
START_DATE = date(2025, 4, 1)
END_DATE = date(2025, 6, 29)  # ~90 days

# Day-of-week multipliers (Mon=0, Sun=6) — typical e-commerce
DOW_MULTIPLIERS = {
    0: 1.00,  # Monday
    1: 1.05,  # Tuesday
    2: 1.05,  # Wednesday
    3: 1.10,  # Thursday
    4: 1.15,  # Friday
    5: 0.90,  # Saturday
    6: 0.85,  # Sunday
}

# Campaign definitions with realistic parameters
CAMPAIGNS = [
    # Google Ads
    {
        "campaign_id": "ggl_brand_001",
        "campaign_name": "Google - Brand Search",
        "platform": "google",
        "daily_budget": 150.0,
        "base_impressions": 2500,
        "base_ctr": 0.12,
        "base_cpc": 0.75,
        "base_conv_rate": 0.10,
        "avg_order_value": 150.0,
        "trend_slope": 0.001,  # Slight growth
    },
    {
        "campaign_id": "ggl_laptops_002",
        "campaign_name": "Google - Search (Laptops)",
        "platform": "google",
        "daily_budget": 300.0,
        "base_impressions": 6500,
        "base_ctr": 0.04,
        "base_cpc": 2.75,
        "base_conv_rate": 0.03,
        "avg_order_value": 800.0,
        "trend_slope": 0.0015,
    },
    {
        "campaign_id": "ggl_headphones_003",
        "campaign_name": "Google - Search (Headphones)",
        "platform": "google",
        "daily_budget": 200.0,
        "base_impressions": 5000,
        "base_ctr": 0.045,
        "base_cpc": 2.00,
        "base_conv_rate": 0.04,
        "avg_order_value": 120.0,
        "trend_slope": 0.0005,
    },
    {
        "campaign_id": "ggl_shopping_004",
        "campaign_name": "Google - Shopping",
        "platform": "google",
        "daily_budget": 250.0,
        "base_impressions": 12000,
        "base_ctr": 0.015,
        "base_cpc": 0.60,
        "base_conv_rate": 0.05,
        "avg_order_value": 300.0,
        "trend_slope": 0.002,
    },
    # Meta Ads
    {
        "campaign_id": "meta_remarket_005",
        "campaign_name": "Meta - Remarketing",
        "platform": "meta",
        "daily_budget": 200.0,
        "base_impressions": 20000,
        "base_ctr": 0.015,
        "base_cpc": 0.45,
        "base_conv_rate": 0.04,
        "avg_order_value": 200.0,
        "trend_slope": 0.001,
    },
    {
        "campaign_id": "meta_lal_006",
        "campaign_name": "Meta - Prospecting (Lookalike)",
        "platform": "meta",
        "daily_budget": 350.0,
        "base_impressions": 40000,
        "base_ctr": 0.008,
        "base_cpc": 1.10,
        "base_conv_rate": 0.015,
        "avg_order_value": 200.0,
        "trend_slope": -0.0005,  # Slowly declining (audience fatigue)
    },
    {
        "campaign_id": "meta_interest_007",
        "campaign_name": "Meta - Prospecting (Interest)",
        "platform": "meta",
        "daily_budget": 250.0,
        "base_impressions": 50000,
        "base_ctr": 0.005,
        "base_cpc": 1.50,
        "base_conv_rate": 0.008,
        "avg_order_value": 180.0,
        "trend_slope": -0.001,  # Declining — should trigger agent recommendation
    },
    # Amazon Ads
    {
        "campaign_id": "amz_sp_auto_008",
        "campaign_name": "Amazon - Sponsored Products (Auto)",
        "platform": "amazon",
        "daily_budget": 180.0,
        "base_impressions": 10000,
        "base_ctr": 0.008,
        "base_cpc": 0.95,
        "base_conv_rate": 0.10,
        "avg_order_value": 250.0,
        "trend_slope": 0.001,
    },
    {
        "campaign_id": "amz_sp_manual_009",
        "campaign_name": "Amazon - Sponsored Products (Manual)",
        "platform": "amazon",
        "daily_budget": 220.0,
        "base_impressions": 6500,
        "base_ctr": 0.015,
        "base_cpc": 1.25,
        "base_conv_rate": 0.12,
        "avg_order_value": 300.0,
        "trend_slope": 0.002,
    },
    {
        "campaign_id": "amz_sb_010",
        "campaign_name": "Amazon - Sponsored Brands",
        "platform": "amazon",
        "daily_budget": 150.0,
        "base_impressions": 25000,
        "base_ctr": 0.004,
        "base_cpc": 2.00,
        "base_conv_rate": 0.04,
        "avg_order_value": 200.0,
        "trend_slope": 0.0005,
    },
]


def generate_daily_data(campaign: dict, rng: np.random.Generator) -> list[dict]:
    """Generate 90 days of realistic daily metrics for one campaign."""
    rows = []
    num_days = (END_DATE - START_DATE).days + 1

    for day_offset in range(num_days):
        current_date = START_DATE + timedelta(days=day_offset)
        day_of_week = current_date.weekday()
        day_fraction = day_offset / num_days  # 0.0 → 1.0 over the period

        # --- Impressions ---
        # Base + trend + day-of-week + noise
        trend_factor = 1.0 + (campaign["trend_slope"] * day_offset)
        dow_factor = DOW_MULTIPLIERS[day_of_week]
        noise_factor = rng.normal(1.0, 0.08)  # ±8% daily noise

        # Memorial Day weekend boost (May 24-26, 2025)
        memorial_boost = 1.0
        if date(2025, 5, 24) <= current_date <= date(2025, 5, 26):
            memorial_boost = 1.25

        # June Prime Day prep boost for Amazon
        prime_boost = 1.0
        if campaign["platform"] == "amazon" and current_date >= date(2025, 6, 15):
            prime_boost = 1.15

        impressions = int(
            campaign["base_impressions"]
            * trend_factor
            * dow_factor
            * noise_factor
            * memorial_boost
            * prime_boost
        )
        impressions = max(0, impressions)

        # --- Clicks ---
        ctr_noise = rng.normal(1.0, 0.10)  # ±10% CTR variation
        effective_ctr = campaign["base_ctr"] * trend_factor * ctr_noise
        effective_ctr = max(0.001, min(effective_ctr, 0.30))  # Clamp

        clicks = int(impressions * effective_ctr)
        clicks = max(0, min(clicks, impressions))

        # --- Spend ---
        cpc_noise = rng.normal(1.0, 0.12)  # CPC varies more
        effective_cpc = campaign["base_cpc"] * cpc_noise
        effective_cpc = max(0.05, effective_cpc)

        spend = round(clicks * effective_cpc, 2)
        # Cap at daily budget (with some overshoot allowed, like real platforms)
        spend = min(spend, campaign["daily_budget"] * 1.1)

        # --- Conversions ---
        conv_noise = rng.normal(1.0, 0.15)
        effective_conv_rate = campaign["base_conv_rate"] * trend_factor * conv_noise
        effective_conv_rate = max(0.001, min(effective_conv_rate, 0.30))

        conversions = int(clicks * effective_conv_rate)
        conversions = max(0, min(conversions, clicks))

        # --- Revenue ---
        aov_noise = rng.normal(1.0, 0.05)
        effective_aov = campaign["avg_order_value"] * aov_noise
        revenue = round(conversions * effective_aov, 2)

        # --- Computed metrics ---
        ctr = round(clicks / impressions, 6) if impressions > 0 else 0.0
        cpc = round(spend / clicks, 4) if clicks > 0 else 0.0
        roas = round(revenue / spend, 4) if spend > 0 else 0.0

        rows.append({
            "campaign_id": campaign["campaign_id"],
            "campaign_name": campaign["campaign_name"],
            "platform": campaign["platform"],
            "date": current_date,
            "impressions": impressions,
            "clicks": clicks,
            "conversions": conversions,
            "spend": spend,
            "revenue": revenue,
            "ctr": ctr,
            "cpc": cpc,
            "roas": roas,
        })

    return rows


def seed_campaigns_and_metrics():
    """Generate and load all sample campaign data into DuckDB."""
    rng = np.random.default_rng(SEED)
    conn = get_db()

    all_rows = []

    for campaign in CAMPAIGNS:
        # Insert campaign
        conn.execute("""
            INSERT OR REPLACE INTO campaigns 
            (campaign_id, campaign_name, platform, status, daily_budget)
            VALUES (?, ?, ?, 'active', ?)
        """, [
            campaign["campaign_id"],
            campaign["campaign_name"],
            campaign["platform"],
            campaign["daily_budget"],
        ])

        # Generate daily data
        rows = generate_daily_data(campaign, rng)
        all_rows.extend(rows)

        logger.info(
            f"  Generated {len(rows)} days for {campaign['campaign_name']}"
        )

    # Bulk insert daily metrics
    metrics_df = pd.DataFrame(all_rows)

    for _, row in metrics_df.iterrows():
        conn.execute("""
            INSERT OR REPLACE INTO daily_metrics
            (campaign_id, date, impressions, clicks, conversions, spend, revenue, ctr, cpc, roas)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            row["campaign_id"], row["date"], int(row["impressions"]),
            int(row["clicks"]), int(row["conversions"]), float(row["spend"]),
            float(row["revenue"]), float(row["ctr"]), float(row["cpc"]),
            float(row["roas"]),
        ])

    logger.info(f"Loaded {len(all_rows)} total metric rows")

    # Save CSVs as format examples
    sample_dir = Path(__file__).resolve().parent.parent / "data" / "sample_campaigns"
    sample_dir.mkdir(parents=True, exist_ok=True)

    for platform in ["google", "meta", "amazon"]:
        platform_df = metrics_df[metrics_df["platform"] == platform].copy()
        # Drop internal columns for the CSV export
        export_df = platform_df.drop(columns=["ctr", "cpc", "roas"])
        filepath = sample_dir / f"sample_{platform}_ads.csv"
        export_df.to_csv(filepath, index=False)
        logger.info(f"  Saved {filepath.name} ({len(export_df)} rows)")

    # Also save a combined CSV
    combined_path = sample_dir / "sample_all_platforms.csv"
    metrics_df.drop(columns=["ctr", "cpc", "roas"]).to_csv(combined_path, index=False)
    logger.info(f"  Saved {combined_path.name} ({len(metrics_df)} rows)")


def seed_trends():
    """Fetch and store Google Trends data."""
    logger.info("Fetching Google Trends data...")

    try:
        from adwork.data.trends import fetch_and_store_trends

        count = fetch_and_store_trends(
            keywords=["laptop", "headphones", "computer monitor", "wireless earbuds", "gaming keyboard"],
            timeframe="today 3-m",
        )
        logger.info(f"Stored {count} trend data points")

    except Exception as e:
        logger.warning(f"Google Trends fetch failed (non-critical): {e}")
        logger.info("Dashboard will work fine without trends data.")
        logger.info("You can retry later with: uv run python -c \"from adwork.data.trends import fetch_and_store_trends; fetch_and_store_trends()\"")


def main():
    print("=" * 60)
    print("  Ad-Work Demo Data Seeder")
    print("=" * 60)
    print()

    # Reset database to clean state
    logger.info("Resetting database...")
    reset_db()

    # Generate campaign data
    logger.info("Generating sample campaign data (10 campaigns × 90 days)...")
    seed_campaigns_and_metrics()

    # Fetch Google Trends
    seed_trends()

    # Summary
    conn = get_db()
    campaign_count = conn.execute("SELECT COUNT(*) FROM campaigns").fetchone()[0]
    metric_count = conn.execute("SELECT COUNT(*) FROM daily_metrics").fetchone()[0]
    trend_count = conn.execute("SELECT COUNT(*) FROM search_trends").fetchone()[0]

    print()
    print("=" * 60)
    print("  Seeding Complete!")
    print(f"  Campaigns:     {campaign_count}")
    print(f"  Daily metrics: {metric_count}")
    print(f"  Trend points:  {trend_count}")
    print("=" * 60)
    print()
    print("Run the dashboard:  uv run streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()