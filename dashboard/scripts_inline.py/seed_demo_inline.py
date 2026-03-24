# dashboard/scripts_inline/seed_demo_inline.py

"""
Inline version of the seed script.
Called directly by the Streamlit app when the database is empty.
This avoids subprocess issues on Streamlit Cloud.

Skips Google Trends (unreliable on cloud, and non-essential).
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
from loguru import logger


SEED = 42
START_DATE = date(2025, 4, 1)
END_DATE = date(2025, 6, 29)

DOW_MULTIPLIERS = {
    0: 1.00, 1: 1.05, 2: 1.05, 3: 1.10,
    4: 1.15, 5: 0.90, 6: 0.85,
}

CAMPAIGNS = [
    {"campaign_id": "ggl_brand_001", "campaign_name": "Google - Brand Search", "platform": "google", "daily_budget": 150.0, "base_impressions": 2500, "base_ctr": 0.12, "base_cpc": 0.75, "base_conv_rate": 0.10, "avg_order_value": 150.0, "trend_slope": 0.001},
    {"campaign_id": "ggl_laptops_002", "campaign_name": "Google - Search (Laptops)", "platform": "google", "daily_budget": 300.0, "base_impressions": 6500, "base_ctr": 0.04, "base_cpc": 2.75, "base_conv_rate": 0.03, "avg_order_value": 800.0, "trend_slope": 0.0015},
    {"campaign_id": "ggl_headphones_003", "campaign_name": "Google - Search (Headphones)", "platform": "google", "daily_budget": 200.0, "base_impressions": 5000, "base_ctr": 0.045, "base_cpc": 2.00, "base_conv_rate": 0.04, "avg_order_value": 120.0, "trend_slope": 0.0005},
    {"campaign_id": "ggl_shopping_004", "campaign_name": "Google - Shopping", "platform": "google", "daily_budget": 250.0, "base_impressions": 12000, "base_ctr": 0.015, "base_cpc": 0.60, "base_conv_rate": 0.05, "avg_order_value": 300.0, "trend_slope": 0.002},
    {"campaign_id": "meta_remarket_005", "campaign_name": "Meta - Remarketing", "platform": "meta", "daily_budget": 200.0, "base_impressions": 20000, "base_ctr": 0.015, "base_cpc": 0.45, "base_conv_rate": 0.04, "avg_order_value": 200.0, "trend_slope": 0.001},
    {"campaign_id": "meta_lal_006", "campaign_name": "Meta - Prospecting (Lookalike)", "platform": "meta", "daily_budget": 350.0, "base_impressions": 40000, "base_ctr": 0.008, "base_cpc": 1.10, "base_conv_rate": 0.015, "avg_order_value": 200.0, "trend_slope": -0.0005},
    {"campaign_id": "meta_interest_007", "campaign_name": "Meta - Prospecting (Interest)", "platform": "meta", "daily_budget": 250.0, "base_impressions": 50000, "base_ctr": 0.005, "base_cpc": 1.50, "base_conv_rate": 0.008, "avg_order_value": 180.0, "trend_slope": -0.001},
    {"campaign_id": "amz_sp_auto_008", "campaign_name": "Amazon - Sponsored Products (Auto)", "platform": "amazon", "daily_budget": 180.0, "base_impressions": 10000, "base_ctr": 0.008, "base_cpc": 0.95, "base_conv_rate": 0.10, "avg_order_value": 250.0, "trend_slope": 0.001},
    {"campaign_id": "amz_sp_manual_009", "campaign_name": "Amazon - Sponsored Products (Manual)", "platform": "amazon", "daily_budget": 220.0, "base_impressions": 6500, "base_ctr": 0.015, "base_cpc": 1.25, "base_conv_rate": 0.12, "avg_order_value": 300.0, "trend_slope": 0.002},
    {"campaign_id": "amz_sb_010", "campaign_name": "Amazon - Sponsored Brands", "platform": "amazon", "daily_budget": 150.0, "base_impressions": 25000, "base_ctr": 0.004, "base_cpc": 2.00, "base_conv_rate": 0.04, "avg_order_value": 200.0, "trend_slope": 0.0005},
]


def _generate_daily_data(campaign: dict, rng: np.random.Generator) -> list[dict]:
    """Generate daily metrics for one campaign."""
    rows = []
    num_days = (END_DATE - START_DATE).days + 1

    for day_offset in range(num_days):
        current_date = START_DATE + timedelta(days=day_offset)
        day_of_week = current_date.weekday()

        trend_factor = 1.0 + (campaign["trend_slope"] * day_offset)
        dow_factor = DOW_MULTIPLIERS[day_of_week]
        noise = rng.normal(1.0, 0.08)

        memorial_boost = 1.25 if date(2025, 5, 24) <= current_date <= date(2025, 5, 26) else 1.0
        prime_boost = 1.15 if campaign["platform"] == "amazon" and current_date >= date(2025, 6, 15) else 1.0

        impressions = max(0, int(campaign["base_impressions"] * trend_factor * dow_factor * noise * memorial_boost * prime_boost))

        effective_ctr = max(0.001, min(campaign["base_ctr"] * trend_factor * rng.normal(1.0, 0.10), 0.30))
        clicks = max(0, min(int(impressions * effective_ctr), impressions))

        effective_cpc = max(0.05, campaign["base_cpc"] * rng.normal(1.0, 0.12))
        spend = round(min(clicks * effective_cpc, campaign["daily_budget"] * 1.1), 2)

        effective_conv = max(0.001, min(campaign["base_conv_rate"] * trend_factor * rng.normal(1.0, 0.15), 0.30))
        conversions = max(0, min(int(clicks * effective_conv), clicks))

        revenue = round(conversions * campaign["avg_order_value"] * rng.normal(1.0, 0.05), 2)

        ctr = round(clicks / impressions, 6) if impressions > 0 else 0.0
        cpc = round(spend / clicks, 4) if clicks > 0 else 0.0
        roas = round(revenue / spend, 4) if spend > 0 else 0.0

        rows.append({
            "campaign_id": campaign["campaign_id"],
            "date": current_date,
            "impressions": impressions,
            "clicks": clicks,
            "conversions": conversions,
            "spend": spend,
            "revenue": revenue,
            "ctr": ctr, "cpc": cpc, "roas": roas,
        })

    return rows


def run_seed():
    """Seed the database with sample data. Called by the Streamlit app."""
    from adwork.db.connection import get_db

    logger.info("Auto-seeding demo data...")
    rng = np.random.default_rng(SEED)
    conn = get_db()

    for campaign in CAMPAIGNS:
        conn.execute("""
            INSERT OR REPLACE INTO campaigns 
            (campaign_id, campaign_name, platform, status, daily_budget)
            VALUES (?, ?, ?, 'active', ?)
        """, [campaign["campaign_id"], campaign["campaign_name"], campaign["platform"], campaign["daily_budget"]])

        for row in _generate_daily_data(campaign, rng):
            conn.execute("""
                INSERT OR REPLACE INTO daily_metrics
                (campaign_id, date, impressions, clicks, conversions, spend, revenue, ctr, cpc, roas)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                row["campaign_id"], row["date"], row["impressions"],
                row["clicks"], row["conversions"], row["spend"],
                row["revenue"], row["ctr"], row["cpc"], row["roas"],
            ])

    logger.info("Auto-seed complete")