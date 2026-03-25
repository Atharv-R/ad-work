# src/adwork/db/queries.py

"""
Database query functions.

Every data access goes through here — the dashboard and agent
never write raw SQL. This makes it easy to test, cache, and optimize.
"""

import json
from datetime import date, timedelta

import pandas as pd
from loguru import logger

from adwork.db.connection import get_db


def get_date_range() -> tuple[date, date] | None:
    """Get the min and max dates in the daily_metrics table."""
    conn = get_db()
    result = conn.execute(
        "SELECT MIN(date), MAX(date) FROM daily_metrics"
    ).fetchone()

    if result and result[0] is not None:
        return result[0], result[1]
    return None


def get_kpis(start_date: date, end_date: date) -> dict:
    """
    Aggregate KPIs across all campaigns for a date range.
    
    Returns:
        Dict with total_spend, total_revenue, total_conversions,
        total_clicks, total_impressions, overall_roas, overall_ctr,
        active_campaigns
    """
    conn = get_db()

    row = conn.execute("""
        SELECT 
            COALESCE(SUM(spend), 0) as total_spend,
            COALESCE(SUM(revenue), 0) as total_revenue,
            COALESCE(SUM(conversions), 0) as total_conversions,
            COALESCE(SUM(clicks), 0) as total_clicks,
            COALESCE(SUM(impressions), 0) as total_impressions,
            COUNT(DISTINCT campaign_id) as active_campaigns
        FROM daily_metrics
        WHERE date BETWEEN ? AND ?
    """, [start_date, end_date]).fetchone()

    total_spend = row[0]
    total_revenue = row[1]

    return {
        "total_spend": total_spend,
        "total_revenue": total_revenue,
        "total_conversions": int(row[2]),
        "total_clicks": int(row[3]),
        "total_impressions": int(row[4]),
        "active_campaigns": int(row[5]),
        "overall_roas": round(total_revenue / total_spend, 2) if total_spend > 0 else 0,
        "overall_ctr": round(row[3] / row[4], 4) if row[4] > 0 else 0,
    }


def get_kpis_comparison(start_date: date, end_date: date) -> dict:
    """
    Get KPIs for current period AND the preceding period of equal length.
    Used for delta calculations on dashboard metric cards.
    """
    current = get_kpis(start_date, end_date)

    period_length = (end_date - start_date).days
    prev_end = start_date - timedelta(days=1)
    prev_start = prev_end - timedelta(days=period_length)

    previous = get_kpis(prev_start, prev_end)

    def delta(current_val, prev_val, is_currency=False, is_pct=False):
        if prev_val == 0:
            return None
        diff = current_val - prev_val
        if is_currency:
            return f"${diff:+,.0f} vs prior"
        if is_pct:
            pct_change = (diff / prev_val) * 100
            return f"{pct_change:+.1f}%"
        return f"{diff:+,.0f}"

    return {
        "kpis": current,
        "deltas": {
            "spend": delta(current["total_spend"], previous["total_spend"], is_currency=True),
            "roas": delta(current["overall_roas"], previous["overall_roas"], is_pct=True),
            "conversions": delta(current["total_conversions"], previous["total_conversions"], is_pct=True),
            "campaigns": None,  # No meaningful delta for count
        },
    }


def get_daily_spend_by_platform(start_date: date, end_date: date) -> pd.DataFrame:
    """
    Daily spend broken down by platform.
    Returns DataFrame with columns: date, platform, spend
    """
    conn = get_db()

    return conn.execute("""
        SELECT 
            dm.date,
            c.platform,
            SUM(dm.spend) as spend
        FROM daily_metrics dm
        JOIN campaigns c ON dm.campaign_id = c.campaign_id
        WHERE dm.date BETWEEN ? AND ?
        GROUP BY dm.date, c.platform
        ORDER BY dm.date
    """, [start_date, end_date]).df()


def get_daily_roas(start_date: date, end_date: date) -> pd.DataFrame:
    """
    Daily ROAS across all campaigns.
    Returns DataFrame with columns: date, spend, revenue, roas
    """
    conn = get_db()

    return conn.execute("""
        SELECT 
            date,
            SUM(spend) as spend,
            SUM(revenue) as revenue,
            CASE WHEN SUM(spend) > 0 
                THEN ROUND(SUM(revenue) / SUM(spend), 2) 
                ELSE 0 
            END as roas
        FROM daily_metrics
        WHERE date BETWEEN ? AND ?
        GROUP BY date
        ORDER BY date
    """, [start_date, end_date]).df()


def get_campaign_summary(start_date: date, end_date: date) -> pd.DataFrame:
    """
    Per-campaign aggregated performance.
    Returns DataFrame sorted by spend (highest first).
    """
    conn = get_db()

    return conn.execute("""
        SELECT 
            c.campaign_name,
            c.platform,
            SUM(dm.impressions) as impressions,
            SUM(dm.clicks) as clicks,
            ROUND(SUM(dm.clicks)::DOUBLE / NULLIF(SUM(dm.impressions), 0), 4) as ctr,
            SUM(dm.spend) as spend,
            ROUND(SUM(dm.spend) / NULLIF(SUM(dm.clicks), 0), 2) as avg_cpc,
            SUM(dm.conversions) as conversions,
            ROUND(SUM(dm.conversions)::DOUBLE / NULLIF(SUM(dm.clicks), 0), 4) as conv_rate,
            SUM(dm.revenue) as revenue,
            ROUND(SUM(dm.revenue) / NULLIF(SUM(dm.spend), 0), 2) as roas
        FROM daily_metrics dm
        JOIN campaigns c ON dm.campaign_id = c.campaign_id
        WHERE dm.date BETWEEN ? AND ?
        GROUP BY c.campaign_name, c.platform
        ORDER BY spend DESC
    """, [start_date, end_date]).df()


def get_daily_metrics_for_campaign(campaign_id: str) -> pd.DataFrame:
    """Get daily time series for a specific campaign."""
    conn = get_db()

    return conn.execute("""
        SELECT * FROM daily_metrics
        WHERE campaign_id = ?
        ORDER BY date
    """, [campaign_id]).df()


def get_all_campaigns() -> pd.DataFrame:
    """Get all campaigns."""
    conn = get_db()
    return conn.execute("SELECT * FROM campaigns ORDER BY platform, campaign_name").df()


def get_trends(keywords: list[str] | None = None) -> pd.DataFrame:
    """
    Get stored Google Trends data.
    If keywords is None, return all trends.
    """
    conn = get_db()

    if keywords:
        placeholders = ", ".join(["?" for _ in keywords])
        return conn.execute(f"""
            SELECT keyword, date, interest
            FROM search_trends
            WHERE keyword IN ({placeholders})
            ORDER BY keyword, date
        """, keywords).df()
    else:
        return conn.execute("""
            SELECT keyword, date, interest
            FROM search_trends
            ORDER BY keyword, date
        """).df()


def has_data() -> bool:
    """Check if the database has any campaign data loaded."""
    conn = get_db()
    result = conn.execute("SELECT COUNT(*) FROM daily_metrics").fetchone()
    return result[0] > 0 if result else False

# src/adwork/db/queries.py  — ADD these functions at the bottom of the existing file


def store_forecast_results(campaign_id: str, metric: str, forecast_df: pd.DataFrame) -> int:
    """
    Store forecast results in DuckDB.
    Replaces any existing forecast for this campaign/metric.
    Only stores the future (forecasted) rows.
    
    Returns number of rows stored.
    """
    conn = get_db()

    # Clear old forecasts for this campaign/metric
    conn.execute(
        "DELETE FROM forecasts WHERE campaign_id = ? AND metric = ?",
        [campaign_id, metric],
    )

    future = forecast_df[forecast_df["is_forecast"]].copy()
    rows = 0

    for _, row in future.iterrows():
        forecast_date = row["date"]
        # Handle both Timestamp and date objects
        if hasattr(forecast_date, "date"):
            forecast_date = forecast_date.date()

        conn.execute("""
            INSERT INTO forecasts (campaign_id, forecast_date, metric, predicted_value, lower_bound, upper_bound)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            campaign_id,
            forecast_date,
            metric,
            float(row["predicted"]),
            float(row["lower"]),
            float(row["upper"]),
        ])
        rows += 1

    logger.info(f"Stored {rows} forecast rows for {campaign_id}/{metric}")
    return rows


def get_stored_forecasts(campaign_id: str | None = None, metric: str | None = None) -> pd.DataFrame:
    """Get stored forecasts, optionally filtered."""
    conn = get_db()

    query = "SELECT * FROM forecasts WHERE 1=1"
    params = []

    if campaign_id:
        query += " AND campaign_id = ?"
        params.append(campaign_id)
    if metric:
        query += " AND metric = ?"
        params.append(metric)

    query += " ORDER BY campaign_id, metric, forecast_date"
    return conn.execute(query, params).df()


def get_forecast_summary() -> pd.DataFrame:
    """Get a summary of all stored forecasts."""
    conn = get_db()

    return conn.execute("""
        SELECT 
            f.campaign_id,
            c.campaign_name,
            c.platform,
            f.metric,
            COUNT(*) as forecast_days,
            MIN(f.forecast_date) as first_date,
            MAX(f.forecast_date) as last_date,
            ROUND(AVG(f.predicted_value), 1) as avg_predicted,
            MAX(f.created_at) as generated_at
        FROM forecasts f
        JOIN campaigns c ON f.campaign_id = c.campaign_id
        GROUP BY f.campaign_id, c.campaign_name, c.platform, f.metric
        ORDER BY c.platform, c.campaign_name, f.metric
    """).df()


def has_forecasts() -> bool:
    """Check if any forecasts have been generated."""
    conn = get_db()
    result = conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()
    return result[0] > 0 if result else False


def store_recommendations(recommendations: list[dict]) -> int:
    """Store optimization recommendations in DuckDB."""
    conn = get_db()
    count = 0

    for rec in recommendations:
        conn.execute("""
            INSERT INTO recommendations 
            (campaign_id, action_type, action_detail, reasoning, confidence, status)
            VALUES (?, ?, ?, ?, ?, 'pending')
        """, [
            rec.get("campaign_id"),
            rec.get("action_type", "bid_adjustment"),
            json.dumps(rec.get("action_detail", {})),
            rec.get("reasoning", ""),
            rec.get("confidence", "medium"),
        ])
        count += 1

    logger.info(f"Stored {count} recommendations")
    return count


def get_recent_recommendations(limit: int = 20) -> pd.DataFrame:
    """Get most recent recommendations with campaign names."""
    conn = get_db()

    return conn.execute("""
        SELECT 
            r.campaign_id,
            c.campaign_name,
            c.platform,
            r.action_type,
            r.action_detail,
            r.reasoning,
            r.confidence,
            r.status,
            r.created_at
        FROM recommendations r
        LEFT JOIN campaigns c ON r.campaign_id = c.campaign_id
        ORDER BY r.created_at DESC
        LIMIT ?
    """, [limit]).df()


def clear_recommendations() -> None:
    """Clear all existing recommendations."""
    conn = get_db()
    conn.execute("DELETE FROM recommendations")

# ── Competitor queries ───────────────────────────────────────────────

def get_competitor_ads(
    advertiser: str | None = None,
    platform: str | None = None,
    category: str | None = None,
) -> pd.DataFrame:
    """Retrieve competitor ads with optional filters."""
    conn = get_db()
    query = "SELECT * FROM competitor_ads WHERE 1=1"
    params: list = []
    if advertiser:
        query += " AND advertiser_name = ?"
        params.append(advertiser)
    if platform:
        query += " AND platform = ?"
        params.append(platform)
    if category:
        query += " AND category = ?"
        params.append(category)
    query += " ORDER BY last_seen DESC"
    return conn.execute(query, params).fetchdf()


def has_competitor_data() -> bool:
    conn = get_db()
    result = conn.execute("SELECT COUNT(*) FROM competitor_ads").fetchone()
    return result[0] > 0 if result else False


def store_cluster_results(clusters: list[dict], analysis_date: str) -> int:
    """Store cluster analysis results. Returns rows stored."""
    conn = get_db()
    rows = [
        (c["cluster_id"], c["label"], json.dumps(c["top_terms"]),
         c["n_ads"], analysis_date)
        for c in clusters
    ]
    conn.executemany(
        """INSERT INTO competitor_clusters
           (cluster_id, cluster_label, top_terms, n_ads, analysis_date)
           VALUES (?,?,?,?,?)
           ON CONFLICT (cluster_id, analysis_date) DO UPDATE SET
             cluster_label=excluded.cluster_label,
             top_terms=excluded.top_terms,
             n_ads=excluded.n_ads""",
        rows,
    )
    return len(rows)


def get_cluster_results() -> pd.DataFrame:
    """Get most recent cluster analysis."""
    conn = get_db()
    return conn.execute("""
        SELECT * FROM competitor_clusters
        WHERE analysis_date = (SELECT MAX(analysis_date) FROM competitor_clusters)
        ORDER BY cluster_id
    """).fetchdf()


def get_competitor_advertisers() -> list[str]:
    """Get distinct advertiser names."""
    conn = get_db()
    result = conn.execute(
        "SELECT DISTINCT advertiser_name FROM competitor_ads ORDER BY advertiser_name"
    ).fetchdf()
    return result["advertiser_name"].tolist() if not result.empty else []


def clear_competitor_data() -> None:
    conn = get_db()
    conn.execute("DELETE FROM competitor_ads")
    conn.execute("DELETE FROM competitor_clusters")