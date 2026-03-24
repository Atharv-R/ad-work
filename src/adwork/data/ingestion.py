# src/adwork/data/ingestion.py

"""
Data ingestion pipeline.

Handles:
1. CSV upload parsing (Google Ads, Meta Ads, Amazon Ads, or internal format)
2. Platform auto-detection from column names
3. Column normalization to internal schema
4. Pydantic validation on every row
5. Loading into DuckDB

This is the "bring your own data" feature — exactly how real tools
like Optmyzr and Adalysis work.
"""

import pandas as pd
from datetime import datetime
from loguru import logger

from adwork.data.schemas import (
    Platform,
    Campaign,
    DailyMetrics,
    detect_platform_from_columns,
    GOOGLE_ADS_COLUMNS,
    META_ADS_COLUMNS,
    AMAZON_ADS_COLUMNS,
)
from adwork.db.connection import get_db


def ingest_csv(
    df: pd.DataFrame,
    platform_override: Platform | None = None,
) -> dict:
    """
    Main ingestion entry point.
    
    Takes a pandas DataFrame (from CSV upload), detects the platform,
    normalizes columns, validates, and loads into DuckDB.
    
    Args:
        df: Raw DataFrame from CSV
        platform_override: Force a specific platform (skip auto-detection)
        
    Returns:
        Dict with ingestion results:
        {
            "status": "success" | "partial" | "error",
            "platform_detected": str,
            "rows_total": int,
            "rows_loaded": int,
            "rows_skipped": int,
            "errors": list[str],
            "campaigns_found": list[str],
        }
    """
    result = {
        "status": "error",
        "platform_detected": "unknown",
        "rows_total": len(df),
        "rows_loaded": 0,
        "rows_skipped": 0,
        "errors": [],
        "campaigns_found": [],
    }

    # Step 1: Detect platform
    if platform_override:
        platform = platform_override
    else:
        platform = detect_platform_from_columns(df.columns.tolist())

    result["platform_detected"] = platform.value
    logger.info(f"Detected platform: {platform.value} | {len(df)} rows")

    # Step 2: Normalize columns
    try:
        normalized = _normalize_columns(df, platform)
    except Exception as e:
        result["errors"].append(f"Column normalization failed: {e}")
        return result

    # Step 3: Validate and load
    conn = get_db()
    campaigns_seen = {}
    rows_loaded = 0
    errors = []

    for idx, row in normalized.iterrows():
        try:
            # Validate metrics
            metrics = DailyMetrics(
                campaign_id=str(row["campaign_id"]).strip(),
                date=pd.to_datetime(row["date"]).date(),
                impressions=max(0, int(row.get("impressions", 0))),
                clicks=max(0, int(row.get("clicks", 0))),
                conversions=max(0, int(row.get("conversions", 0))),
                spend=max(0.0, float(row.get("spend", 0))),
                revenue=max(0.0, float(row.get("revenue", 0))),
            )

            # Track campaigns
            cid = metrics.campaign_id
            if cid not in campaigns_seen:
                campaign_name = str(row.get("campaign_name", f"Campaign {cid}")).strip()
                # Determine platform for this campaign
                row_platform = row.get("platform", platform.value)
                if row_platform == "unknown" or not row_platform:
                    row_platform = platform.value

                campaigns_seen[cid] = Campaign(
                    campaign_id=cid,
                    campaign_name=campaign_name,
                    platform=Platform(row_platform) if row_platform != "unknown" else Platform.GOOGLE,
                    daily_budget=float(row.get("daily_budget", 0)),
                )

            # Insert metrics into DuckDB
            conn.execute("""
                INSERT OR REPLACE INTO daily_metrics 
                (campaign_id, date, impressions, clicks, conversions, spend, revenue, ctr, cpc, roas)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                metrics.campaign_id,
                metrics.date,
                metrics.impressions,
                metrics.clicks,
                metrics.conversions,
                metrics.spend,
                metrics.revenue,
                metrics.ctr,
                metrics.cpc,
                metrics.roas,
            ])

            rows_loaded += 1

        except Exception as e:
            errors.append(f"Row {idx}: {e}")
            continue

    # Insert campaigns
    for campaign in campaigns_seen.values():
        try:
            conn.execute("""
                INSERT OR REPLACE INTO campaigns 
                (campaign_id, campaign_name, platform, status, daily_budget)
                VALUES (?, ?, ?, ?, ?)
            """, [
                campaign.campaign_id,
                campaign.campaign_name,
                campaign.platform.value,
                campaign.status.value,
                campaign.daily_budget,
            ])
        except Exception as e:
            errors.append(f"Campaign {campaign.campaign_id}: {e}")

    # Build result
    result["rows_loaded"] = rows_loaded
    result["rows_skipped"] = len(df) - rows_loaded
    result["errors"] = errors[:20]  # Cap error list
    result["campaigns_found"] = list(campaigns_seen.keys())

    if rows_loaded == len(df):
        result["status"] = "success"
    elif rows_loaded > 0:
        result["status"] = "partial"
    else:
        result["status"] = "error"

    logger.info(
        f"Ingestion complete: {rows_loaded}/{len(df)} rows loaded, "
        f"{len(errors)} errors, {len(campaigns_seen)} campaigns"
    )

    return result


def _normalize_columns(df: pd.DataFrame, platform: Platform) -> pd.DataFrame:
    """
    Rename platform-specific columns to our internal schema.
    
    Each platform exports CSVs with different column names.
    This function maps them all to a common format.
    """
    df = df.copy()

    if platform == Platform.GOOGLE:
        column_map = {v_orig: v_internal for v_orig, v_internal in GOOGLE_ADS_COLUMNS.items()}
        df = df.rename(columns=column_map)

        # Google exports CTR as "5.23%" string — clean it
        if "_ctr" in df.columns:
            df.drop(columns=["_ctr"], inplace=True, errors="ignore")
        if "_cpc" in df.columns:
            df.drop(columns=["_cpc"], inplace=True, errors="ignore")

        # Add platform column
        df["platform"] = "google"

    elif platform == Platform.META:
        column_map = {v_orig: v_internal for v_orig, v_internal in META_ADS_COLUMNS.items()}
        df = df.rename(columns=column_map)
        df["platform"] = "meta"

    elif platform == Platform.AMAZON:
        column_map = {v_orig: v_internal for v_orig, v_internal in AMAZON_ADS_COLUMNS.items()}
        df = df.rename(columns=column_map)
        df["platform"] = "amazon"

    else:
        # Internal format — already has correct column names
        pass

    # Ensure required columns exist with defaults
    for col, default in [
        ("campaign_id", "unknown"),
        ("campaign_name", "Unknown Campaign"),
        ("platform", "unknown"),
        ("date", None),
        ("impressions", 0),
        ("clicks", 0),
        ("conversions", 0),
        ("spend", 0.0),
        ("revenue", 0.0),
    ]:
        if col not in df.columns:
            df[col] = default

    return df