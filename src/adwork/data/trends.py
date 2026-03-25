# src/adwork/data/trends.py

"""
Google Trends data fetcher.

Pulls search interest data for product keywords.
Used in Phase 2 as features for demand forecasting.
For now, we display it on the dashboard as market context.
"""


import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from adwork.db.connection import get_db

# Default keywords matching our sample campaign product categories
DEFAULT_KEYWORDS = [
    "laptop",
    "headphones",
    "computer monitor",
    "wireless earbuds",
    "gaming keyboard",
]


@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def _fetch_from_google(keywords: list[str], timeframe: str = "today 3-m") -> pd.DataFrame:
    """
    Fetch interest-over-time data from Google Trends.
    
    Args:
        keywords: Up to 5 keywords to compare
        timeframe: Google Trends timeframe string
                   "today 3-m" = last 3 months
                   "today 12-m" = last 12 months
    
    Returns:
        DataFrame with columns: date, keyword, interest
    """
    from pytrends.request import TrendReq

    # pytrends setup — hl=en-US for English, tz=360 for US Central
    pytrends = TrendReq(hl="en-US", tz=360)

    # Google Trends allows max 5 keywords per request
    keywords = keywords[:5]

    logger.info(f"Fetching Google Trends for: {keywords}")
    pytrends.build_payload(keywords, timeframe=timeframe, geo="US")

    # Get interest over time
    interest_df = pytrends.interest_over_time()

    if interest_df.empty:
        logger.warning("Google Trends returned empty data")
        return pd.DataFrame(columns=["date", "keyword", "interest"])

    # Drop the 'isPartial' column if present
    if "isPartial" in interest_df.columns:
        interest_df = interest_df.drop(columns=["isPartial"])

    # Melt from wide format (one column per keyword) to long format
    interest_df = interest_df.reset_index()
    melted = interest_df.melt(
        id_vars=["date"],
        var_name="keyword",
        value_name="interest",
    )

    melted["date"] = pd.to_datetime(melted["date"]).dt.date
    melted["interest"] = melted["interest"].astype(int)

    logger.info(f"Fetched {len(melted)} trend data points")
    return melted


def fetch_and_store_trends(
    keywords: list[str] | None = None,
    timeframe: str = "today 3-m",
) -> int:
    """
    Fetch Google Trends data and store in DuckDB.
    
    Args:
        keywords: Keywords to fetch (defaults to DEFAULT_KEYWORDS)
        timeframe: Google Trends timeframe
        
    Returns:
        Number of rows stored
    """
    if keywords is None:
        keywords = DEFAULT_KEYWORDS

    try:
        trends_df = _fetch_from_google(keywords, timeframe)
    except Exception as e:
        logger.error(f"Failed to fetch Google Trends: {e}")
        return 0

    if trends_df.empty:
        return 0

    conn = get_db()
    rows_stored = 0

    for _, row in trends_df.iterrows():
        try:
            conn.execute("""
                INSERT OR REPLACE INTO search_trends (keyword, date, interest)
                VALUES (?, ?, ?)
            """, [row["keyword"], row["date"], int(row["interest"])])
            rows_stored += 1
        except Exception as e:
            logger.warning(f"Failed to store trend: {e}")

    logger.info(f"Stored {rows_stored} trend data points")
    return rows_stored