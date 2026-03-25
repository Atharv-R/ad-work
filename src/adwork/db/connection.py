# src/adwork/db/connection.py

"""
DuckDB connection management.
DuckDB is an embedded analytical database — think SQLite but for analytics.
No server needed, just a file.
"""

import duckdb
from loguru import logger

from adwork.config import settings

_connection: duckdb.DuckDBPyConnection | None = None


def get_db() -> duckdb.DuckDBPyConnection:
    """Get or create the DuckDB connection."""
    global _connection

    if _connection is None:
        db_path = str(settings.duckdb_full_path)
        _connection = duckdb.connect(db_path)
        logger.info(f"Connected to DuckDB at {db_path}")
        _initialize_tables(_connection)

    return _connection


def _initialize_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create tables if they don't exist."""

    conn.execute("""
        CREATE TABLE IF NOT EXISTS campaigns (
            campaign_id VARCHAR PRIMARY KEY,
            campaign_name VARCHAR NOT NULL,
            platform VARCHAR NOT NULL,
            status VARCHAR DEFAULT 'active',
            daily_budget DOUBLE DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_metrics (
            campaign_id VARCHAR NOT NULL,
            date DATE NOT NULL,
            impressions INTEGER DEFAULT 0,
            clicks INTEGER DEFAULT 0,
            conversions INTEGER DEFAULT 0,
            spend DOUBLE DEFAULT 0.0,
            revenue DOUBLE DEFAULT 0.0,
            ctr DOUBLE DEFAULT 0.0,
            cpc DOUBLE DEFAULT 0.0,
            roas DOUBLE DEFAULT 0.0,
            PRIMARY KEY (campaign_id, date)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER,
            campaign_id VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            action_type VARCHAR NOT NULL,
            action_detail VARCHAR NOT NULL,
            reasoning TEXT NOT NULL,
            confidence VARCHAR NOT NULL,
            status VARCHAR DEFAULT 'pending',
            llm_provider VARCHAR,
            llm_model VARCHAR
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            id INTEGER,
            campaign_id VARCHAR NOT NULL,
            forecast_date DATE NOT NULL,
            metric VARCHAR NOT NULL,
            predicted_value DOUBLE NOT NULL,
            lower_bound DOUBLE,
            upper_bound DOUBLE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS search_trends (
            keyword VARCHAR NOT NULL,
            date DATE NOT NULL,
            interest INTEGER NOT NULL,
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (keyword, date)
        )
    """)

    # ── inside _initialize_tables(conn) ──────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS competitor_ads (
            ad_id VARCHAR PRIMARY KEY,
            advertiser_name VARCHAR NOT NULL,
            platform VARCHAR NOT NULL,
            ad_copy TEXT NOT NULL,
            headline VARCHAR DEFAULT '',
            cta VARCHAR DEFAULT '',
            category VARCHAR DEFAULT '',
            first_seen DATE,
            last_seen DATE,
            is_active BOOLEAN DEFAULT TRUE,
            spend_tier VARCHAR DEFAULT 'medium',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS competitor_clusters (
            cluster_id INTEGER NOT NULL,
            cluster_label VARCHAR NOT NULL,
            top_terms TEXT NOT NULL,
            n_ads INTEGER NOT NULL,
            analysis_date DATE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (cluster_id, analysis_date)
        )
    """)

    logger.info("Database tables initialized")


def close_db() -> None:
    """Close the database connection."""
    global _connection
    if _connection:
        _connection.close()
        _connection = None
        logger.info("Database connection closed")


def reset_db() -> None:
    """Drop all tables and reinitialize. Used for testing/reseeding."""
    conn = get_db()
    for table in ["search_trends", "forecasts", "recommendations", "daily_metrics", "campaigns"]:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    _initialize_tables(conn)
    logger.info("Database reset complete")