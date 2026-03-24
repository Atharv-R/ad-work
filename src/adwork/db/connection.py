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
            platform VARCHAR NOT NULL,        -- 'google', 'meta', 'amazon'
            status VARCHAR DEFAULT 'active',
            daily_budget DOUBLE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_metrics (
            id INTEGER PRIMARY KEY,
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
            UNIQUE(campaign_id, date)
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY,
            campaign_id VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            action_type VARCHAR NOT NULL,     -- 'bid_adjustment', 'budget_reallocation', 'creative_rotation'
            action_detail VARCHAR NOT NULL,    -- JSON with specific action
            reasoning TEXT NOT NULL,           -- Plain English explanation
            confidence VARCHAR NOT NULL,       -- 'high', 'medium', 'low'
            status VARCHAR DEFAULT 'pending',  -- 'pending', 'applied', 'dismissed'
            llm_provider VARCHAR,
            llm_model VARCHAR
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            id INTEGER PRIMARY KEY,
            campaign_id VARCHAR NOT NULL,
            forecast_date DATE NOT NULL,
            metric VARCHAR NOT NULL,           -- 'clicks', 'conversions', 'spend'
            predicted_value DOUBLE NOT NULL,
            lower_bound DOUBLE,
            upper_bound DOUBLE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create sequence for auto-incrementing IDs
    conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_daily_metrics START 1")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_recommendations START 1")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_forecasts START 1")
    
    logger.info("Database tables initialized")


def close_db() -> None:
    """Close the database connection."""
    global _connection
    if _connection:
        _connection.close()
        _connection = None
        logger.info("Database connection closed")