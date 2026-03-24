# tests/test_ingestion.py

"""Test data ingestion pipeline."""

import pandas as pd
import pytest
from datetime import date


@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    """Use a temporary database for each test."""
    import os
    os.environ["DUCKDB_PATH"] = str(tmp_path / "test.duckdb")

    # Reset the cached settings and db connection
    from adwork.db.connection import close_db
    close_db()

    # Force settings reload
    from adwork.config import Settings
    import adwork.config
    adwork.config.settings = Settings(duckdb_path=str(tmp_path / "test.duckdb"))

    yield

    close_db()


def test_ingest_internal_format():
    from adwork.data.ingestion import ingest_csv

    df = pd.DataFrame({
        "campaign_id": ["c1", "c1", "c2"],
        "campaign_name": ["Camp A", "Camp A", "Camp B"],
        "platform": ["google", "google", "meta"],
        "date": ["2025-06-01", "2025-06-02", "2025-06-01"],
        "impressions": [1000, 1200, 5000],
        "clicks": [50, 60, 40],
        "conversions": [5, 6, 3],
        "spend": [25.0, 30.0, 20.0],
        "revenue": [100.0, 120.0, 60.0],
    })

    result = ingest_csv(df)

    assert result["status"] == "success"
    assert result["rows_loaded"] == 3
    assert len(result["campaigns_found"]) == 2


def test_ingest_with_bad_rows():
    from adwork.data.ingestion import ingest_csv

    df = pd.DataFrame({
        "campaign_id": ["c1", "c1"],
        "campaign_name": ["Camp A", "Camp A"],
        "platform": ["google", "google"],
        "date": ["2025-06-01", "not-a-date"],
        "impressions": [1000, 500],
        "clicks": [50, 25],
        "conversions": [5, 2],
        "spend": [25.0, 12.0],
        "revenue": [100.0, 50.0],
    })

    result = ingest_csv(df)

    # First row should succeed, second should fail due to bad date
    assert result["rows_loaded"] >= 1