"""Tests for competitor data + NLP pipeline."""

import pandas as pd
import pytest

# ── DB fixture (matches your existing test pattern) ─────────────────


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_comp.duckdb")
    monkeypatch.setenv("DUCKDB_PATH", db_path)

    from adwork.db import connection as conn_mod
    conn_mod.close_db()

    # Force settings to pick up new path
    from adwork import config
    monkeypatch.setattr(config.settings, "duckdb_path", db_path)

    yield

    conn_mod.close_db()


# ── Synthetic data ──────────────────────────────────────────────────


def test_generate_synthetic_ads():
    from adwork.data.competitors import generate_synthetic_ads

    ads = generate_synthetic_ads(seed=42)
    assert len(ads) == 50
    assert all(a.ad_copy.strip() for a in ads)
    assert all(a.ad_id.startswith("syn_") for a in ads)

    # Deterministic
    ads2 = generate_synthetic_ads(seed=42)
    assert [a.ad_id for a in ads] == [a.ad_id for a in ads2]


def test_seed_and_retrieve():
    from adwork.data.competitors import seed_competitor_data
    from adwork.db.queries import get_competitor_ads, has_competitor_data

    assert not has_competitor_data()
    n = seed_competitor_data()
    assert n == 50
    assert has_competitor_data()

    df = get_competitor_ads()
    assert len(df) == 50

    # Filter works
    google_ads = get_competitor_ads(platform="google")
    assert len(google_ads) > 0
    assert all(google_ads["platform"] == "google")


# ── CSV ingestion ───────────────────────────────────────────────────


def test_csv_ingestion():
    from adwork.data.competitors import ingest_competitor_csv
    from adwork.db.queries import get_competitor_ads

    df = pd.DataFrame({
        "brand": ["TestCo", "RivalInc"],
        "text": ["Save 50% on widgets today!", "Best widget on the market."],
        "title": ["Widget Sale", "Widget Review"],
    })
    # "brand" → advertiser_name, "text" → ad_copy, "title" → headline
    # (via _COL_ALIASES)

    ads = ingest_competitor_csv(df)
    assert len(ads) == 2
    assert ads[0].advertiser_name == "TestCo"

    stored = get_competitor_ads()
    assert len(stored) == 2


def test_csv_ingestion_missing_copy_raises():
    from adwork.data.competitors import ingest_competitor_csv

    df = pd.DataFrame({"brand": ["X"], "price": [9.99]})
    with pytest.raises(ValueError, match="ad copy column"):
        ingest_competitor_csv(df)


# ── NLP pipeline ────────────────────────────────────────────────────


def test_analyzer_clusters_synthetic():
    from adwork.data.competitors import generate_synthetic_ads
    from adwork.models.competitor_nlp import CompetitorAnalyzer

    ads = generate_synthetic_ads()
    ads_df = pd.DataFrame([a.model_dump() for a in ads])

    analyzer = CompetitorAnalyzer(n_clusters=5)
    result = analyzer.analyze(ads_df)

    # Structure
    assert "ads_df" in result
    assert "clusters" in result
    assert "strategy_matrix" in result
    assert len(result["clusters"]) == 5

    # Every ad got a cluster
    assert "cluster" in result["ads_df"].columns
    assert result["ads_df"]["cluster"].notna().all()

    # PCA coords exist
    assert "x" in result["ads_df"].columns
    assert "y" in result["ads_df"].columns

    # Each cluster has a label and top terms
    for c in result["clusters"]:
        assert c["label"]
        assert len(c["top_terms"]) > 0
        assert c["n_ads"] > 0

    # Strategy matrix is advertiser × cluster
    sm = result["strategy_matrix"]
    assert not sm.empty
    assert sm.sum().sum() == 50  # all ads accounted for


def test_analyzer_small_dataset():
    """Handles fewer docs than requested clusters."""
    from adwork.models.competitor_nlp import CompetitorAnalyzer

    df = pd.DataFrame({"ad_copy": [
        "Buy now and save 50 percent on all laptops",
        "Award winning noise cancellation headphones",
        "Limited edition only 100 units remaining",
    ]})
    analyzer = CompetitorAnalyzer(n_clusters=5)  # asks for 5, gets 3
    result = analyzer.analyze(df)
    assert len(result["clusters"]) == 3


def test_analyzer_rejects_tiny():
    from adwork.models.competitor_nlp import CompetitorAnalyzer

    df = pd.DataFrame({"ad_copy": ["one ad"]})
    with pytest.raises(ValueError, match="at least 3"):
        CompetitorAnalyzer().analyze(df)