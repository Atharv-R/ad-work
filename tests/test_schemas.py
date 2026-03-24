# tests/test_schemas.py

"""Test data schemas and validation."""

import pytest
from datetime import date


def test_daily_metrics_computed_fields():
    from adwork.data.schemas import DailyMetrics

    m = DailyMetrics(
        campaign_id="test_001",
        date=date(2025, 6, 1),
        impressions=10000,
        clicks=500,
        conversions=25,
        spend=250.00,
        revenue=1000.00,
    )

    assert m.ctr == pytest.approx(0.05, abs=0.001)
    assert m.cpc == pytest.approx(0.50, abs=0.01)
    assert m.roas == pytest.approx(4.0, abs=0.01)
    assert m.conversion_rate == pytest.approx(0.05, abs=0.001)


def test_daily_metrics_zero_division():
    from adwork.data.schemas import DailyMetrics

    m = DailyMetrics(
        campaign_id="test_002",
        date=date(2025, 6, 1),
        impressions=0,
        clicks=0,
        spend=0.0,
        revenue=0.0,
    )

    assert m.ctr == 0.0
    assert m.cpc == 0.0
    assert m.roas == 0.0


def test_campaign_validation():
    from adwork.data.schemas import Campaign, Platform

    c = Campaign(
        campaign_id="ggl_001",
        campaign_name="Test Campaign",
        platform=Platform.GOOGLE,
        daily_budget=100.0,
    )

    assert c.platform == Platform.GOOGLE


def test_campaign_empty_id_rejected():
    from adwork.data.schemas import Campaign, Platform

    with pytest.raises(Exception):
        Campaign(
            campaign_id="   ",
            campaign_name="Bad Campaign",
            platform=Platform.GOOGLE,
            daily_budget=100.0,
        )


def test_detect_platform():
    from adwork.data.schemas import detect_platform_from_columns, Platform

    # Google-style columns
    assert detect_platform_from_columns(
        ["Campaign", "Day", "Impressions", "Clicks", "Cost", "Conv. value", "Avg. CPC"]
    ) == Platform.GOOGLE

    # Meta-style columns
    assert detect_platform_from_columns(
        ["Campaign name", "Day", "Impressions", "Link clicks", "Amount spent (USD)"]
    ) == Platform.META

    # Internal format
    assert detect_platform_from_columns(
        ["campaign_id", "platform", "date", "impressions", "clicks"]
    ) == Platform.UNKNOWN  # Internal format detected