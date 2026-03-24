# tests/test_forecaster.py

"""Test the demand forecaster."""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta


def _make_sample_data(days=90):
    """Create realistic sample data for testing."""
    rng = np.random.default_rng(42)
    dates = [date(2025, 4, 1) + timedelta(days=i) for i in range(days)]

    base = 100
    clicks = []
    for i, d in enumerate(dates):
        dow_effect = {0: 1.0, 1: 1.05, 2: 1.05, 3: 1.1, 4: 1.15, 5: 0.9, 6: 0.85}
        val = base * dow_effect[d.weekday()] * rng.normal(1.0, 0.08)
        val = val * (1 + 0.001 * i)  # slight trend
        clicks.append(max(1, int(val)))

    return pd.DataFrame({
        "date": dates,
        "clicks": clicks,
        "conversions": [max(0, int(c * 0.05 * rng.normal(1, 0.2))) for c in clicks],
        "spend": [round(c * 1.5 * rng.normal(1, 0.1), 2) for c in clicks],
        "revenue": [round(c * 0.05 * 50 * rng.normal(1, 0.15), 2) for c in clicks],
    })


def test_forecaster_runs():
    """Verify the forecaster produces output without crashing."""
    from adwork.models.forecaster import DemandForecaster

    df = _make_sample_data(90)
    forecaster = DemandForecaster()

    result = forecaster.run(
        historical_df=df,
        metric="clicks",
        campaign_id="test_001",
    )

    assert result["engine"] in ("prophet", "statsmodels")
    assert "forecast" in result
    assert "backtest" in result
    assert "evaluation" in result

    # Forecast should have future rows
    fwd = result["forecast"][result["forecast"]["is_forecast"]]
    assert len(fwd) == 14  # default horizon

    # Backtest should have rows
    assert len(result["backtest"]) > 0

    # Evaluation should have all metrics
    ev = result["evaluation"]
    assert "mape" in ev
    assert "mae" in ev
    assert "rmse" in ev
    assert "coverage" in ev


def test_forecaster_evaluation_reasonable():
    """MAPE should be reasonable on clean data with weekly pattern."""
    from adwork.models.forecaster import DemandForecaster

    df = _make_sample_data(90)
    forecaster = DemandForecaster()

    result = forecaster.run(
        historical_df=df,
        metric="clicks",
        campaign_id="test_002",
    )

    # MAPE should be under 50% on this clean synthetic data
    assert result["evaluation"]["mape"] < 50, (
        f"MAPE too high: {result['evaluation']['mape']}%"
    )

    # Coverage should be at least 50%
    assert result["evaluation"]["coverage"] >= 0.5, (
        f"Coverage too low: {result['evaluation']['coverage']}"
    )


def test_forecaster_rejects_short_data():
    """Should raise error with less than 30 days."""
    from adwork.models.forecaster import DemandForecaster

    df = _make_sample_data(20)  # Too short
    forecaster = DemandForecaster()

    with pytest.raises(ValueError, match="at least 30 days"):
        forecaster.run(historical_df=df, metric="clicks", campaign_id="test_003")


def test_forecaster_with_trends():
    """Verify trends regressor doesn't crash the pipeline."""
    from adwork.models.forecaster import DemandForecaster

    df = _make_sample_data(90)

    # Create fake trends data
    trends = pd.DataFrame({
        "date": df["date"],
        "keyword": "laptop",
        "interest": np.random.randint(30, 80, size=len(df)),
    })

    forecaster = DemandForecaster()
    result = forecaster.run(
        historical_df=df,
        metric="clicks",
        campaign_id="test_004",
        trends_df=trends,
    )

    assert result["forecast"] is not None
    assert len(result["backtest"]) > 0