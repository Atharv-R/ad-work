# tests/test_bandit.py

"""Test bandit algorithms and optimization modules."""

import pytest
import numpy as np


def test_beta_arm_basics():
    from adwork.models.bandit import BetaArm

    arm = BetaArm(alpha=1, beta=1)
    assert arm.mean == 0.5
    assert arm.n_observations == 0

    arm.update(1)  # success
    assert arm.alpha == 2
    assert arm.mean > 0.5

    arm.update(0)  # failure
    assert arm.beta == 2


def test_normal_arm_basics():
    from adwork.models.bandit import NormalArm

    arm = NormalArm(mu=0.0, sigma=1.0)
    assert arm.mean == 0.0

    arm.update(5.0, obs_sigma=1.0)
    assert arm.mean > 0.0  # Should shift toward 5.0
    assert arm.uncertainty < 1.0  # Should decrease


def test_thompson_sampling_selects():
    from adwork.models.bandit import ThompsonSampling

    bandit = ThompsonSampling(["a", "b", "c"], arm_type="normal")
    arm, samples = bandit.select_arm()

    assert arm in ["a", "b", "c"]
    assert len(samples) == 3
    assert all(isinstance(v, float) for v in samples.values())


def test_thompson_sampling_learns():
    """After many updates, the bandit should prefer the rewarded arm."""
    from adwork.models.bandit import ThompsonSampling

    bandit = ThompsonSampling(["good", "bad"], arm_type="normal")

    rng = np.random.default_rng(42)
    for _ in range(200):
        bandit.update("good", rng.normal(10.0, 1.0), obs_sigma=1.0)
        bandit.update("bad", rng.normal(2.0, 1.0), obs_sigma=1.0)

    assert bandit.best_arm() == "good"
    beliefs = bandit.get_beliefs()
    assert beliefs["good"]["mean"] > beliefs["bad"]["mean"]


def test_simulation_runs():
    from adwork.models.bandit import run_bandit_simulation

    results = run_bandit_simulation(
        base_metrics={
            "base_impressions": 5000,
            "base_ctr": 0.04,
            "base_cpc": 2.0,
            "base_conv_rate": 0.03,
            "base_aov": 200.0,
        },
        n_rounds=30,
        seed=42,
    )

    assert len(results["strategies"]) == 3
    assert "oracle" in results
    assert results["oracle"]["total_reward"] > 0

    for s in results["strategies"]:
        assert len(s.cumulative_regret) == 30
        assert s.total_reward != 0


def test_simulation_ts_beats_random():
    """Thompson Sampling should have lower regret than random ON AVERAGE across seeds."""
    from adwork.models.bandit import run_bandit_simulation

    base_metrics = {
        "base_impressions": 5000,
        "base_ctr": 0.04,
        "base_cpc": 2.0,
        "base_conv_rate": 0.03,
        "base_aov": 200.0,
    }

    ts_wins = 0
    n_trials = 10

    for seed in range(n_trials):
        results = run_bandit_simulation(
            base_metrics=base_metrics,
            n_rounds=90,
            seed=seed * 17,  # Spread seeds out
        )

        ts = next(s for s in results["strategies"] if "Thompson" in s.strategy)
        rand = next(s for s in results["strategies"] if "Random" in s.strategy)

        if ts.cumulative_regret[-1] <= rand.cumulative_regret[-1]:
            ts_wins += 1

    # Thompson Sampling should beat random in the majority of trials
    assert ts_wins >= 5, (
        f"Thompson Sampling only beat Random in {ts_wins}/{n_trials} trials "
        f"(expected at least 5)"
    )


def test_bid_recommender():
    """Bid recommender should produce recommendations."""
    import pandas as pd
    from datetime import date, timedelta

    from adwork.optimization.bid_recommender import BidRecommender

    # Build minimal campaign + metrics data
    campaigns = pd.DataFrame({
        "campaign_id": ["c1"],
        "campaign_name": ["Test Campaign"],
        "platform": ["google"],
    })

    rng = np.random.default_rng(42)
    dates = [date(2025, 4, 1) + timedelta(days=i) for i in range(60)]
    metrics = pd.DataFrame({
        "campaign_id": ["c1"] * 60,
        "date": dates,
        "impressions": rng.integers(3000, 8000, 60),
        "clicks": rng.integers(100, 400, 60),
        "conversions": rng.integers(5, 30, 60),
        "spend": np.round(rng.uniform(100, 500, 60), 2),
        "revenue": np.round(rng.uniform(200, 1500, 60), 2),
        "roas": np.round(rng.uniform(1.0, 5.0, 60), 2),
    })

    recommender = BidRecommender()
    recs = recommender.generate_recommendations(campaigns, metrics)

    assert len(recs) == 1
    rec = recs[0]
    assert rec.campaign_id == "c1"
    assert rec.recommended_action in ("increase", "decrease", "hold")
    assert rec.confidence in ("high", "medium", "low")
    assert len(rec.reasoning) > 0


def test_budget_allocator():
    """Budget allocator should produce allocations."""
    import pandas as pd
    from adwork.optimization.budget_allocator import BudgetAllocator

    summary = pd.DataFrame({
        "campaign_name": ["Camp A", "Camp B", "Camp C"],
        "campaign_id": ["a", "b", "c"],
        "spend": [5000, 3000, 2000],
        "revenue": [20000, 6000, 3000],
        "roas": [4.0, 2.0, 1.5],
    })

    allocator = BudgetAllocator()
    result = allocator.allocate(summary, total_daily_budget=500)

    assert "current_allocation" in result
    assert "recommended_allocation" in result
    assert len(result["current_allocation"]) == 3
    assert len(result["recommended_allocation"]) == 3

    # Total should roughly equal the budget
    total_rec = sum(result["recommended_allocation"].values())
    assert abs(total_rec - 500) < 5  # Allow small rounding error