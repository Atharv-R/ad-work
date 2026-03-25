# src/adwork/optimization/bid_recommender.py

"""
Bid Recommendation Engine
=========================
Uses Thompson Sampling beliefs + forecasts + historical data
to generate per-campaign bid adjustment recommendations.

Each recommendation includes:
- Action: increase/decrease/hold bid by X%
- Confidence: high/medium/low (based on bandit uncertainty)
- Reasoning: plain-English explanation of why
"""

import json
import numpy as np
import pandas as pd
from datetime import date
from loguru import logger
from pydantic import BaseModel, Field

from adwork.models.bandit import ThompsonSampling, BidResponseModel


class BidRecommendation(BaseModel):
    """One bid recommendation for one campaign."""
    campaign_id: str
    campaign_name: str
    platform: str
    current_roas: float
    recommended_action: str          # "increase", "decrease", "hold"
    recommended_change_pct: float    # e.g., 12.0 for +12%
    recommended_multiplier: float    # e.g., 1.12
    expected_roas_after: float
    confidence: str                  # "high", "medium", "low"
    reasoning: str


# ROAS target per platform (configurable)
ROAS_TARGETS = {
    "google": 3.0,
    "meta": 2.5,
    "amazon": 3.5,
}

BID_LEVELS = [0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20]


class BidRecommender:
    """Generate bid recommendations for all campaigns."""

    def __init__(self, roas_targets: dict | None = None):
        self.roas_targets = roas_targets or ROAS_TARGETS

    def generate_recommendations(
        self,
        campaigns_df: pd.DataFrame,
        metrics_df: pd.DataFrame,
        forecasts_df: pd.DataFrame | None = None,
    ) -> list[BidRecommendation]:
        """
        Generate bid recommendations for all campaigns.
        
        Args:
            campaigns_df: All campaigns from DB
            metrics_df: Daily metrics (needs last 30+ days per campaign)
            forecasts_df: Optional forecast data from Phase 2
            
        Returns:
            List of BidRecommendation objects
        """
        recommendations = []

        for _, campaign in campaigns_df.iterrows():
            cid = campaign["campaign_id"]
            cname = campaign["campaign_name"]
            platform = campaign["platform"]

            # Get this campaign's recent metrics
            camp_metrics = metrics_df[metrics_df["campaign_id"] == cid].copy()
            if len(camp_metrics) < 14:
                logger.warning(f"Skipping {cname}: only {len(camp_metrics)} days of data")
                continue

            camp_metrics = camp_metrics.sort_values("date")

            rec = self._recommend_single(
                campaign_id=cid,
                campaign_name=cname,
                platform=platform,
                metrics=camp_metrics,
                forecasts_df=forecasts_df,
            )

            if rec:
                recommendations.append(rec)

        logger.info(f"Generated {len(recommendations)} bid recommendations")
        return recommendations

    def _recommend_single(
        self,
        campaign_id: str,
        campaign_name: str,
        platform: str,
        metrics: pd.DataFrame,
        forecasts_df: pd.DataFrame | None,
    ) -> BidRecommendation | None:
        """Generate a recommendation for one campaign using Thompson Sampling."""

        # Compute base metrics from recent 30 days
        recent = metrics.tail(30)
        total_impr = recent["impressions"].sum()
        total_clicks = recent["clicks"].sum()
        total_conv = recent["conversions"].sum()
        total_spend = recent["spend"].sum()
        total_revenue = recent["revenue"].sum()

        if total_spend == 0 or total_clicks == 0:
            return None

        base_ctr = total_clicks / total_impr if total_impr > 0 else 0.01
        base_cpc = total_spend / total_clicks
        base_conv_rate = total_conv / total_clicks if total_clicks > 0 else 0.01
        base_aov = total_revenue / total_conv if total_conv > 0 else 100.0
        base_impressions = total_impr / len(recent)
        current_roas = total_revenue / total_spend

        # Run Thompson Sampling simulation with this campaign's metrics
        arm_names = [str(b) for b in BID_LEVELS]
        bandit = ThompsonSampling(arm_names, arm_type="normal")
        response_model = BidResponseModel()
        rng = np.random.default_rng(hash(campaign_id) % 2**31)

        # Feed historical days through the bandit to build beliefs
        for _, day_row in recent.iterrows():
            for bid_level in BID_LEVELS:
                outcome = response_model.simulate(
                    base_impressions=base_impressions,
                    base_ctr=base_ctr,
                    base_cpc=base_cpc,
                    base_conv_rate=base_conv_rate,
                    base_aov=base_aov,
                    bid_multiplier=bid_level,
                    rng=rng,
                )
                bandit.update(
                    str(bid_level),
                    outcome["profit"],
                    obs_sigma=max(abs(outcome["profit"]) * 0.3, 1.0),
                )

        # Get the bandit's recommendation
        beliefs = bandit.get_beliefs()
        best_arm = bandit.best_arm()
        best_multiplier = float(best_arm)

        # Compute expected ROAS at recommended bid
        expected_outcomes = []
        for _ in range(100):
            out = response_model.simulate(
                base_impressions, base_ctr, base_cpc,
                base_conv_rate, base_aov, best_multiplier, rng,
            )
            if out["spend"] > 0:
                expected_outcomes.append(out["roas"])
        expected_roas = float(np.mean(expected_outcomes)) if expected_outcomes else current_roas

        # Determine action
        change_pct = round((best_multiplier - 1.0) * 100, 1)
        if abs(change_pct) < 3:
            action = "hold"
            change_pct = 0.0
            best_multiplier = 1.0
        elif change_pct > 0:
            action = "increase"
        else:
            action = "decrease"

        # Determine confidence from bandit uncertainty
        best_belief = beliefs[best_arm]
        uncertainty = best_belief["uncertainty"]

        # Compare best arm vs second best — larger gap = higher confidence
        sorted_beliefs = sorted(beliefs.items(), key=lambda x: x[1]["mean"], reverse=True)
        if len(sorted_beliefs) >= 2:
            gap = sorted_beliefs[0][1]["mean"] - sorted_beliefs[1][1]["mean"]
            relative_gap = gap / (abs(sorted_beliefs[0][1]["mean"]) + 1e-10)
        else:
            relative_gap = 0

        if relative_gap > 0.15 and best_belief["n_obs"] >= 20:
            confidence = "high"
        elif relative_gap > 0.05 and best_belief["n_obs"] >= 10:
            confidence = "medium"
        else:
            confidence = "low"

        # Build reasoning
        roas_target = self.roas_targets.get(platform, 3.0)
        reasoning = self._build_reasoning(
            campaign_name, platform, current_roas, roas_target,
            action, change_pct, expected_roas, confidence, metrics,
            forecasts_df, campaign_id,
        )

        return BidRecommendation(
            campaign_id=campaign_id,
            campaign_name=campaign_name,
            platform=platform,
            current_roas=round(current_roas, 2),
            recommended_action=action,
            recommended_change_pct=change_pct,
            recommended_multiplier=best_multiplier,
            expected_roas_after=round(expected_roas, 2),
            confidence=confidence,
            reasoning=reasoning,
        )

    def _build_reasoning(
        self, name, platform, current_roas, target, action,
        change_pct, expected_roas, confidence, metrics,
        forecasts_df, campaign_id,
    ) -> str:
        """Build plain-English reasoning for the recommendation."""
        parts = []

        # Current performance vs target
        if current_roas >= target:
            parts.append(
                f"Current ROAS is {current_roas:.1f}x (above {target:.1f}x target)."
            )
        else:
            parts.append(
                f"Current ROAS is {current_roas:.1f}x (below {target:.1f}x target)."
            )

        # Trend analysis (last 14 days vs prior 14 days)
        if len(metrics) >= 28:
            recent_roas = metrics.tail(14)["roas"].mean()
            prior_roas = metrics.iloc[-28:-14]["roas"].mean()
            if prior_roas > 0:
                trend_pct = ((recent_roas - prior_roas) / prior_roas) * 100
                direction = "improving" if trend_pct > 2 else "declining" if trend_pct < -2 else "stable"
                parts.append(f"Performance is {direction} ({trend_pct:+.1f}% ROAS change over 14 days).")

        # Forecast signal
        if forecasts_df is not None and not forecasts_df.empty:
            camp_fc = forecasts_df[forecasts_df["campaign_id"] == campaign_id]
            if not camp_fc.empty:
                parts.append("Demand forecast factored into optimization.")

        # Action reasoning
        if action == "increase":
            parts.append(
                f"Thompson Sampling recommends increasing bid by {change_pct:.0f}% "
                f"(expected ROAS: {expected_roas:.1f}x). "
                f"The model has {confidence} confidence based on posterior convergence."
            )
        elif action == "decrease":
            parts.append(
                f"Thompson Sampling recommends decreasing bid by {abs(change_pct):.0f}% "
                f"to improve efficiency (expected ROAS: {expected_roas:.1f}x)."
            )
        else:
            parts.append(
                "Current bid level is near-optimal. No adjustment recommended."
            )

        return " ".join(parts)