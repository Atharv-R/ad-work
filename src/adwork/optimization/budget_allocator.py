# src/adwork/optimization/budget_allocator.py

"""
Cross-Channel Budget Allocator
===============================
Uses Thompson Sampling to allocate budget across campaigns
proportionally to their sampled ROAS, naturally balancing
exploitation (fund high-ROAS campaigns) and exploration
(give uncertain campaigns a chance).
"""

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel

from adwork.models.bandit import ThompsonSampling


class AllocationRecommendation(BaseModel):
    """One budget shift recommendation."""
    from_campaign: str
    from_campaign_name: str
    to_campaign: str
    to_campaign_name: str
    amount: float
    reasoning: str
    confidence: str


class BudgetAllocator:
    """Allocate budget across campaigns using Thompson Sampling."""

    def allocate(
        self,
        campaign_summary: pd.DataFrame,
        total_daily_budget: float | None = None,
        n_samples: int = 1000,
    ) -> dict:
        """
        Recommend budget allocation across campaigns.
        
        Args:
            campaign_summary: Per-campaign aggregated metrics
                              (needs: campaign_name, spend, revenue, roas)
            total_daily_budget: Total budget constraint. If None, uses sum of current spend.
            n_samples: Number of Thompson Sampling draws for robust allocation
            
        Returns:
            Dict with current_allocation, recommended_allocation, shifts
        """
        if campaign_summary.empty:
            return {"current": {}, "recommended": {}, "shifts": []}

        # Current allocation (proportion of total spend)
        total_spend = campaign_summary["spend"].sum()
        if total_daily_budget is None:
            total_daily_budget = total_spend / 90  # Rough daily average from 90-day totals

        campaigns = campaign_summary.to_dict("records")

        # Build a bandit with one arm per campaign, seeded with ROAS data
        arm_names = [c["campaign_name"] for c in campaigns]
        bandit = ThompsonSampling(arm_names, arm_type="normal")

        # Seed beliefs using campaign ROAS data
        rng = np.random.default_rng(42)
        for c in campaigns:
            roas = c.get("roas", 1.0)
            if isinstance(roas, str):
                roas = float(roas.replace("x", ""))
            # Feed multiple observations to build belief
            for _ in range(30):
                noisy_roas = roas * rng.normal(1.0, 0.15)
                bandit.update(c["campaign_name"], noisy_roas, obs_sigma=roas * 0.2 + 0.1)

        # Sample allocations many times and average
        allocation_totals = {name: 0.0 for name in arm_names}

        for _ in range(n_samples):
            _, samples = bandit.select_arm()
            # Shift samples to be positive for proportional allocation
            min_sample = min(samples.values())
            shifted = {k: max(v - min_sample + 0.1, 0.1) for k, v in samples.items()}
            total_sampled = sum(shifted.values())

            for name, val in shifted.items():
                allocation_totals[name] += (val / total_sampled)

        # Average allocation proportions
        recommended_props = {
            name: round(total / n_samples, 4)
            for name, total in allocation_totals.items()
        }

        # Current proportions
        current_props = {}
        for c in campaigns:
            prop = c["spend"] / total_spend if total_spend > 0 else 1.0 / len(campaigns)
            current_props[c["campaign_name"]] = round(prop, 4)

        # Convert to dollar amounts
        current_alloc = {k: round(v * total_daily_budget, 2) for k, v in current_props.items()}
        recommended_alloc = {k: round(v * total_daily_budget, 2) for k, v in recommended_props.items()}

        # Generate shift recommendations (only material shifts > 5%)
        shifts = []
        changes = {k: recommended_alloc[k] - current_alloc.get(k, 0) for k in recommended_alloc}
        increases = sorted([(k, v) for k, v in changes.items() if v > total_daily_budget * 0.03],
                           key=lambda x: x[1], reverse=True)
        decreases = sorted([(k, v) for k, v in changes.items() if v < -total_daily_budget * 0.03],
                           key=lambda x: x[1])

        # Build campaign name → id lookup
        name_to_id = {c["campaign_name"]: c.get("campaign_id", c["campaign_name"]) for c in campaigns}
        name_to_roas = {}
        for c in campaigns:
            r = c.get("roas", 0)
            name_to_roas[c["campaign_name"]] = float(str(r).replace("x", "")) if r else 0

        i, j = 0, 0
        while i < len(increases) and j < len(decreases):
            to_name, to_amt = increases[i]
            from_name, from_amt = decreases[j]
            shift_amount = round(min(to_amt, abs(from_amt)), 2)

            if shift_amount >= total_daily_budget * 0.02:
                from_roas = name_to_roas.get(from_name, 0)
                to_roas = name_to_roas.get(to_name, 0)

                shifts.append(AllocationRecommendation(
                    from_campaign=name_to_id.get(from_name, from_name),
                    from_campaign_name=from_name,
                    to_campaign=name_to_id.get(to_name, to_name),
                    to_campaign_name=to_name,
                    amount=shift_amount,
                    reasoning=(
                        f"Shift ${shift_amount:.0f}/day from {from_name} (ROAS {from_roas:.1f}x) "
                        f"to {to_name} (ROAS {to_roas:.1f}x). "
                        f"Thompson Sampling posterior indicates higher expected returns."
                    ),
                    confidence="high" if abs(to_roas - from_roas) > 1.0 else "medium",
                ))

            i += 1
            j += 1

        logger.info(f"Budget allocation: {len(shifts)} shift recommendations")

        return {
            "current_allocation": current_alloc,
            "recommended_allocation": recommended_alloc,
            "shifts": shifts,
            "total_daily_budget": total_daily_budget,
            "beliefs": bandit.get_beliefs(),
        }