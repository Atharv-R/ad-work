# src/adwork/models/bandit.py

"""
Multi-Armed Bandit Algorithms
==============================
Thompson Sampling for ad bid optimization.

Two variants:
- BetaArm: For binary outcomes (click/no-click → CTR optimization)
- NormalArm: For continuous outcomes (profit/ROAS optimization)

Thompson Sampling works by:
1. Maintaining a probability distribution (belief) about each arm's reward
2. Sampling from each belief
3. Picking the arm with the highest sample
4. Observing the reward and updating the belief

This naturally balances exploration (trying uncertain arms) and
exploitation (favoring arms that look good) — arms with high
uncertainty get sampled high sometimes, pulling them into selection.

The key advantage over epsilon-greedy: Thompson Sampling explores
*intelligently* — it explores arms proportionally to their probability
of being optimal, not uniformly.
"""

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field


# ─── Arm Beliefs ──────────────────────────────────────

class BetaArm:
    """
    Beta-Bernoulli arm for binary outcomes.
    
    Prior: Beta(alpha, beta)
    Update: observe success → alpha += 1, failure → beta += 1
    Posterior mean: alpha / (alpha + beta) = estimated CTR
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta

    def sample(self, rng: np.random.Generator) -> float:
        return rng.beta(self.alpha, self.beta)

    def update(self, reward: int):
        if reward:
            self.alpha += 1
        else:
            self.beta += 1

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @property
    def n_observations(self) -> int:
        return int(self.alpha + self.beta - 2)  # Subtract prior counts


class NormalArm:
    """
    Normal-Normal arm for continuous outcomes (profit, ROAS).
    
    Bayesian update with known observation noise sigma_obs:
    - Prior: N(mu_0, sigma_0^2)
    - After n observations with mean x_bar:
    - Posterior precision = prior_precision + n * obs_precision
    - Posterior mean = weighted combination of prior mean and data mean
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        self.mu = mu
        self.sigma = sigma
        self.n = 0
        self._obs_sum = 0.0

    def sample(self, rng: np.random.Generator) -> float:
        return rng.normal(self.mu, self.sigma)

    def update(self, reward: float, obs_sigma: float = 1.0):
        self.n += 1
        self._obs_sum += reward

        prior_prec = 1.0 / (self.sigma ** 2 + 1e-10)
        obs_prec = 1.0 / (obs_sigma ** 2 + 1e-10)

        post_prec = prior_prec + obs_prec
        self.mu = (prior_prec * self.mu + obs_prec * reward) / post_prec
        self.sigma = 1.0 / np.sqrt(post_prec)

    @property
    def mean(self) -> float:
        return self.mu

    @property
    def uncertainty(self) -> float:
        return self.sigma


# ─── Thompson Sampling Bandit ─────────────────────────

class ThompsonSampling:
    """
    Thompson Sampling multi-armed bandit.
    
    Args:
        arm_names: List of arm identifiers (e.g., bid multiplier levels)
        arm_type: 'beta' for binary, 'normal' for continuous rewards
        prior_params: Dict of prior parameters per arm (optional)
    """

    def __init__(
        self,
        arm_names: list[str],
        arm_type: str = "normal",
        prior_params: dict | None = None,
    ):
        self.arm_names = arm_names
        self.arm_type = arm_type
        self.rng = np.random.default_rng()

        prior_params = prior_params or {}

        if arm_type == "beta":
            self.arms = {
                name: BetaArm(**prior_params.get(name, {}))
                for name in arm_names
            }
        elif arm_type == "normal":
            self.arms = {
                name: NormalArm(**prior_params.get(name, {}))
                for name in arm_names
            }
        else:
            raise ValueError(f"Unknown arm type: {arm_type}")

    def select_arm(self) -> tuple[str, dict[str, float]]:
        """
        Sample from each arm's belief distribution, pick the highest.
        
        Returns:
            (selected_arm_name, {arm_name: sampled_value})
        """
        samples = {name: arm.sample(self.rng) for name, arm in self.arms.items()}
        best = max(samples, key=samples.get)
        return best, samples

    def update(self, arm_name: str, reward: float, **kwargs):
        """Update the selected arm's belief with observed reward."""
        self.arms[arm_name].update(reward, **kwargs)

    def get_beliefs(self) -> dict:
        """Return current belief summary for all arms."""
        beliefs = {}
        for name, arm in self.arms.items():
            beliefs[name] = {
                "mean": round(arm.mean, 4),
                "uncertainty": round(
                    arm.variance ** 0.5 if hasattr(arm, "variance") else arm.uncertainty, 4
                ),
                "n_obs": arm.n_observations if hasattr(arm, "n_observations") else arm.n,
            }
        return beliefs

    def best_arm(self) -> str:
        """Return the arm with the highest posterior mean (exploitation only)."""
        return max(self.arms, key=lambda name: self.arms[name].mean)


# ─── Bid Response Model ──────────────────────────────

class BidResponseModel:
    """
    Models how bid changes affect campaign outcomes.
    
    Assumptions (standard in ad auction theory):
    - Impressions scale with bid^0.5 (diminishing returns)
    - CTR stays constant (bid doesn't affect ad quality)
    - CPC scales linearly with bid
    - Conversion rate stays constant
    
    Result: profit = revenue - spend has a concave shape
    with a single optimal bid level.
    """

    @staticmethod
    def simulate(
        base_impressions: float,
        base_ctr: float,
        base_cpc: float,
        base_conv_rate: float,
        base_aov: float,
        bid_multiplier: float,
        rng: np.random.Generator,
    ) -> dict:
        imp_mult = bid_multiplier ** 0.5  # Diminishing returns on volume
        impressions = max(0, int(base_impressions * imp_mult * rng.normal(1.0, 0.10)))
        clicks = max(0, int(impressions * base_ctr * rng.normal(1.0, 0.05)))
        cpc = max(0.01, base_cpc * bid_multiplier * rng.normal(1.0, 0.08))
        spend = round(clicks * cpc, 2)
        conversions = max(0, int(clicks * base_conv_rate * rng.normal(1.0, 0.12)))
        revenue = round(conversions * base_aov * rng.normal(1.0, 0.05), 2)
        profit = round(revenue - spend, 2)
        roas = round(revenue / spend, 4) if spend > 0 else 0.0

        return {
            "impressions": impressions,
            "clicks": clicks,
            "spend": spend,
            "conversions": conversions,
            "revenue": revenue,
            "profit": profit,
            "roas": roas,
        }


# ─── Simulation Runner ────────────────────────────────

class SimulationResult(BaseModel):
    """Results from one strategy in a simulation."""
    strategy: str
    total_reward: float
    cumulative_rewards: list[float]
    cumulative_regret: list[float]
    arm_counts: dict[str, int]


def run_bandit_simulation(
    base_metrics: dict,
    bid_levels: list[float] | None = None,
    n_rounds: int = 90,
    seed: int = 42,
) -> dict:
    """
    Run a multi-armed bandit simulation comparing strategies.
    
    Simulates n_rounds of bidding decisions on a single campaign,
    comparing Thompson Sampling, Epsilon-Greedy, and Uniform Random.
    
    Args:
        base_metrics: Dict with base_impressions, base_ctr, base_cpc,
                      base_conv_rate, base_aov
        bid_levels: List of bid multipliers to try (arms)
        n_rounds: Number of rounds (days) to simulate
        seed: Random seed for reproducibility
        
    Returns:
        Dict with 'strategies' (list of SimulationResult) and 'oracle' info
    """
    if bid_levels is None:
        bid_levels = [0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20]

    arm_names = [str(b) for b in bid_levels]
    rng = np.random.default_rng(seed)
    response_model = BidResponseModel()

    # Pre-compute all outcomes for all arms × all rounds (for fair comparison)
    all_outcomes = {}
    for bl in bid_levels:
        outcomes = []
        for _ in range(n_rounds):
            result = response_model.simulate(
                base_impressions=base_metrics["base_impressions"],
                base_ctr=base_metrics["base_ctr"],
                base_cpc=base_metrics["base_cpc"],
                base_conv_rate=base_metrics["base_conv_rate"],
                base_aov=base_metrics["base_aov"],
                bid_multiplier=bl,
                rng=rng,
            )
            outcomes.append(result["profit"])
        all_outcomes[str(bl)] = outcomes

    # Oracle: always picks the best arm per round
    oracle_rewards = []
    for t in range(n_rounds):
        best_reward = max(all_outcomes[arm][t] for arm in arm_names)
        oracle_rewards.append(best_reward)
    oracle_total = sum(oracle_rewards)
    oracle_best_arm = max(
        arm_names,
        key=lambda arm: sum(all_outcomes[arm]),
    )

    # ── Strategy: Thompson Sampling ──
    ts_bandit = ThompsonSampling(arm_names, arm_type="normal")
    ts_rewards = []
    ts_counts = {arm: 0 for arm in arm_names}

    for t in range(n_rounds):
        arm, _ = ts_bandit.select_arm()
        reward = all_outcomes[arm][t]
        ts_bandit.update(arm, reward, obs_sigma=max(abs(reward) * 0.3, 1.0))
        ts_rewards.append(reward)
        ts_counts[arm] += 1

    # ── Strategy: Epsilon-Greedy (ε=0.1) ──
    eg_means = {arm: 0.0 for arm in arm_names}
    eg_counts = {arm: 0 for arm in arm_names}
    eg_rewards = []
    eg_rng = np.random.default_rng(seed + 1)

    for t in range(n_rounds):
        if eg_rng.random() < 0.1 or t < len(arm_names):
            arm = arm_names[t % len(arm_names)] if t < len(arm_names) else eg_rng.choice(arm_names)
        else:
            arm = max(arm_names, key=lambda a: eg_means[a])

        reward = all_outcomes[arm][t]
        eg_counts[arm] += 1
        eg_means[arm] += (reward - eg_means[arm]) / eg_counts[arm]
        eg_rewards.append(reward)

    # ── Strategy: Uniform Random ──
    ur_rng = np.random.default_rng(seed + 2)
    ur_rewards = []
    ur_counts = {arm: 0 for arm in arm_names}

    for t in range(n_rounds):
        arm = ur_rng.choice(arm_names)
        reward = all_outcomes[arm][t]
        ur_rewards.append(reward)
        ur_counts[arm] += 1

    # ── Build results ──
    def build_result(name, rewards, counts):
        cum_rewards = list(np.cumsum(rewards))
        cum_oracle = list(np.cumsum(oracle_rewards))
        cum_regret = [round(o - r, 2) for o, r in zip(cum_oracle, cum_rewards)]
        return SimulationResult(
            strategy=name,
            total_reward=round(sum(rewards), 2),
            cumulative_rewards=cum_rewards,
            cumulative_regret=cum_regret,
            arm_counts=counts,
        )

    results = {
        "strategies": [
            build_result("Thompson Sampling", ts_rewards, ts_counts),
            build_result("Epsilon-Greedy (ε=0.1)", eg_rewards, dict(eg_counts)),
            build_result("Uniform Random", ur_rewards, ur_counts),
        ],
        "oracle": {
            "total_reward": round(oracle_total, 2),
            "best_arm": oracle_best_arm,
        },
        "n_rounds": n_rounds,
        "bid_levels": bid_levels,
        "ts_beliefs": ts_bandit.get_beliefs(),
    }

    logger.info(
        f"Simulation complete ({n_rounds} rounds): "
        f"TS={results['strategies'][0].total_reward:.0f}, "
        f"EG={results['strategies'][1].total_reward:.0f}, "
        f"Random={results['strategies'][2].total_reward:.0f}, "
        f"Oracle={oracle_total:.0f}"
    )

    return results