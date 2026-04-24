import random
import numpy as np
from server.market import MarketPhase, PhaseConfig

# All possible complaint strings the agent might see
ALL_COMPLAINTS = [
    "missing_feature",
    "slow_performance",
    "ui_confusing",
    "too_expensive",
    "competitor_is_better",
    "switching_to_X",
    "poor_customer_support",
    "unreliable_uptime",
    "onboarding_too_hard",
    "pricing_unclear",
    "mobile_app_missing",
    "api_too_complex",
    "docs_outdated",
    "cant_export_data",
    "no_team_features",
]


class SignalGenerator:
    """
    Takes true market state and produces noisy observations.
    The agent only ever sees signal output — never the raw truth.
    """

    def __init__(self, rng_seed: int = 42):
        self._rng = np.random.default_rng(rng_seed)
        self._noise_reduction = 1.0  # RESEARCH action lowers this temporarily

    def reduce_noise(self, factor: float = 0.3, duration: int = 3):
        """Called when agent takes RESEARCH action."""
        self._noise_reduction = factor
        self._noise_duration = duration

    def tick(self):
        """Called each step to decay noise reduction back to normal."""
        if hasattr(self, "_noise_duration") and self._noise_duration > 0:
            self._noise_duration -= 1
            if self._noise_duration == 0:
                self._noise_reduction = 1.0

    def generate_observation(
        self,
        true_revenue: float,
        true_churn: float,
        true_nps: float,
        true_competitor_event: bool,
        phase_config: PhaseConfig,
        step: int,
    ) -> dict:
        nr = self._noise_reduction  # 1.0 = full noise, 0.3 = reduced noise

        noisy_revenue = float(true_revenue * (1 + self._rng.normal(0, 0.15 * nr)))
        noisy_churn = float(np.clip(true_churn + self._rng.normal(0, 0.08 * nr), 0.01, 0.99))
        noisy_nps = int(np.clip(true_nps + self._rng.normal(0, 12 * nr), -100, 100))

        complaints = self._sample_complaints(phase_config.complaint_weights, nr)

        # Competitor events: sometimes reported falsely, sometimes missed
        if true_competitor_event:
            competitor_reported = random.random() > 0.15 * nr   # 85% chance seen
        else:
            competitor_reported = random.random() < 0.05 * nr   # 5% false positive

        return {
            "noisy_revenue": noisy_revenue,
            "noisy_churn": noisy_churn,
            "noisy_nps": noisy_nps,
            "user_complaints": complaints,
            "competitor_launched": competitor_reported,
        }

    def _sample_complaints(self, weights: dict[str, float], noise_reduction: float) -> list[str]:
        categories = list(weights.keys())
        probs = np.array(list(weights.values()))

        # Add noise to probabilities when noise is high
        if noise_reduction > 0.5:
            noise = self._rng.dirichlet(np.ones(len(probs)) * 2)
            probs = 0.7 * probs + 0.3 * noise  # blend truth with random noise
        probs = probs / probs.sum()

        n_complaints = random.randint(2, 5)
        chosen = list(self._rng.choice(categories, size=min(n_complaints, len(categories)), replace=False, p=probs))

        # Occasionally inject a random off-distribution complaint
        if random.random() < 0.15 * noise_reduction:
            wild_card = random.choice([c for c in ALL_COMPLAINTS if c not in chosen])
            chosen.append(wild_card)

        return chosen
