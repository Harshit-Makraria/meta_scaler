import random
import numpy as np
from server.market import MarketPhase

ADVICE_OPTIONS = ["stay_course", "pivot_now", "cut_costs", "raise_funding"]

# What correct advice looks like per phase
CORRECT_ADVICE: dict[MarketPhase, str] = {
    MarketPhase.GROWTH: "stay_course",
    MarketPhase.SATURATION: "raise_funding",
    MarketPhase.DECLINE: "pivot_now",
}


class FounderAgent:
    """
    Ghost Protocol mechanic: founder wisdom decays under financial pressure.
    The agent sees founder_confidence and founder_desperation_signal — not decay_level directly.
    """

    def __init__(self, rng_seed: int = 7):
        self._rng = np.random.default_rng(rng_seed)
        self._decay_level = 0.0  # hidden — rises as runway drops

    def update_decay(self, runway_remaining: int, max_runway: int = 18):
        """
        Decay rises as runway shrinks.
        At 18 months: decay=0.0. At 6 months: decay~0.5. At 0 months: decay=1.0.
        """
        normalized_runway = runway_remaining / max_runway
        # Decay is steeper when runway drops below 6 months
        if normalized_runway > 0.33:
            self._decay_level = 1.0 - normalized_runway
        else:
            self._decay_level = min(1.0, 0.67 + (0.33 - normalized_runway) * 1.5)

    def get_advice(self, market_phase: MarketPhase, step: int, runway: int) -> str:
        self.update_decay(runway)
        correct = CORRECT_ADVICE[market_phase]

        decay = self._decay_level
        # High decay biases the founder toward denial ("stay_course") even when wrong
        if decay < 0.3:
            accuracy = 0.85
        elif decay < 0.7:
            accuracy = 0.55
        else:
            accuracy = 0.30

        if random.random() < accuracy:
            return correct
        else:
            if decay > 0.7:
                # Panicking founders default to denial or cost-cutting
                return random.choice(["stay_course", "cut_costs"])
            return random.choice([a for a in ADVICE_OPTIONS if a != correct])

    def get_confidence(self) -> float:
        """Observable: how confident the founder sounds. Drops with decay but with noise."""
        base = 1.0 - self._decay_level
        noise = float(self._rng.normal(0, 0.08))
        return float(np.clip(base + noise, 0.05, 1.0))

    def get_desperation_signal(self) -> float:
        """
        Observable proxy for decay. Noisy — agent can't perfectly infer decay from this.
        Think of it as: tone of voice, how often they check the bank balance, etc.
        """
        noise = float(self._rng.normal(0, 0.12))
        return float(np.clip(self._decay_level + noise, 0.0, 1.0))

    @property
    def decay_level(self) -> float:
        """Internal use only — never sent to the agent."""
        return self._decay_level
