from __future__ import annotations
import random
from dataclasses import dataclass
from enum import Enum


class MarketPhase(str, Enum):
    GROWTH = "GROWTH"
    SATURATION = "SATURATION"
    DECLINE = "DECLINE"


@dataclass
class PhaseConfig:
    revenue_growth_rate: float
    churn_drift: float
    nps_drift: float
    competitor_activity: float
    complaint_weights: dict[str, float]


# Default configs used when no scenario is loaded
DEFAULT_PHASE_CONFIGS: dict[MarketPhase, PhaseConfig] = {
    MarketPhase.GROWTH: PhaseConfig(
        revenue_growth_rate=0.08, churn_drift=0.002, nps_drift=-0.3,
        competitor_activity=0.05,
        complaint_weights={"missing_feature": 0.50, "slow_performance": 0.25, "ui_confusing": 0.20, "too_expensive": 0.05},
    ),
    MarketPhase.SATURATION: PhaseConfig(
        revenue_growth_rate=0.01, churn_drift=0.008, nps_drift=-1.8,
        competitor_activity=0.25,
        complaint_weights={"too_expensive": 0.35, "competitor_is_better": 0.30, "missing_feature": 0.25, "slow_performance": 0.10},
    ),
    MarketPhase.DECLINE: PhaseConfig(
        revenue_growth_rate=-0.04, churn_drift=0.015, nps_drift=-3.2,
        competitor_activity=0.55,
        complaint_weights={"competitor_is_better": 0.45, "too_expensive": 0.30, "switching_to_X": 0.15, "missing_feature": 0.10},
    ),
}

DECLINE_GRACE_PERIOD = 3

# ── Macro shock events ────────────────────────────────────────────────────────
# Each event: name, probability per step, which phases it can appear in,
# and numeric effects applied to the environment that step.
SHOCK_EVENTS: list[dict] = [
    {
        "name": "funding_winter",
        "prob": 0.04,
        "phases": ["SATURATION", "DECLINE"],
        "burn_multiplier": 1.25,
        "revenue_multiplier": 0.92,
        "nps_delta": -4,
        "message": "📉 VC funding winter: LPs are pulling back. Burn costs spike, investor sentiment drops.",
    },
    {
        "name": "viral_moment",
        "prob": 0.03,
        "phases": ["GROWTH"],
        "revenue_multiplier": 1.35,
        "nps_delta": 10,
        "message": "🚀 Viral moment! A tweet went huge. Revenue spikes this month, NPS jumps.",
    },
    {
        "name": "key_engineer_quits",
        "prob": 0.05,
        "phases": ["SATURATION", "DECLINE"],
        "burn_multiplier": 1.0,
        "revenue_multiplier": 0.97,
        "nps_delta": -3,
        "morale_hit": 0.15,
        "message": "😱 Your lead engineer just quit. Product velocity drops. Team morale takes a hit.",
    },
    {
        "name": "competitor_acquired",
        "prob": 0.03,
        "phases": ["DECLINE"],
        "revenue_multiplier": 0.93,
        "nps_delta": -7,
        "competitor_strength_boost": 0.25,
        "message": "💀 Your main competitor was acquired by a tech giant. Expect heavier competition.",
    },
    {
        "name": "regulatory_change",
        "prob": 0.03,
        "phases": ["SATURATION"],
        "burn_multiplier": 1.15,
        "revenue_multiplier": 0.96,
        "message": "⚖️ New regulation affects your sector. Compliance costs rise, growth slows.",
    },
    {
        "name": "key_customer_churns",
        "prob": 0.04,
        "phases": ["SATURATION", "DECLINE"],
        "revenue_multiplier": 0.88,
        "nps_delta": -5,
        "message": "😬 Your top enterprise customer just churned. Revenue takes an immediate hit.",
    },
]


def sample_shock(step: int, phase: str, rng: random.Random) -> dict | None:
    """Return a shock event dict if one triggers this step, else None."""
    eligible = [e for e in SHOCK_EVENTS if phase in e["phases"]]
    for event in eligible:
        if rng.random() < event["prob"]:
            return event
    return None


class MarketSimulator:
    """
    Holds the true (hidden) market state.
    The agent never reads this directly — only noisy signals reach it via SignalGenerator.
    Can be configured from a scenario JSON for curriculum learning.
    """

    def __init__(self, scenario: dict | None = None):
        if scenario:
            self._phase_schedule = self._parse_schedule(scenario["phases"])
            self._phase_configs = self._parse_configs(scenario["phases"], scenario.get("complaint_distributions", {}))
            self._optimal_window = tuple(scenario.get("optimal_pivot_window", [39, 46]))
        else:
            self._phase_schedule = [
                (MarketPhase.GROWTH, 0, 20),
                (MarketPhase.SATURATION, 21, 35),
                (MarketPhase.DECLINE, 36, 60),
            ]
            self._phase_configs = DEFAULT_PHASE_CONFIGS
            self._optimal_window = (39, 46)

        # Precompute decline start for pivot checks
        self._decline_start = next(
            start for phase, start, _ in self._phase_schedule if phase == MarketPhase.DECLINE
        )

    def _parse_schedule(self, phases: dict) -> list:
        schedule = []
        for phase_name, cfg in phases.items():
            phase = MarketPhase(phase_name)
            start, end = cfg["steps"]
            schedule.append((phase, start, end))
        return sorted(schedule, key=lambda x: x[1])

    def _parse_configs(self, phases: dict, complaint_dists: dict) -> dict:
        configs = {}
        for phase_name, cfg in phases.items():
            phase = MarketPhase(phase_name)
            complaint_weights = complaint_dists.get(phase_name, DEFAULT_PHASE_CONFIGS[phase].complaint_weights)
            configs[phase] = PhaseConfig(
                revenue_growth_rate=cfg["revenue_growth_rate"],
                churn_drift=cfg["churn_drift"],
                nps_drift=cfg["nps_drift"],
                competitor_activity=cfg["competitor_activity"],
                complaint_weights=complaint_weights,
            )
        return configs

    def get_phase(self, step: int) -> MarketPhase:
        for phase, start, end in self._phase_schedule:
            if start <= step <= end:
                return phase
        return MarketPhase.DECLINE

    def get_config(self, step: int) -> PhaseConfig:
        return self._phase_configs[self.get_phase(step)]

    def check_pivot_necessity(self, step: int, pivot_step: int | None) -> bool:
        """Pivot is necessary only after DECLINE has been running for DECLINE_GRACE_PERIOD steps."""
        if pivot_step is None:
            return False
        return pivot_step >= (self._decline_start + DECLINE_GRACE_PERIOD)

    def get_optimal_pivot_window(self) -> tuple[int, int]:
        return (self._decline_start + DECLINE_GRACE_PERIOD, self._optimal_window[1])
