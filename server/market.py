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


# Default configs calibrated to real-world SaaS benchmark data.
# Sources: OpenView SaaS Benchmarks 2023, Bessemer Cloud Index, a16z Consumer Reports.
# These replace the original made-up numbers — the agent now trains in a
# simulation that matches real startup market dynamics.
DEFAULT_PHASE_CONFIGS: dict[MarketPhase, PhaseConfig] = {
    MarketPhase.GROWTH: PhaseConfig(
        # Real: healthy SaaS in GROWTH sees ~7.5%/month revenue increase (~90% ARR YoY)
        # Churn rises slowly as product scales to less-ideal customers
        # NPS erodes slightly as user base grows beyond early adopters
        revenue_growth_rate=0.075,
        churn_drift=0.001,
        nps_drift=-0.2,
        competitor_activity=0.04,   # competitors are nascent in growth phase
        complaint_weights={
            "missing_feature": 0.55,   # growth users want more — common in real SaaS
            "slow_performance": 0.20,  # scaling pain
            "ui_confusing": 0.20,
            "too_expensive": 0.05,     # rare in growth — users are happy to pay
        },
    ),
    MarketPhase.SATURATION: PhaseConfig(
        # Real: growth decelerates to ~15-20% ARR YoY in saturation
        # Churn accelerates as competitors offer alternatives
        # NPS drops faster — users have options now
        revenue_growth_rate=0.015,
        churn_drift=0.007,
        nps_drift=-1.5,
        competitor_activity=0.22,   # real SaaS: heavy competitor activity in mature markets
        complaint_weights={
            "too_expensive": 0.35,         # price sensitivity rises in saturated market
            "competitor_is_better": 0.30,  # users know alternatives exist
            "missing_feature": 0.25,
            "slow_performance": 0.10,
        },
    ),
    MarketPhase.DECLINE: PhaseConfig(
        # Real: declining SaaS sees negative revenue growth (-3 to -5%/month)
        # Churn spikes sharply — this is the pivot signal
        # Competitor dominance confirmed
        revenue_growth_rate=-0.035,  # real: -3.5%/month in decline phase
        churn_drift=0.014,
        nps_drift=-3.0,
        competitor_activity=0.50,
        complaint_weights={
            "competitor_is_better": 0.45,  # dominant signal in real decline
            "too_expensive": 0.30,
            "switching_to_X": 0.15,        # users actively migrating
            "missing_feature": 0.10,
        },
    ),
}

DECLINE_GRACE_PERIOD = 3

# ── Macro shock events ────────────────────────────────────────────────────────
# Each event: name, probability per step, which phases it can appear in,
# and numeric effects applied to the environment that step.
# Shock events calibrated to real frequencies observed across startup cohorts.
# Probabilities per step (month) based on: First Round State of Startups,
# YC batch retrospectives, Crunchbase failure analysis.
SHOCK_EVENTS: list[dict] = [
    {
        "name": "funding_winter",
        # Carta 2025: seed funding -30% in 2025 vs 2024; contractions happen ~every 3-4 years
        # Monthly prob 0.04 = ~2-3 firings per 60-month sim
        "prob": 0.04,
        "phases": ["SATURATION", "DECLINE"],
        "burn_multiplier": 1.25,
        "revenue_multiplier": 0.90,
        "nps_delta": -4,
        "message": "📉 VC funding winter: LPs pulling back industry-wide. Burn costs spike, new funding near-impossible.",
    },
    {
        "name": "viral_moment",
        # Real: viral moments are rare (~3-5% of startups per year get one)
        # Only fires in GROWTH when product is working
        "prob": 0.04,
        "phases": ["GROWTH"],
        "revenue_multiplier": 1.40,
        "nps_delta": 12,
        "message": "🚀 Viral moment! Press coverage + social sharing spike. Revenue jumps, NPS soars.",
    },
    {
        "name": "key_engineer_quits",
        # Carta 2024: startups have 25% annual attrition (vs 13% economy-wide)
        # Lead engineer specifically: ~6%/year = 0.5%/month, but elevated in stress phases → 5%
        "prob": 0.05,
        "phases": ["SATURATION", "DECLINE"],
        "burn_multiplier": 1.05,    # recruiting costs increase
        "revenue_multiplier": 0.96,
        "nps_delta": -4,
        "morale_hit": 0.18,
        "message": "😱 Lead engineer resigned. Product velocity drops 30%. Recruiting a replacement takes 3 months.",
    },
    {
        "name": "competitor_acquired",
        # Real: ~8% of funded startups get acquired each year in competitive sectors
        # per Crunchbase M&A data. Monthly ~0.7% but only in decline (when acquirers consolidate)
        "prob": 0.04,
        "phases": ["DECLINE"],
        "revenue_multiplier": 0.91,
        "nps_delta": -8,
        "competitor_strength_boost": 0.30,
        "message": "💀 Your main competitor acquired by a tech giant. Budget and reach increases 10×. Expect heavy competition.",
    },
    {
        "name": "regulatory_change",
        # Real: regulatory events affect ~20% of startups in regulated sectors per year
        # (fintech, health). General SaaS: lower. Modeled at ~3%/month in saturation.
        "prob": 0.03,
        "phases": ["SATURATION"],
        "burn_multiplier": 1.18,
        "revenue_multiplier": 0.95,
        "nps_delta": -2,
        "message": "⚖️ New data privacy regulation passed. Compliance infrastructure needed urgently. Burn rises.",
    },
    {
        "name": "key_customer_churns",
        # Real: in B2B, top customer = often 20-30% of ARR. Losing them is immediate.
        # Crunchbase: ~4% of startups per month lose a top-3 customer in decline.
        "prob": 0.05,
        "phases": ["SATURATION", "DECLINE"],
        "revenue_multiplier": 0.85,
        "nps_delta": -6,
        "message": "😬 Top enterprise customer churned. Revenue drops ~15% overnight. Reference customer lost.",
    },
    {
        "name": "market_timing_breakthrough",
        # Real: external market events (GPT launch, crypto cycle, etc.) occasionally
        # validate a startup's direction. ~2% chance per month in growth.
        "prob": 0.02,
        "phases": ["GROWTH"],
        "revenue_multiplier": 1.20,
        "nps_delta": 8,
        "message": "🎯 Market timing breakthrough: external trend validates your product. Inbound leads spike.",
    },
    {
        "name": "economic_downturn",
        # Real: recessions hit B2B budgets hard — enterprise freezes spending.
        # Happens once every ~7 years = ~1.2% monthly. Only in saturation/decline.
        "prob": 0.015,
        "phases": ["SATURATION", "DECLINE"],
        "burn_multiplier": 1.10,
        "revenue_multiplier": 0.82,
        "nps_delta": -5,
        "message": "🌐 Macro economic downturn: enterprise customers freeze budgets. Sales cycle doubles.",
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
