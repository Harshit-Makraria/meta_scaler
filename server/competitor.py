"""
CompetitorAgent — rule-based market rival that responds to the startup's weakness.

How it works (plain English):
- The competitor watches NPS, churn, and runway signals (same noisy view the env has).
- It picks one of four "plays" each step based on thresholds:
    LAUNCH_FEATURE  — when your NPS is high and they want to match you
    PRICE_WAR       — when you're burning cash, they undercut on price
    TALENT_RAID     — when your churn is spiking, they poach your team
    AGGRESSIVE_MKT  — when market is saturating, they spend on ads
- Each play has a market_share_steal value (how many % points of your revenue it bleeds).
- The environment calls competitor.tick(obs_snapshot) each step and reads
  competitor.market_share_impact to apply a revenue penalty.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from enum import Enum


class CompetitorPlay(str, Enum):
    DORMANT        = "dormant"          # no threat this step
    LAUNCH_FEATURE = "launch_feature"   # copies / one-ups your features
    PRICE_WAR      = "price_war"        # undercuts pricing → steals price-sensitive users
    TALENT_RAID    = "talent_raid"      # poaches engineers → slows your execution
    AGGRESSIVE_MKT = "aggressive_mkt"  # outspends you on marketing


# Revenue steal per play (fraction of monthly_revenue lost)
_STEAL_RATE: dict[CompetitorPlay, float] = {
    CompetitorPlay.DORMANT:        0.00,
    CompetitorPlay.LAUNCH_FEATURE: 0.03,
    CompetitorPlay.PRICE_WAR:      0.05,
    CompetitorPlay.TALENT_RAID:    0.02,
    CompetitorPlay.AGGRESSIVE_MKT: 0.04,
}

# Burn-rate increase per play (competitor actions waste your talent/time)
_BURN_DELTA: dict[CompetitorPlay, float] = {
    CompetitorPlay.DORMANT:        0.00,
    CompetitorPlay.LAUNCH_FEATURE: 0.01,
    CompetitorPlay.PRICE_WAR:      0.00,
    CompetitorPlay.TALENT_RAID:    0.03,
    CompetitorPlay.AGGRESSIVE_MKT: 0.00,
}


@dataclass
class CompetitorAgent:
    """
    Rule-based competitor. Call tick() every step with the env's internal snapshot.
    Then read market_share_impact (revenue fraction to subtract) and burn_impact
    (burn fraction to add).
    """
    rng_seed: int = 7

    # readable outputs after each tick()
    current_play: CompetitorPlay = field(default=CompetitorPlay.DORMANT, init=False)
    market_share_impact: float = field(default=0.0, init=False)  # fraction of revenue stolen
    burn_impact: float = field(default=0.0, init=False)          # fraction of burn added
    strength: float = field(default=0.3, init=False)             # grows over time [0, 1]
    last_play_description: str = field(default="", init=False)

    def __post_init__(self):
        self._rng = random.Random(self.rng_seed)
        self._step = 0

    # ── Main API ──────────────────────────────────────────────────────────────

    def tick(self, snapshot: dict) -> CompetitorPlay:
        """
        Decide competitor play for this step.
        snapshot keys used: monthly_revenue, burn_rate, churn_rate, nps_score,
                            runway_remaining, revenue_delta_3m
        """
        self._step += 1
        # Competitor grows stronger over time (simulates funded rival)
        self.strength = min(1.0, 0.2 + self._step * 0.012)

        play = self._choose_play(snapshot)
        self.current_play = play
        self.market_share_impact = _STEAL_RATE[play] * self.strength
        self.burn_impact = _BURN_DELTA[play] * self.strength
        self.last_play_description = _DESCRIPTIONS[play]
        return play

    def reset(self):
        self._step = 0
        self.current_play = CompetitorPlay.DORMANT
        self.market_share_impact = 0.0
        self.burn_impact = 0.0
        self.strength = 0.3
        self.last_play_description = ""

    # ── Decision logic ────────────────────────────────────────────────────────

    def _choose_play(self, snap: dict) -> CompetitorPlay:
        nps        = snap.get("nps_score", 50)
        churn      = snap.get("churn_rate", 0.12)
        burn       = snap.get("burn_rate", 120_000)
        revenue    = snap.get("monthly_revenue", 45_000)
        runway     = snap.get("runway_remaining", 18)
        rev_delta  = snap.get("revenue_delta_3m", 0.0)

        # Dormancy window: competitor is quiet in early steps
        if self._step < 8 and self._rng.random() < 0.6:
            return CompetitorPlay.DORMANT

        plays: list[tuple[float, CompetitorPlay]] = []

        # PRICE_WAR: when you're burning more than you earn (distressed)
        if burn > revenue * 1.5 or runway < 9:
            plays.append((0.40, CompetitorPlay.PRICE_WAR))

        # TALENT_RAID: when your churn is spiking (team morale low)
        if churn > 0.28:
            plays.append((0.35, CompetitorPlay.TALENT_RAID))

        # LAUNCH_FEATURE: when your NPS is high (they want to close the gap)
        if nps > 55:
            plays.append((0.30, CompetitorPlay.LAUNCH_FEATURE))

        # AGGRESSIVE_MKT: when your revenue growth is slowing
        if rev_delta < 0.05:
            plays.append((0.30, CompetitorPlay.AGGRESSIVE_MKT))

        # Dormant if nothing triggers
        if not plays:
            return CompetitorPlay.DORMANT

        # Weighted random pick
        total = sum(w for w, _ in plays)
        r = self._rng.random() * total
        cumulative = 0.0
        for weight, play in plays:
            cumulative += weight
            if r < cumulative:
                return play

        return plays[-1][1]


_DESCRIPTIONS: dict[CompetitorPlay, str] = {
    CompetitorPlay.DORMANT:
        "Competitor quiet this month.",
    CompetitorPlay.LAUNCH_FEATURE:
        "Rival launches a competing feature — users are comparing products.",
    CompetitorPlay.PRICE_WAR:
        "Competitor slashes prices to undercut you — price-sensitive users are churning.",
    CompetitorPlay.TALENT_RAID:
        "Rival is poaching engineers — your velocity slows and burn ticks up.",
    CompetitorPlay.AGGRESSIVE_MKT:
        "Competitor floods paid channels — your customer acquisition cost is rising.",
}
