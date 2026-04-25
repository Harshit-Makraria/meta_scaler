"""
CompetitorAgent — reactive market rival that mirrors and counters the startup's moves.

Upgraded from static rule-based plays to a fully REACTIVE agent.
Per plan.md Section 4:
  - If the Strategist advises LAUNCH_FEATURE → competitor fast-follows (50% chance next step)
  - If the Strategist advises HIRE → competitor triggers TALENT_RAID
  - If the Strategist advises MARKETING_CAMPAIGN → competitor triggers AGGRESSIVE_MKT
  - If the Strategist advises SET_PRICING (raise) → competitor triggers PRICE_WAR
  - If the Strategist is pivoting → competitor acquires new users in the vacuum

Underlying logic:
  The competitor represents a well-funded rival with a 3-person intelligence team
  that monitors public signals (job postings, press releases, pricing pages).
  They react with 1-2 month lag — just enough for the startup to gain advantage
  if they move fast, but punishing if they telegraph moves without executing.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from enum import Enum


class CompetitorPlay(str, Enum):
    DORMANT        = "dormant"
    LAUNCH_FEATURE = "launch_feature"
    PRICE_WAR      = "price_war"
    TALENT_RAID    = "talent_raid"
    AGGRESSIVE_MKT = "aggressive_mkt"
    VACUUM_GRAB    = "vacuum_grab"   # NEW: competitor fills pivot vacuum


# Revenue steal per play (fraction of monthly_revenue lost)
_STEAL_RATE: dict[CompetitorPlay, float] = {
    CompetitorPlay.DORMANT:        0.00,
    CompetitorPlay.LAUNCH_FEATURE: 0.03,
    CompetitorPlay.PRICE_WAR:      0.06,
    CompetitorPlay.TALENT_RAID:    0.02,
    CompetitorPlay.AGGRESSIVE_MKT: 0.04,
    CompetitorPlay.VACUUM_GRAB:    0.08,  # biggest steal: fills your pivot vacuum
}

_BURN_DELTA: dict[CompetitorPlay, float] = {
    CompetitorPlay.DORMANT:        0.00,
    CompetitorPlay.LAUNCH_FEATURE: 0.01,
    CompetitorPlay.PRICE_WAR:      0.00,
    CompetitorPlay.TALENT_RAID:    0.04,   # recruiting costs rise
    CompetitorPlay.AGGRESSIVE_MKT: 0.00,
    CompetitorPlay.VACUUM_GRAB:    0.00,
}

_DESCRIPTIONS: dict[CompetitorPlay, str] = {
    CompetitorPlay.DORMANT:
        "Competitor quiet this month.",
    CompetitorPlay.LAUNCH_FEATURE:
        "Rival launched a competing feature — users are comparing products.",
    CompetitorPlay.PRICE_WAR:
        "Competitor slashes prices to undercut you — price-sensitive users are churning.",
    CompetitorPlay.TALENT_RAID:
        "Rival is poaching engineers — your velocity slows and burn ticks up.",
    CompetitorPlay.AGGRESSIVE_MKT:
        "Competitor floods paid channels — your CAC is rising, pipeline threatened.",
    CompetitorPlay.VACUUM_GRAB:
        "Competitor capitalizes on your pivot — grabbing users you left behind.",
}


@dataclass
class CompetitorAgent:
    """
    Reactive competitor. Call notify_agent_action() BEFORE tick() to give it the
    startup's move so it can react with 1-step lag.
    """
    rng_seed: int = 7

    current_play: CompetitorPlay = field(default=CompetitorPlay.DORMANT, init=False)
    market_share_impact: float   = field(default=0.0, init=False)
    burn_impact: float           = field(default=0.0, init=False)
    strength: float              = field(default=0.3, init=False)
    last_play_description: str   = field(default="", init=False)

    def __post_init__(self):
        self._rng   = random.Random(self.rng_seed)
        self._step  = 0
        # Queued reaction from last step's agent action (1-step lag)
        self._queued_reaction: CompetitorPlay | None = None

    # ── Main API ──────────────────────────────────────────────────────────────

    def notify_agent_action(self, action_type_value: str) -> None:
        """
        Tell the competitor what move the startup just made.
        Competitor will react next step (1-month intelligence lag).
        Called BEFORE tick() in the environment step.
        """
        reaction_map = {
            "LAUNCH_FEATURE":     CompetitorPlay.LAUNCH_FEATURE,   # fast-follow 50% of time
            "HIRE":               CompetitorPlay.TALENT_RAID,        # poach the type of hire
            "MARKETING_CAMPAIGN": CompetitorPlay.AGGRESSIVE_MKT,    # counter-spend
            "SET_PRICING":        CompetitorPlay.PRICE_WAR,          # undercut
            "PIVOT":              CompetitorPlay.VACUUM_GRAB,        # fill the vacuum
        }
        if action_type_value in reaction_map:
            # Not every signal triggers a reaction — depends on strength + RNG
            prob = 0.35 + self.strength * 0.30   # stronger competitor reacts more reliably
            if self._rng.random() < prob:
                self._queued_reaction = reaction_map[action_type_value]

    def tick(self, snapshot: dict) -> CompetitorPlay:
        """
        Decide competitor play for this step.
        1. If a queued reaction exists, fire it.
        2. Otherwise, fall back to signal-based heuristics.
        """
        self._step += 1
        self.strength = min(1.0, 0.20 + self._step * 0.012)

        if self._queued_reaction is not None:
            play = self._queued_reaction
            self._queued_reaction = None
        else:
            play = self._choose_play(snapshot)

        self.current_play        = play
        self.market_share_impact = _STEAL_RATE[play] * self.strength
        self.burn_impact         = _BURN_DELTA[play] * self.strength
        self.last_play_description = _DESCRIPTIONS[play]
        return play

    def reset(self):
        self._step               = 0
        self.current_play        = CompetitorPlay.DORMANT
        self.market_share_impact = 0.0
        self.burn_impact         = 0.0
        self.strength            = 0.3
        self.last_play_description = ""
        self._queued_reaction    = None

    # ── Heuristic fallback ────────────────────────────────────────────────────

    def _choose_play(self, snap: dict) -> CompetitorPlay:
        """Signal-based heuristic when no queued reaction. Preserved from v1."""
        nps       = snap.get("nps_score", 50)
        churn     = snap.get("churn_rate", 0.12)
        burn      = snap.get("burn_rate", 120_000)
        revenue   = snap.get("monthly_revenue", 45_000)
        runway    = snap.get("runway_remaining", 18)
        rev_delta = snap.get("revenue_delta_3m", 0.0)

        # Dormancy in early steps
        if self._step < 8 and self._rng.random() < 0.6:
            return CompetitorPlay.DORMANT

        plays: list[tuple[float, CompetitorPlay]] = []

        if burn > revenue * 1.5 or runway < 9:
            plays.append((0.40, CompetitorPlay.PRICE_WAR))
        if churn > 0.28:
            plays.append((0.35, CompetitorPlay.TALENT_RAID))
        if nps > 55:
            plays.append((0.30, CompetitorPlay.LAUNCH_FEATURE))
        if rev_delta < 0.05:
            plays.append((0.30, CompetitorPlay.AGGRESSIVE_MKT))

        if not plays:
            return CompetitorPlay.DORMANT

        total = sum(w for w, _ in plays)
        r = self._rng.random() * total
        cumulative = 0.0
        for weight, play in plays:
            cumulative += weight
            if r < cumulative:
                return play
        return plays[-1][1]
