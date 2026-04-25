"""
ProductManager — tracks product health for the CoFounder environment.

Manages three interconnected metrics:
  - PMF Score: product-market fit signal (0-1) derived from NPS + complaint patterns
  - Tech Debt: accumulated shortcuts that slow future velocity (0-1 ratio)
  - Feature Pipeline: what is queued and what shipped

Real-world basis:
  - Superhuman PMF framework: score = % users who'd be "very disappointed" if product vanished
  - DORA metrics: deployment frequency, lead time, change failure rate
  - Tech debt compounding: 1 month rushed = ~3 months refactor debt (industry rule of thumb)

Cross-manager interactions (resolved in cofounder_environment.py):
  - Low TeamMorale → features_shipped drops, tech_debt grows faster
  - MARKETING_CAMPAIGN success + high tech_debt → churn spike (product can't absorb load)
  - Competitor TALENT_RAID → delivery timeline extends (fewer eng hours)
"""
from __future__ import annotations
import random


class ProductManager:
    """
    Tracks PMF score, tech debt, and feature velocity.
    Called each step by CoFounderEnvironment.
    """

    def __init__(
        self,
        initial_pmf_score: float = 0.55,
        initial_tech_debt: float = 0.15,
        initial_pipeline:  int   = 3,
        rng_seed: int = 42,
    ):
        self._pmf_score     = float(initial_pmf_score)   # 0-1
        self._tech_debt     = float(initial_tech_debt)   # 0-1
        self._pipeline      = initial_pipeline            # features queued
        self._features_shipped_last = 1
        self._rng = random.Random(rng_seed)

        # Velocity multiplier — reduced by talent raids and morale drops
        self._velocity_multiplier = 1.0
        # Whether we're in "post-pivot reboot" mode (everything resets)
        self._pivot_rebooting = False
        self._pivot_reboot_steps = 0

    # ── Main step update ──────────────────────────────────────────────────────

    def tick(
        self,
        nps_score: int,
        churn_rate: float,
        complaint_types: list[str],
        team_morale: float,
        eng_headcount: int,
        phase_name: str,
    ) -> None:
        """
        Update PMF, tech debt, and feature velocity based on current world state.
        Called BEFORE action processing so the state reflects end-of-previous-month.
        """
        # PMF update from NPS + churn (real Superhuman-style signal)
        self._update_pmf(nps_score, churn_rate, complaint_types)

        # Tech debt grows passively (engineers write imperfect code under pressure)
        self._passive_debt_accumulation(team_morale, phase_name)

        # Feature velocity
        self._update_velocity(team_morale, eng_headcount)

        # Post-pivot reboot decay
        if self._pivot_rebooting:
            self._pivot_reboot_steps += 1
            if self._pivot_reboot_steps >= 3:
                self._pivot_rebooting = False
                self._velocity_multiplier = 1.0

    # ── Action handlers ───────────────────────────────────────────────────────

    def handle_launch_feature(
        self,
        eng_headcount: int,
        team_morale: float,
        carefully: bool = False,
    ) -> dict:
        """
        Agent chose LAUNCH_FEATURE.
        - If done carefully (research-backed): reduces tech debt, good PMF boost
        - If rushed (low morale or high existing debt): adds tech debt, small PMF boost
        Returns effects dict.
        """
        base_capacity = max(1, eng_headcount)
        debt_penalty  = self._tech_debt * 0.5   # existing debt slows delivery

        # Can we actually ship? (velocity check)
        can_ship = self._rng.random() < (self._velocity_multiplier * (1.0 - debt_penalty * 0.5))

        if not can_ship:
            self._features_shipped_last = 0
            self._tech_debt = min(1.0, self._tech_debt + 0.04)  # rushed attempt adds debt
            return {"shipped": 0, "pmf_delta": 0.0, "debt_delta": +0.04, "message": "Feature delayed — team capacity exhausted."}

        # Ship it
        self._features_shipped_last = 1
        if self._pipeline > 0:
            self._pipeline -= 1

        if carefully or team_morale > 0.65:
            # Good engineering practices: pay down debt slightly
            pmf_boost = 0.04 + self._rng.uniform(0, 0.03)
            debt_change = -0.03  # careful delivery pays down debt
        else:
            # Rushed feature launch: PMF boost but debt grows
            pmf_boost = 0.02 + self._rng.uniform(0, 0.02)
            debt_change = +0.05

        self._pmf_score  = min(1.0, self._pmf_score + pmf_boost)
        self._tech_debt  = max(0.0, min(1.0, self._tech_debt + debt_change))
        self._pipeline  += 1  # pipeline auto-refills with new customer feedback

        return {
            "shipped": 1,
            "pmf_delta": pmf_boost,
            "debt_delta": debt_change,
            "message": f"Feature shipped. PMF +{pmf_boost:.2f}, tech debt {'+' if debt_change > 0 else ''}{debt_change:.2f}.",
        }

    def handle_pivot(self, scenario_multiplier: float = 0.60) -> None:
        """
        Pivot resets the product. PMF drops to near zero (new market).
        Tech debt persists (same codebase). Feature pipeline clears.
        """
        self._pmf_score   = max(0.05, self._pmf_score * 0.25)  # pivot kills existing PMF
        self._pipeline    = 2    # minimal pipeline in new direction
        self._features_shipped_last = 0
        self._pivot_rebooting = True
        self._pivot_reboot_steps = 0
        self._velocity_multiplier = 0.5  # team needs to re-orient

    def handle_research(self) -> None:
        """RESEARCH action: adds 2 well-defined features to pipeline."""
        self._pipeline = min(self._pipeline + 2, 8)
        self._pmf_score = min(1.0, self._pmf_score + 0.02)  # clearer direction slightly boosts PMF

    def handle_competitor_talent_raid(self) -> None:
        """Competitor poaches an engineer — velocity drops for 2 months."""
        self._velocity_multiplier = max(0.4, self._velocity_multiplier - 0.25)
        self._tech_debt = min(1.0, self._tech_debt + 0.03)   # unfinished work adds debt

    # ── Private helpers ───────────────────────────────────────────────────────

    def _update_pmf(
        self,
        nps_score: int,
        churn_rate: float,
        complaint_types: list[str],
    ) -> None:
        """
        PMF score derived from three signals:
          1. NPS contribution (normalized -100..100 → 0..1)
          2. Churn penalty (high churn = low PMF)
          3. Complaint signal (switching_to_X / competitor_is_better = PMF failure)
        Weighted blend: 40% NPS, 40% churn inverse, 20% complaints
        """
        nps_norm   = (nps_score + 100) / 200.0          # 0-1
        churn_inv  = max(0.0, 1.0 - churn_rate * 4.0)  # 25%+ churn → 0 PMF from this signal
        bad_complaints = {"switching_to_X", "competitor_is_better", "too_expensive"}
        complaint_penalty = sum(1 for c in complaint_types if c in bad_complaints) * 0.08

        raw_pmf = 0.40 * nps_norm + 0.40 * churn_inv - complaint_penalty
        target  = max(0.05, min(1.0, raw_pmf))

        # Smooth toward target (PMF doesn't change overnight)
        self._pmf_score += (target - self._pmf_score) * 0.20

    def _passive_debt_accumulation(self, team_morale: float, phase_name: str) -> None:
        """
        Tech debt grows passively each step.
        - Faster in GROWTH (shipping fast = cutting corners)
        - Slower in DECLINE (less new code being written)
        - Offset by morale: high-morale teams write cleaner code
        """
        phase_rates = {"GROWTH": 0.008, "SATURATION": 0.005, "DECLINE": 0.003}
        base_rate   = phase_rates.get(phase_name, 0.005)
        morale_mod  = 1.0 + (0.5 - team_morale)   # morale < 0.5 accelerates debt

        self._tech_debt = min(1.0, self._tech_debt + base_rate * morale_mod)

    def _update_velocity(self, team_morale: float, eng_headcount: int) -> None:
        """
        Velocity multiplier recovers toward 1.0 each step unless something suppresses it.
        Morale < 0.4 is the key danger zone (DORA research: low morale = 2x longer lead times).
        """
        # Natural recovery
        self._velocity_multiplier = min(1.0, self._velocity_multiplier + 0.1)

        # Morale drag
        if team_morale < 0.40:
            self._velocity_multiplier = max(0.3, self._velocity_multiplier - 0.15)
        elif team_morale < 0.60:
            self._velocity_multiplier = max(0.6, self._velocity_multiplier - 0.05)

        # Headcount drag: < 2 engineers = severe velocity problem
        if eng_headcount < 2:
            self._velocity_multiplier = max(0.2, self._velocity_multiplier - 0.20)

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def pmf_score(self) -> float:
        return round(self._pmf_score, 3)

    @property
    def tech_debt_ratio(self) -> float:
        return round(self._tech_debt, 3)

    @property
    def tech_debt_severity(self) -> str:
        if self._tech_debt < 0.20:
            return "low"
        elif self._tech_debt < 0.45:
            return "medium"
        elif self._tech_debt < 0.70:
            return "high"
        return "critical"

    @property
    def features_shipped_last(self) -> int:
        return self._features_shipped_last

    @property
    def feature_pipeline_depth(self) -> int:
        return self._pipeline

    @property
    def velocity_multiplier(self) -> float:
        return self._velocity_multiplier

    def snapshot(self) -> dict:
        return {
            "pmf_score":             self.pmf_score,
            "tech_debt_ratio":       self.tech_debt_ratio,
            "tech_debt_severity":    self.tech_debt_severity,
            "features_shipped_last": self.features_shipped_last,
            "feature_pipeline_depth":self.feature_pipeline_depth,
            "velocity_multiplier":   self.velocity_multiplier,
        }
