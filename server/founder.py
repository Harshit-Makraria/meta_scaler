"""
FounderAgent — fully realized CEO NPC with hidden psychological state.

Upgraded from the original "Ghost Protocol" decay model to a three-axis
psychological entity with:
  - Burnout: grows under sustained financial pressure (hidden)
  - Stubbornness: probability the founder ignores the advisor's recommendation (partially observable)
  - Trust: founder's confidence in the advisor agent — earned through good outcomes (hidden)

Why this matters (plan.md Section 4):
  If the Strategist communicates poorly, the Founder's trust drops, and they may
  override or ignore optimal advice. The Strategist must learn "Founder management"
  alongside business management.

Observable signals (what the agent sees):
  - founder_advice: recommendation (may be wrong if burned out)
  - founder_confidence: how sure they sound (drops with burnout)
  - founder_trust: visible level of trust they have in the advisor
  - founder_burnout: visible but noisy
  - founder_stubbornness: estimated probability they'll ignore advice

Hidden state (never sent to agent):
  - _true_burnout: actual burnout level
  - _trust_history: running history of advice outcomes
"""
from __future__ import annotations
import random
import numpy as np
from server.market import MarketPhase

ADVICE_OPTIONS = ["stay_course", "pivot_now", "cut_costs", "raise_funding",
                  "launch_feature", "hire_aggressively", "run_campaign"]

CORRECT_ADVICE: dict[MarketPhase, str] = {
    MarketPhase.GROWTH:      "stay_course",
    MarketPhase.SATURATION:  "raise_funding",
    MarketPhase.DECLINE:     "pivot_now",
}

# Trust thresholds
TRUST_HIGH   = 0.70
TRUST_MEDIUM = 0.45
TRUST_LOW    = 0.25


class FounderAgent:
    """
    Psychologically realistic CEO NPC.
    The advisor agent (LLM) must manage the founder relationship while giving
    correct strategic advice — both matter for full reward.
    """

    def __init__(self, rng_seed: int = 7):
        self._rng = np.random.default_rng(rng_seed)
        self._rngl = random.Random(rng_seed + 3)

        # ── Hidden states (never sent to agent) ──────────────────────────────
        self._true_burnout       = 0.10   # 0 = fresh, 1 = completely burned out
        self._decay_level        = 0.0    # legacy alias for burnout (backward compat)
        self._trust              = 0.75   # trust in advisor; earned through good outcomes
        self._advice_overridden  = 0      # how many times advisor disagreed with founder
        self._advice_vindicated  = 0      # how many times disagreement led to good outcome
        self._trust_history: list[float] = []

        # ── Stubbornness (partially observable, but noisy) ───────────────────
        # Drawn from a distribution: some founders are naturally more stubborn
        self._base_stubbornness  = float(np.clip(self._rng.normal(0.25, 0.10), 0.05, 0.60))

    # ── Main tick ─────────────────────────────────────────────────────────────

    def tick(
        self,
        runway_remaining: int,
        team_morale: float,
        step: int,
        advisor_was_right_last_step: bool | None = None,
    ) -> None:
        """
        Update hidden state each month.
        advisor_was_right_last_step: True/False/None for trust calibration.
        """
        self._update_burnout(runway_remaining, team_morale, step)
        self._decay_level = self._true_burnout  # keep legacy alias in sync

        if advisor_was_right_last_step is True:
            self._trust = min(1.0, self._trust + 0.06)
            self._advice_vindicated += 1
        elif advisor_was_right_last_step is False:
            self._trust = max(0.05, self._trust - 0.08)

        self._trust_history.append(self._trust)

    # ── Advice generation ─────────────────────────────────────────────────────

    def get_advice(self, market_phase: MarketPhase, step: int, runway: int) -> str:
        """
        Returns the founder's strategic recommendation.
        Accuracy degrades with burnout. High burnout → denial ("stay_course").
        High stubbornness AND low trust → may give advice that contradicts all signals.
        """
        correct = CORRECT_ADVICE[market_phase]
        burnout = self._true_burnout

        # Accuracy curve (mimics real cognitive decline under pressure)
        if burnout < 0.30:
            accuracy = 0.82
        elif burnout < 0.60:
            accuracy = 0.55
        else:
            accuracy = 0.28   # burned-out founder = unreliable signal

        if self._rngl.random() < accuracy:
            return correct
        else:
            if burnout > 0.70:
                # Panicking/burned-out founders default to denial or cost-cutting
                return self._rngl.choice(["stay_course", "cut_costs"])
            # Confident but wrong: picks a random non-correct option
            wrong_options = [a for a in ADVICE_OPTIONS[:4] if a != correct]
            return self._rngl.choice(wrong_options)

    # ── Observable signals ────────────────────────────────────────────────────

    def get_confidence(self) -> float:
        """
        Observable: how confident the founder sounds.
        Drops with burnout. Has noise (founders mask feelings).
        """
        base  = 1.0 - self._true_burnout
        noise = float(self._rng.normal(0, 0.08))
        return float(np.clip(base + noise, 0.05, 1.0))

    def get_trust(self) -> float:
        """
        Observable (with slight noise): how much the founder trusts the advisor.
        If trust < TRUST_LOW, founder will override advisor 40% of the time.
        """
        noise = float(self._rng.normal(0, 0.05))
        return float(np.clip(self._trust + noise, 0.0, 1.0))

    def get_burnout_signal(self) -> float:
        """
        Noisy observable burnout proxy.
        Think: tone of voice, how often they check Slack at 2am, eye contact quality.
        """
        noise = float(self._rng.normal(0, 0.12))
        return float(np.clip(self._true_burnout + noise, 0.0, 1.0))

    def get_stubbornness(self) -> float:
        """
        Stubbornness = base + burnout amplifier.
        High burnout makes stubborn founders worse; even low-stubborn founders get defensive.
        Noisy observable.
        """
        effective = self._base_stubbornness + self._true_burnout * 0.20
        noise     = float(self._rng.normal(0, 0.06))
        return float(np.clip(effective + noise, 0.0, 0.85))

    def will_override_advisor(self) -> bool:
        """
        Returns True if the founder ignores the advisor's recommendation this step.
        Called ONLY if advisor and founder disagree. Probability = stubbornness.
        Below TRUST_LOW: stubbornness amplified (lost faith in advisor).
        """
        p = self.get_stubbornness()
        if self._trust < TRUST_LOW:
            p = min(0.90, p + 0.25)   # founder in low-trust mode: very resistant
        elif self._trust > TRUST_HIGH:
            p = max(0.05, p - 0.15)   # high trust: open to being pushed
        return self._rngl.random() < p

    def get_desperation_signal(self) -> float:
        """Legacy method — returns burnout proxy for backward compat."""
        return self.get_burnout_signal()

    # ── Trust manipulation (called from environment after observing outcome) ──

    def record_advisor_outcome(self, was_positive: bool) -> None:
        """
        After each step resolves, the environment records whether the advisor's
        recommendation led to a positive outcome. Updates trust accordingly.
        """
        if was_positive:
            self._trust = min(1.0, self._trust + 0.04)
        else:
            self._trust = max(0.05, self._trust - 0.06)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _update_burnout(
        self,
        runway_remaining: int,
        team_morale: float,
        step: int,
    ) -> None:
        """
        Burnout grows under:
          - Short runway (existential stress)
          - Low team morale (loneliness at the top)
          - Late stage with no pivot (trapped feeling)
        Recovers slowly (founders are resilient but need time).
        """
        # Financial stress driver
        if runway_remaining < 3:
            self._true_burnout = min(1.0, self._true_burnout + 0.06)
        elif runway_remaining < 6:
            self._true_burnout = min(1.0, self._true_burnout + 0.025)
        else:
            self._true_burnout = max(0.0, self._true_burnout - 0.008)  # natural recovery

        # Team morale compounds stress
        if team_morale < 0.30:
            self._true_burnout = min(1.0, self._true_burnout + 0.03)

        # Late-stage no-resolution fatigue (past month 45 with no exit)
        if step > 45:
            self._true_burnout = min(1.0, self._true_burnout + 0.010)

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def decay_level(self) -> float:
        """Legacy alias for backward compatibility."""
        return self._true_burnout

    @property
    def true_trust(self) -> float:
        """Internal use only — never sent to agent."""
        return self._trust

    @property
    def true_burnout(self) -> float:
        """Internal use only — never sent to agent."""
        return self._true_burnout

    def snapshot(self) -> dict:
        """Returns observable fields for observation building."""
        return {
            "founder_advice":       None,   # filled by get_advice() call
            "founder_confidence":   self.get_confidence(),
            "founder_trust":        self.get_trust(),
            "founder_burnout":      self.get_burnout_signal(),
            "founder_stubbornness": self.get_stubbornness(),
        }
