"""
RewardCalculator — dense, multi-objective, phase-weighted reward function.

Upgraded from sparse pivot-timing rewards to a continuous management scorecard
per plan.md Section 6.

11 composable rubrics, phase-weighted:

  ALWAYS ACTIVE:
    1. survival          — dense time penalty + death penalty + survival bonus
    2. growth            — revenue delta + churn penalty
    3. pivot_timing      — core signal: early = penalize, on-time = reward, late = decay

  FINANCIAL:
    4. milestone         — investor funding secured
    5. board_pressure    — action under board ultimatum
    6. acqui_hire        — SELL action timing quality
    7. shock_survival    — survived a macro shock

  NEW — MULTI-DIMENSIONAL (plan.md):
    8. product_health    — PMF score + tech debt management
    9. team_morale       — maintaining team above 0.4 morale threshold
   10. unit_economics    — LTV:CAC ratio optimization
   11. founder_trust     — maintaining founder trust above 0.5

Phase-weighted rewards (plan.md Section 2 — Reward Shape):
  PRE-SEED / GROWTH:   weights PMF + morale heavily, ignores tech debt penalty
  SERIES A / SATURATION: weights unit economics + churn harshly
  DISTRESSED / DECLINE: pivot timing dominant, burn management critical
"""
from models import ActionType
from server.market import MarketPhase


# ── Phase-specific reward multipliers ────────────────────────────────────────
PHASE_WEIGHTS: dict[str, dict[str, float]] = {
    "GROWTH": {
        "product_health": 1.5,    # PMF matters most in growth
        "team_morale":    1.3,    # morale drives velocity
        "unit_economics": 0.5,    # unit econ matters less when growing fast
        "founder_trust":  1.2,
        "growth":         1.5,
        "pivot_timing":   0.3,    # don't pivot in growth
    },
    "SATURATION": {
        "product_health": 1.0,
        "team_morale":    1.0,
        "unit_economics": 1.5,   # unit econ now critical
        "founder_trust":  1.0,
        "growth":         1.2,
        "pivot_timing":   1.0,
    },
    "DECLINE": {
        "product_health": 0.8,
        "team_morale":    0.8,
        "unit_economics": 1.0,
        "founder_trust":  0.8,
        "growth":         0.8,
        "pivot_timing":   2.0,   # pivot timing is the dominant signal in decline
    },
}


class RewardCalculator:
    """
    11 composable reward rubrics. Total reward = phase-weighted sum.
    Dense signals every step prevent the Credit Assignment Problem
    (the agent knows immediately which decisions are paying off).
    """

    def compute(
        self,
        state: dict,
        action_type: ActionType,
        next_state: dict,
        episode_data: dict,
    ) -> tuple[float, dict]:
        breakdown = {}
        phase = episode_data.get("phase", "GROWTH")
        pw = PHASE_WEIGHTS.get(phase, PHASE_WEIGHTS["GROWTH"])

        # ── Always-active rubrics ─────────────────────────────────────────────
        breakdown["survival"]      = self._survival(next_state, episode_data)
        breakdown["growth"]        = self._growth(state, next_state) * pw.get("growth", 1.0)
        breakdown["pivot_timing"]  = self._pivot_timing(action_type, state, episode_data) * pw.get("pivot_timing", 1.0)

        # ── Financial rubrics ─────────────────────────────────────────────────
        breakdown["milestone"]     = self._milestone(episode_data)
        breakdown["board_pressure"]= self._board_pressure(action_type, state)
        breakdown["acqui_hire"]    = self._acqui_hire(action_type, state)
        breakdown["shock_survival"]= self._shock_survival(next_state, episode_data)

        # ── Multi-dimensional rubrics (NEW) ───────────────────────────────────
        breakdown["product_health"]  = self._product_health(state, next_state) * pw.get("product_health", 1.0)
        breakdown["team_morale"]     = self._team_morale(state, next_state) * pw.get("team_morale", 1.0)
        breakdown["unit_economics"]  = self._unit_economics(state, next_state) * pw.get("unit_economics", 1.0)
        breakdown["founder_trust"]   = self._founder_trust(state, next_state) * pw.get("founder_trust", 1.0)

        total = sum(breakdown.values())
        breakdown["total"] = total
        return total, breakdown

    # ── Rubric 1: Survival ────────────────────────────────────────────────────
    def _survival(self, next_state: dict, episode_data: dict) -> float:
        reward = -1.0   # dense time penalty: alive costs 1/step
        if next_state.get("runway_remaining", 0) <= 0:
            reward -= 200.0   # death penalty
        elif episode_data.get("done") and episode_data.get("step", 0) >= 59:
            reward += 150.0   # full survival bonus
        return reward

    # ── Rubric 2: Growth ──────────────────────────────────────────────────────
    def _growth(self, state: dict, next_state: dict) -> float:
        reward = 0.0
        prev_rev = state.get("monthly_revenue", 1)
        next_rev = next_state.get("monthly_revenue", 1)
        if prev_rev > 0:
            pct_change = (next_rev - prev_rev) / prev_rev
            reward += pct_change * 50

        churn_delta = next_state.get("churn_rate", 0) - state.get("churn_rate", 0)
        reward -= churn_delta * 200
        return reward

    # ── Rubric 3: Pivot timing ────────────────────────────────────────────────
    def _pivot_timing(self, action_type: ActionType, state: dict, episode_data: dict) -> float:
        if action_type != ActionType.PIVOT:
            return 0.0
        if not episode_data.get("pivot_was_necessary", False):
            return -20.0

        optimal_start = episode_data.get("optimal_pivot_start", 39)
        current_step  = state.get("step", 0)
        steps_late    = current_step - optimal_start

        if steps_late < 0:
            return -10.0   # too early
        timing_score = max(0.1, 1.0 - steps_late * 0.09)
        return 50.0 * timing_score

    # ── Rubric 4: Milestone ───────────────────────────────────────────────────
    def _milestone(self, episode_data: dict) -> float:
        if episode_data.get("investor_milestone_hit", False):
            return 100.0
        return 0.0

    # ── Rubric 5: Board pressure ──────────────────────────────────────────────
    def _board_pressure(self, action_type: ActionType, state: dict) -> float:
        step   = state.get("step", 0)
        runway = state.get("runway_remaining", 18)
        if step < 40 or runway >= 6:
            return 0.0
        decisive = {ActionType.PIVOT, ActionType.CUT_COSTS, ActionType.FUNDRAISE,
                    ActionType.SELL, ActionType.FIRE}
        if action_type in decisive:
            return 10.0
        return -15.0

    # ── Rubric 6: Acqui-hire ──────────────────────────────────────────────────
    def _acqui_hire(self, action_type: ActionType, state: dict) -> float:
        if action_type != ActionType.SELL:
            return 0.0
        runway = state.get("runway_remaining", 18)
        if runway <= 2:
            return 50.0
        elif runway <= 5:
            return 10.0
        return -40.0

    # ── Rubric 7: Shock survival ──────────────────────────────────────────────
    def _shock_survival(self, next_state: dict, episode_data: dict) -> float:
        if not episode_data.get("shock_active"):
            return 0.0
        if next_state.get("runway_remaining", 0) > 0:
            return 5.0
        return 0.0

    # ── Rubric 8: Product health (NEW) ───────────────────────────────────────
    def _product_health(self, state: dict, next_state: dict) -> float:
        """
        Dense reward for PMF management and tech debt control.
        Plan.md: 'reward function shifts to heavily penalize high churn, poor unit
        economics, and unmanaged tech debt once Series A revenue thresholds are hit.'
        """
        reward = 0.0

        # PMF improvement bonus
        pmf_now  = next_state.get("pmf_score", 0.5)
        pmf_prev = state.get("pmf_score", 0.5)
        pmf_delta = pmf_now - pmf_prev
        reward += pmf_delta * 30.0   # +30 reward per PMF point gained

        # PMF zone rewards/penalties
        if pmf_now >= 0.70:
            reward += 4.0    # excellent PMF: team is building the right thing
        elif pmf_now >= 0.50:
            reward += 1.0    # decent
        elif pmf_now < 0.30:
            reward -= 6.0    # PMF failure signal

        # Tech debt penalty (high debt slows everything)
        debt = next_state.get("tech_debt_ratio", 0.15)
        severity = next_state.get("tech_debt_severity", "low")
        debt_penalties = {"low": 0, "medium": -1, "high": -4, "critical": -10}
        reward += debt_penalties.get(severity, 0)

        # Feature shipped bonus
        if next_state.get("features_shipped_last", 0) > 0:
            reward += 2.0

        return reward

    # ── Rubric 9: Team morale (NEW) ───────────────────────────────────────────
    def _team_morale(self, state: dict, next_state: dict) -> float:
        """
        Dense reward for keeping the team healthy.
        Plan.md: 'morale above 40% maintained — morale drops trigger multi-step cascades.'
        """
        reward = 0.0
        morale = next_state.get("team_morale", 0.75)

        if morale >= 0.70:
            reward += 3.0
        elif morale >= 0.50:
            reward += 1.0
        elif morale >= 0.40:
            reward -= 1.0    # warning zone
        elif morale < 0.30:
            reward -= 8.0    # danger zone: productivity 2x slower (DORA research)

        # Morale improvement bonus (recovering from crisis)
        morale_prev = state.get("team_morale", 0.75)
        if morale > morale_prev + 0.05:
            reward += 3.0   # morale recovery is hard — reward it

        # FIRE action morale penalty amplifier
        if next_state.get("team_size", 8) < state.get("team_size", 8):
            reward -= 5.0   # losing team member always hurts morale

        return reward

    # ── Rubric 10: Unit economics (NEW) ──────────────────────────────────────
    def _unit_economics(self, state: dict, next_state: dict) -> float:
        """
        LTV:CAC ratio management.
        Plan.md: 'CAC on LinkedIn ads $450, LTV capped at $900 due to 6% churn → kills runway.'
        """
        reward = 0.0
        ratio = next_state.get("ltv_cac_ratio", 3.0)

        if ratio >= 4.0:
            reward += 5.0    # excellent unit economics
        elif ratio >= 3.0:
            reward += 2.0    # healthy (industry benchmark)
        elif ratio >= 1.5:
            reward -= 1.0    # borderline
        elif ratio >= 1.0:
            reward -= 4.0    # bad — burning money on acquisition
        else:
            reward -= 10.0   # catastrophic — each customer costs more than they return

        # CAC payback trend
        cac_now  = next_state.get("cac", 1500)
        cac_prev = state.get("cac", 1500)
        if cac_now < cac_prev * 0.95:
            reward += 2.0   # improving unit economics

        return reward

    # ── Rubric 11: Founder trust (NEW) ───────────────────────────────────────
    def _founder_trust(self, state: dict, next_state: dict) -> float:
        """
        Reward for maintaining founder trust in the advisor.
        Plan.md: 'Strategist must learn Founder management alongside business management.'
        """
        reward = 0.0
        trust = next_state.get("founder_trust", 0.75)
        prev  = state.get("founder_trust", 0.75)

        if trust >= 0.70:
            reward += 2.0
        elif trust >= 0.50:
            reward += 0.5
        elif trust < 0.30:
            reward -= 5.0   # founder ignoring advisor = no leverage

        # Trust recovery bonus
        if trust > prev + 0.04:
            reward += 3.0   # hard to rebuild trust — reward it

        # Burnout penalty (burned-out founder makes bad decisions even with good advice)
        burnout = next_state.get("founder_burnout", 0.10)
        if burnout > 0.70:
            reward -= 3.0

        return reward
