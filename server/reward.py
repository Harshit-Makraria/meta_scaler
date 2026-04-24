from models import ActionType
from server.market import MarketPhase


class RewardCalculator:
    """
    Five composable rubrics. Each captures a different dimension of good founder behavior.
    Total reward = sum of all rubrics applied each step.
    """

    def compute(
        self,
        state: dict,
        action_type: ActionType,
        next_state: dict,
        episode_data: dict,
    ) -> tuple[float, dict]:
        breakdown = {}

        # 1. Survival: dense penalty for passing time, big penalty for dying
        survival = self._survival(next_state, episode_data)
        breakdown["survival"] = survival

        # 2. Growth: reward revenue gains, penalize churn increases
        growth = self._growth(state, next_state)
        breakdown["growth"] = growth

        # 3. Pivot timing: the core learning signal
        pivot = self._pivot_timing(action_type, state, episode_data)
        breakdown["pivot_timing"] = pivot

        # 4. Milestone: sparse reward for satisfying the investor
        milestone = self._milestone(episode_data)
        breakdown["milestone"] = milestone

        # 5. Founder awareness: bonus for correctly overriding decayed founder
        awareness = self._founder_awareness(action_type, state, next_state, episode_data)
        breakdown["founder_awareness"] = awareness

        # 6. Board pressure: extra penalty for inaction when board is watching
        board = self._board_pressure(action_type, state)
        breakdown["board_pressure"] = board

        # 7. Acqui-hire: graceful exit reward/penalty
        acqui = self._acqui_hire(action_type, state)
        breakdown["acqui_hire"] = acqui

        # 8. Shock survival bonus: small bonus for surviving a shock step
        shock_bonus = self._shock_survival(next_state, episode_data)
        breakdown["shock_survival"] = shock_bonus

        total = sum(breakdown.values())
        breakdown["total"] = total
        return total, breakdown

    def _survival(self, next_state: dict, episode_data: dict) -> float:
        reward = -1.0   # time penalty every step
        if next_state["runway_remaining"] <= 0:
            reward -= 200.0
        elif episode_data.get("done") and episode_data.get("step", 0) >= 59:
            reward += 150.0
        return reward

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

    def _pivot_timing(self, action_type: ActionType, state: dict, episode_data: dict) -> float:
        if action_type != ActionType.PIVOT:
            return 0.0

        if not episode_data.get("pivot_was_necessary", False):
            return -20.0   # penalize unnecessary pivots

        optimal_start = episode_data.get("optimal_pivot_start", 39)
        current_step = state.get("step", 0)
        steps_late = current_step - optimal_start

        if steps_late < 0:
            # Pivoted too early — still in saturation, unnecessary
            return -10.0
        else:
            # Decay score from 1.0 (perfect) to 0.1 (10 steps late)
            timing_score = max(0.1, 1.0 - steps_late * 0.09)
            return 50.0 * timing_score

    def _milestone(self, episode_data: dict) -> float:
        if episode_data.get("investor_milestone_hit", False):
            return 100.0
        return 0.0

    def _founder_awareness(
        self,
        action_type: ActionType,
        state: dict,
        next_state: dict,
        episode_data: dict,
    ) -> float:
        founder_decay = episode_data.get("founder_decay", 0.0)
        if founder_decay < 0.7:
            return 0.0   # founder still reliable — no bonus for overriding

        founder_advice = state.get("founder_advice", "")
        action_str = action_type.value.lower()

        # Map action to what founder would have advised
        advice_to_action = {
            "stay_course": "execute",
            "pivot_now": "pivot",
            "cut_costs": "cut_costs",
            "raise_funding": "fundraise",
        }
        founder_action = advice_to_action.get(founder_advice, "")
        agent_disagreed = action_str != founder_action

        if not agent_disagreed:
            return 0.0

        # Did overriding the founder lead to a better outcome?
        rev_improved = next_state.get("monthly_revenue", 0) >= state.get("monthly_revenue", 0)
        runway_ok = next_state.get("runway_remaining", 0) > 2

        if rev_improved and runway_ok:
            return 30.0   # correctly overrode panicking founder
        return 0.0

    def _board_pressure(self, action_type: ActionType, state: dict) -> float:
        """
        After step 40, if runway < 6 months, the board is watching.
        Reward decisive action (PIVOT, CUT_COSTS, FUNDRAISE, SELL).
        Penalise blindly EXECUTEing with bad metrics.
        """
        step = state.get("step", 0)
        runway = state.get("runway_remaining", 18)
        if step < 40 or runway >= 6:
            return 0.0
        decisive = {ActionType.PIVOT, ActionType.CUT_COSTS, ActionType.FUNDRAISE, ActionType.SELL}
        if action_type in decisive:
            return 10.0   # board happy: you're taking action
        return -15.0      # board ultimatum penalty for blind EXECUTE in crisis

    def _acqui_hire(self, action_type: ActionType, state: dict) -> float:
        """
        SELL action: graceful acqui-hire.
        Rewards survival-exit when truly out of options (runway < 3).
        Punishes selling too early.
        """
        if action_type != ActionType.SELL:
            return 0.0
        runway = state.get("runway_remaining", 18)
        if runway <= 2:
            return 50.0   # good call — sold before dying
        elif runway <= 5:
            return 10.0   # marginal, but defensible
        return -40.0      # sold too early — wasted upside

    def _shock_survival(self, next_state: dict, episode_data: dict) -> float:
        """Small bonus for surviving a macro shock event without dying."""
        if not episode_data.get("shock_active"):
            return 0.0
        if next_state.get("runway_remaining", 0) > 0:
            return 5.0   # survived the shock — resilience bonus
        return 0.0
