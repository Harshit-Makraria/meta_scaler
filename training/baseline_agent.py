"""
Baseline agents for comparison against the trained LLM Strategist.

Four baselines (plan.md Section 8 — StrategistAgent added):

1. RandomAgent       — picks a random action every step (lower bound)
2. StubbornAgent     — always EXECUTE, never adapts (common founder failure mode)
3. PanicAgent        — pivots at first sign of trouble (over-reactive)
4. StrategistAgent   — rule-based heuristic CEO executing standard startup playbooks.
                       The LLM agent must out-perform this to prove it learned nuance.

Per plan.md: "The RL agent's strategic advice MUST outlive and out-earn this simple
baseline on average to prove it is learning advanced, multi-factor nuance."

Each agent's act() signature accepts a CoFounderObservation and returns
a CoFounderAction. run_episodes() collects the 30+ dimensional balanced scorecard.

Run:
  python training/baseline_agent.py
"""
import json
import random
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["WANDB_MODE"] = "disabled"

from models import CoFounderAction, ActionType, CoFounderObservation
from server.cofounder_environment import CoFounderEnvironment

ALL_ACTIONS = list(ActionType)


class RandomAgent:
    """Picks a uniformly random action every step. Establishes lower bound."""
    name = "random"

    def act(self, obs: CoFounderObservation) -> CoFounderAction:
        return CoFounderAction(action_type=random.choice(ALL_ACTIONS))


class StubbornAgent:
    """Always executes. Never pivots, never fundraises. Common founder failure."""
    name = "stubborn"

    def act(self, obs: CoFounderObservation) -> CoFounderAction:
        return CoFounderAction(action_type=ActionType.EXECUTE)


class PanicAgent:
    """
    Pivots as soon as any two danger signals appear.
    Represents the over-reactive founder who pivot-chases every dip.
    """
    name = "panic"

    def __init__(self):
        self._pivoted = False

    def act(self, obs: CoFounderObservation) -> CoFounderAction:
        if self._pivoted:
            return CoFounderAction(action_type=ActionType.EXECUTE)

        danger_signals = sum([
            obs.churn_rate > 0.25,
            obs.nps_score < 20,
            obs.revenue_trend == "declining",
            obs.competitor_launched,
            obs.pmf_score < 0.35,
        ])
        if danger_signals >= 2:
            self._pivoted = True
            return CoFounderAction(action_type=ActionType.PIVOT)
        return CoFounderAction(action_type=ActionType.EXECUTE)

    def reset(self):
        self._pivoted = False


class StrategistAgent:
    """
    Rule-based CEO executing standard startup playbooks perfectly.
    Per plan.md: this is the competent benchmark the RL agent must beat.

    Logic follows the lifecycle stages described in plan.md Section 7D:
      Pre-Seed/Growth:  focus on PMF, ship features, stay lean
      Series A/Scaling: fix unit economics, pay down tech debt, hire sales
      Distressed/Decline: ruthlessly manage burn, pivot or sell at right time

    Also incorporates multi-factor logic:
      - Morale < 0.40 → FIRE or CUT_COSTS before hiring
      - Tech debt CRITICAL → LAUNCH_FEATURE with debt payment in mind
      - CAC > LTV → MARKETING_CAMPAIGN only after PMF fix
      - Low runway + good NPS → FUNDRAISE
    """
    name = "strategist"

    def __init__(self):
        self._pivoted        = False
        self._last_action    = ActionType.EXECUTE
        self._months_waiting = 0

    def act(self, obs: CoFounderObservation) -> CoFounderAction:
        action = self._decide(obs)
        self._last_action = action
        return CoFounderAction(action_type=action)

    def _decide(self, obs: CoFounderObservation) -> ActionType:
        # ── EMERGENCY: sell before dying ──────────────────────────────────────
        if obs.runway_remaining <= 2:
            return ActionType.SELL

        # ── CRITICAL runway: aggressive burn control ──────────────────────────
        if obs.runway_remaining <= 4 and obs.burn_rate > obs.monthly_revenue * 2.5:
            if obs.team_morale > 0.50:
                return ActionType.FIRE      # layoff before run out of cash
            return ActionType.CUT_COSTS

        # ── Decline + PMF failure → PIVOT ─────────────────────────────────────
        if (obs.churn_rate > 0.20 and obs.nps_score < 10 and
                not self._pivoted and obs.runway_remaining > 5):
            self._pivoted = True
            return ActionType.PIVOT

        # ── Burnout high → rebuild team morale first ─────────────────────────
        if obs.team_morale < 0.35 and obs.runway_remaining > 6:
            # Don't hire into low morale — address the root cause
            return ActionType.EXECUTE   # wait, let morale recover naturally

        # ── Tech debt CRITICAL + feature backlog → fix before shipping ────────
        if obs.tech_debt_severity == "critical":
            # Pay down debt via a careful feature launch cycle
            return ActionType.LAUNCH_FEATURE   # careful = debt reduction mode

        # ── Unit economics broken → fix PMF before campaigns ─────────────────
        if obs.ltv_cac_ratio < 1.5:
            if obs.pmf_score < 0.40:
                return ActionType.LAUNCH_FEATURE   # fix product first
            elif obs.pmf_score >= 0.55:
                return ActionType.MARKETING_CAMPAIGN   # product ready, optimize CAC

        # ── Good PMF + low brand → partnership for cheap growth ──────────────
        if obs.pmf_score > 0.60 and obs.brand_awareness < 0.25 and obs.runway_remaining > 8:
            return ActionType.PARTNERSHIP

        # ── Low runway + good traction → fundraise ────────────────────────────
        if obs.runway_remaining < 8 and obs.nps_score > 30 and obs.ltv_cac_ratio >= 2.0:
            return ActionType.FUNDRAISE

        # ── Growth phase with high churn → research why ───────────────────────
        if obs.churn_rate > 0.15 and obs.step < 25:
            return ActionType.RESEARCH

        # ── Late stage, slow decline, competitor strong → cut costs ───────────
        if (obs.revenue_trend == "declining" and obs.step > 30 and
                obs.competitor_strength > 0.60):
            return ActionType.CUT_COSTS

        # ── Good growth + team capacity → hire to accelerate ─────────────────
        if (obs.revenue_trend == "growing" and obs.runway_remaining > 12 and
                obs.eng_headcount < 6 and obs.team_morale > 0.65):
            return ActionType.HIRE

        # ── Default: execute and monitor ──────────────────────────────────────
        return ActionType.EXECUTE

    def reset(self):
        self._pivoted        = False
        self._last_action    = ActionType.EXECUTE
        self._months_waiting = 0


def run_episodes(
    agent,
    scenario: dict | None,
    n_episodes: int = 50,
    seed: int = 0,
) -> dict:
    """
    Run N episodes and return aggregate stats including full balanced scorecard.
    Updated to collect 30+ dimensional final state per plan.md.
    """
    env = CoFounderEnvironment(scenario=scenario, rng_seed=seed)
    rewards, lengths, survived, pivot_steps = [], [], [], []

    # New multi-dimensional metrics
    final_pmfs, final_morales, final_ltv_cacs, final_trusts = [], [], [], []
    action_counts = {a.value: 0 for a in ActionType}

    for ep in range(n_episodes):
        if hasattr(agent, "reset"):
            agent.reset()
        obs      = env.reset(seed=seed + ep)
        ep_reward = 0.0

        for _ in range(60):
            action   = agent.act(obs)
            action_counts[action.action_type.value] += 1
            obs      = env.step(action)
            ep_reward += obs.reward or 0
            if obs.done:
                break

        rewards.append(ep_reward)
        lengths.append(obs.step)
        survived.append(obs.step >= 60 or obs.runway_remaining > 0)
        pivot_steps.append(env._pivot_step)

        # Collect final state
        final_pmfs.append(obs.pmf_score)
        final_morales.append(obs.team_morale)
        final_ltv_cacs.append(obs.ltv_cac_ratio)
        final_trusts.append(obs.founder_trust)

    n = len(rewards)

    return {
        "agent":                   agent.name,
        "scenario":                scenario["name"] if scenario else "default",
        "n_episodes":              n_episodes,
        # Core metrics
        "mean_reward":             sum(rewards) / n,
        "min_reward":              min(rewards),
        "max_reward":              max(rewards),
        "survival_rate":           sum(survived) / n,
        "mean_length":             sum(lengths) / n,
        "pivot_rate":              sum(1 for p in pivot_steps if p is not None) / n,
        # Balanced scorecard (new multi-dimensional metrics)
        "mean_final_pmf":          sum(final_pmfs) / n,
        "mean_final_morale":       sum(final_morales) / n,
        "mean_final_ltv_cac":      sum(final_ltv_cacs) / n,
        "mean_final_trust":        sum(final_trusts) / n,
        "mean_balanced_score":     _compute_balanced(
                                      sum(survived) / n,
                                      sum(final_pmfs) / n,
                                      sum(final_morales) / n,
                                      sum(final_ltv_cacs) / n,
                                      sum(final_trusts) / n,
                                   ),
        # Action distribution
        "action_distribution":     {k: v / max(sum(action_counts.values()), 1) for k, v in action_counts.items()},
    }


def _compute_balanced(
    survival_rate: float,
    mean_pmf: float,
    mean_morale: float,
    mean_ltv_cac: float,
    mean_trust: float,
) -> float:
    """Simple balanced score 0-100 matching wandb_logger formula."""
    score = (
        survival_rate * 100 * 0.30 +
        mean_pmf      * 100 * 0.20 +
        mean_morale   * 100 * 0.20 +
        min(mean_ltv_cac / 5.0, 1.0) * 100 * 0.15 +
        mean_trust    * 100 * 0.15
    )
    return round(score, 1)


def run_all_baselines(n_episodes: int = 50) -> list[dict]:
    scenarios_dir = pathlib.Path(__file__).parent.parent / "scenarios"
    scenarios     = []
    for f in scenarios_dir.glob("*.json"):
        with open(f) as fh:
            scenarios.append(json.load(fh))

    agents  = [RandomAgent(), StubbornAgent(), PanicAgent(), StrategistAgent()]
    results = []

    for scenario in scenarios:
        print(f"\n=== {scenario['display_name']} ({scenario['difficulty']}) ===")
        for agent in agents:
            r = run_episodes(agent, scenario, n_episodes)
            results.append(r)
            print(f"  {agent.name:12s}  "
                  f"reward={r['mean_reward']:7.1f}  "
                  f"survival={r['survival_rate']:.0%}  "
                  f"pmf={r['mean_final_pmf']:.2f}  "
                  f"morale={r['mean_final_morale']:.2f}  "
                  f"score={r['mean_balanced_score']:.1f}")
    return results


if __name__ == "__main__":
    import pathlib
    print("Running baseline agents across all 5 scenarios (50 episodes each)...")
    results  = run_all_baselines(n_episodes=50)
    out_path = os.path.join(os.path.dirname(__file__), "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
