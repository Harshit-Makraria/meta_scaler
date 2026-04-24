"""
Baseline agents for comparison against the trained LLM.
Three baselines, each representing a naive human strategy:

1. RandomAgent       — picks a random action every step (lower bound)
2. StubbornAgent     — always EXECUTE, never adapts (common founder failure mode)
3. PanicAgent        — pivots the moment any negative signal appears (over-reactive)

Run this file directly to benchmark all three:
  python training/baseline_agent.py
"""
import json
import random
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["WANDB_MODE"] = "disabled"

from models import PivotAction, ActionType, PivotObservation
from server.pivot_environment import ThePivotEnvironment

ALL_ACTIONS = list(ActionType)


class RandomAgent:
    """Picks a uniformly random action every step. Establishes lower bound."""
    name = "random"

    def act(self, obs: PivotObservation) -> PivotAction:
        return PivotAction(action_type=random.choice(ALL_ACTIONS))


class StubbornAgent:
    """Always executes. Never pivots, never fundraises. Common founder failure."""
    name = "stubborn"

    def act(self, obs: PivotObservation) -> PivotAction:
        return PivotAction(action_type=ActionType.EXECUTE)


class PanicAgent:
    """
    Pivots as soon as any two of these signals appear:
    - churn > 0.25
    - NPS < 20
    - revenue trending down
    This is the over-reactive founder who pivot-chases every dip.
    """
    name = "panic"

    def __init__(self):
        self._pivoted = False

    def act(self, obs: PivotObservation) -> PivotAction:
        if self._pivoted:
            return PivotAction(action_type=ActionType.EXECUTE)

        danger_signals = sum([
            obs.churn_rate > 0.25,
            obs.nps_score < 20,
            obs.revenue_trend == "declining",
            obs.competitor_launched,
        ])
        if danger_signals >= 2:
            self._pivoted = True
            return PivotAction(action_type=ActionType.PIVOT)
        return PivotAction(action_type=ActionType.EXECUTE)

    def reset(self):
        self._pivoted = False


def run_episodes(agent, scenario: dict | None, n_episodes: int = 50, seed: int = 0) -> dict:
    """Run N episodes and return aggregate stats."""
    env = ThePivotEnvironment(scenario=scenario, rng_seed=seed)
    rewards, lengths, survived, pivot_steps = [], [], [], []

    for ep in range(n_episodes):
        if hasattr(agent, "reset"):
            agent.reset()
        obs = env.reset(seed=seed + ep)
        ep_reward = 0.0

        for _ in range(60):
            action = agent.act(obs)
            obs = env.step(action)
            ep_reward += obs.reward or 0
            if obs.done:
                break

        rewards.append(ep_reward)
        lengths.append(obs.step)
        survived.append(obs.step >= 60)
        pivot_steps.append(env._pivot_step)

    return {
        "agent": agent.name,
        "scenario": scenario["name"] if scenario else "default",
        "n_episodes": n_episodes,
        "mean_reward": sum(rewards) / len(rewards),
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "survival_rate": sum(survived) / len(survived),
        "mean_length": sum(lengths) / len(lengths),
        "pivot_rate": sum(1 for p in pivot_steps if p is not None) / len(pivot_steps),
    }


def run_all_baselines(n_episodes: int = 50) -> list[dict]:
    import json, pathlib
    scenarios_dir = pathlib.Path(__file__).parent.parent / "scenarios"
    scenarios = []
    for f in scenarios_dir.glob("*.json"):
        with open(f) as fh:
            scenarios.append(json.load(fh))

    agents = [RandomAgent(), StubbornAgent(), PanicAgent()]
    results = []

    for scenario in scenarios:
        print(f"\n=== Scenario: {scenario['display_name']} ({scenario['difficulty']}) ===")
        for agent in agents:
            r = run_episodes(agent, scenario, n_episodes)
            results.append(r)
            print(f"  {agent.name:10s}  reward={r['mean_reward']:7.1f}  "
                  f"survival={r['survival_rate']:.0%}  pivot_rate={r['pivot_rate']:.0%}")

    return results


if __name__ == "__main__":
    print("Running baseline agents across all 5 scenarios (50 episodes each)...")
    results = run_all_baselines(n_episodes=50)
    out_path = os.path.join(os.path.dirname(__file__), "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
