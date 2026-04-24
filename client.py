"""
ThePivotEnv — openenv-core EnvClient for The Pivot environment.
Uses persistent WebSocket connection (lower latency for training loops).
For sync usage: env = ThePivotEnv(...).sync()
"""
from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import PivotAction, PivotObservation


class ThePivotEnv(EnvClient[PivotAction, PivotObservation, State]):
    """
    WebSocket client for The Pivot environment server.

    Async usage (recommended for training):
        async with ThePivotEnv("http://localhost:8000") as env:
            obs = await env.reset()
            while not obs.done:
                action = agent.act(obs)
                obs = await env.step(action)

    Sync usage (for quick tests):
        env = ThePivotEnv("http://localhost:8000").sync()
        with env:
            obs = env.reset()
            obs = env.step(PivotAction(action_type="EXECUTE"))
    """

    def _step_payload(self, action: PivotAction) -> Dict:
        return {
            "action_type": action.action_type.value,
            "action_params": action.action_params,
            "metadata": action.metadata,
        }

    def _parse_result(self, payload: Dict) -> StepResult[PivotObservation]:
        obs_data = payload.get("observation", payload)
        observation = PivotObservation(
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward")),
            metadata=obs_data.get("metadata", {}),
            runway_remaining=obs_data.get("runway_remaining", 18),
            monthly_revenue=obs_data.get("monthly_revenue", 45000.0),
            burn_rate=obs_data.get("burn_rate", 120000.0),
            revenue_delta_3m=obs_data.get("revenue_delta_3m", 0.0),
            churn_rate=obs_data.get("churn_rate", 0.12),
            nps_score=obs_data.get("nps_score", 52),
            user_complaints=obs_data.get("user_complaints", []),
            competitor_launched=obs_data.get("competitor_launched", False),
            founder_advice=obs_data.get("founder_advice", "stay_course"),
            founder_confidence=obs_data.get("founder_confidence", 1.0),
            investor_sentiment=obs_data.get("investor_sentiment", 0.65),
            next_milestone=obs_data.get("next_milestone", ""),
            pivot_cost_estimate=obs_data.get("pivot_cost_estimate", 3),
            step=obs_data.get("step", 0),
            max_steps=obs_data.get("max_steps", 60),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
