"""
W&B logging for The Pivot environment.
Logs two streams:
  - Per-step metrics (reward breakdown, financial state)
  - Per-episode summary (survival, pivot timing, final state)
"""
from __future__ import annotations
import os

_wandb = None          # lazily imported so env works without wandb installed
_run = None
_enabled = False


def init(project: str = "the-pivot", run_name: str | None = None, config: dict | None = None):
    """Call once at server startup if W&B logging is desired."""
    global _wandb, _run, _enabled
    try:
        import wandb
        _wandb = wandb
        _run = wandb.init(
            project=project,
            name=run_name,
            config=config or {},
            reinit=True,
        )
        _enabled = True
        print(f"[wandb] Run started: {_run.url}")
    except Exception as e:
        print(f"[wandb] Disabled — {e}")
        _enabled = False


def is_enabled() -> bool:
    return _enabled


def log_step(
    step: int,
    episode: int,
    reward: float,
    reward_breakdown: dict,
    obs_snapshot: dict,
    true_phase: str,
    action_type: str,
):
    if not _enabled:
        return
    _wandb.log({
        "step/reward": reward,
        "step/reward_survival": reward_breakdown.get("survival", 0),
        "step/reward_growth": reward_breakdown.get("growth", 0),
        "step/reward_pivot_timing": reward_breakdown.get("pivot_timing", 0),
        "step/reward_milestone": reward_breakdown.get("milestone", 0),
        "step/reward_founder_awareness": reward_breakdown.get("founder_awareness", 0),
        "step/runway_remaining": obs_snapshot.get("runway_remaining", 0),
        "step/monthly_revenue": obs_snapshot.get("monthly_revenue", 0),
        "step/burn_rate": obs_snapshot.get("burn_rate", 0),
        "step/churn_rate": obs_snapshot.get("churn_rate", 0),
        "step/nps_score": obs_snapshot.get("nps_score", 0),
        "step/investor_sentiment": obs_snapshot.get("investor_sentiment", 0),
        "step/true_phase": true_phase,
        "step/action": action_type,
        "env/step": step,
        "env/episode": episode,
    })


def log_episode(
    episode: int,
    total_reward: float,
    episode_length: int,
    survived: bool,
    pivot_step: int | None,
    final_runway: int,
    final_revenue: float,
    final_churn: float,
    pivot_timing_score: float,
    milestones_hit: int,
    founder_overrides_correct: int,
    scenario_name: str = "default",
):
    if not _enabled:
        return

    # Pivot timing score: 1.0 = optimal, 0.0 = never pivoted or too late
    _wandb.log({
        "episode/total_reward": total_reward,
        "episode/length": episode_length,
        "episode/survived": int(survived),
        "episode/pivot_step": pivot_step if pivot_step is not None else -1,
        "episode/pivot_timing_score": pivot_timing_score,
        "episode/final_runway": final_runway,
        "episode/final_revenue": final_revenue,
        "episode/final_churn": final_churn,
        "episode/milestones_hit": milestones_hit,
        "episode/founder_overrides_correct": founder_overrides_correct,
        "episode/scenario": scenario_name,
        "env/episode": episode,
    })


def finish():
    if _enabled and _wandb:
        _wandb.finish()
