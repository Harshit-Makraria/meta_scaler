"""
W&B logging for the CoFounder Strategist environment.

Expanded from 5 metrics to 30+ per plan.md Section 8:
  - Per-step: reward breakdown (11 rubrics), financial state, product health,
              team dynamics, marketing metrics, founder state
  - Per-episode: survival, pivot timing, final state across all dimensions,
                 balanced scorecard summary

The 30+ dimensional state space enables the multi-layer dashboards described
in plan.md: PMF scores mapped against CAC decay, headcount against feature
velocity per month, the exact point where tech debt overwhelmed shipping ability.
"""
from __future__ import annotations
import os

_wandb   = None
_run     = None
_enabled = False


def init(project: str = "the-pivot", run_name: str | None = None, config: dict | None = None):
    """Call once at server startup if W&B logging is desired."""
    global _wandb, _run, _enabled
    try:
        import wandb
        _wandb = wandb
        _run   = wandb.init(
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
    """
    Per-step logging: all 11 reward rubrics + full 30+ field state.
    Used to plot multi-dimensional dashboards tracking cascading effects.
    """
    if not _enabled:
        return

    log = {
        # ── Reward breakdown (all 11 rubrics) ─────────────────────────────────
        "step/reward":                  reward,
        "step/reward_survival":         reward_breakdown.get("survival", 0),
        "step/reward_growth":           reward_breakdown.get("growth", 0),
        "step/reward_pivot_timing":     reward_breakdown.get("pivot_timing", 0),
        "step/reward_milestone":        reward_breakdown.get("milestone", 0),
        "step/reward_board_pressure":   reward_breakdown.get("board_pressure", 0),
        "step/reward_acqui_hire":       reward_breakdown.get("acqui_hire", 0),
        "step/reward_shock_survival":   reward_breakdown.get("shock_survival", 0),
        "step/reward_product_health":   reward_breakdown.get("product_health", 0),
        "step/reward_team_morale":      reward_breakdown.get("team_morale", 0),
        "step/reward_unit_economics":   reward_breakdown.get("unit_economics", 0),
        "step/reward_founder_trust":    reward_breakdown.get("founder_trust", 0),

        # ── Financial state ────────────────────────────────────────────────────
        "step/runway_remaining":        obs_snapshot.get("runway_remaining", 0),
        "step/monthly_revenue":         obs_snapshot.get("monthly_revenue", 0),
        "step/burn_rate":               obs_snapshot.get("burn_rate", 0),
        "step/revenue_delta_3m":        obs_snapshot.get("revenue_delta_3m", 0),
        "step/churn_rate":              obs_snapshot.get("churn_rate", 0),
        "step/nps_score":               obs_snapshot.get("nps_score", 0),

        # ── Product health ─────────────────────────────────────────────────────
        "step/pmf_score":               obs_snapshot.get("pmf_score", 0),
        "step/tech_debt_ratio":         obs_snapshot.get("tech_debt_ratio", 0),
        "step/features_shipped":        obs_snapshot.get("features_shipped_last", 0),
        "step/pipeline_depth":          obs_snapshot.get("feature_pipeline_depth", 0),

        # ── Team dynamics ──────────────────────────────────────────────────────
        "step/team_morale":             obs_snapshot.get("team_morale", 0),
        "step/team_size":               obs_snapshot.get("team_size", 0),
        "step/eng_headcount":           obs_snapshot.get("eng_headcount", 0),
        "step/sales_headcount":         obs_snapshot.get("sales_headcount", 0),

        # ── Marketing / unit economics ─────────────────────────────────────────
        "step/cac":                     obs_snapshot.get("cac", 0),
        "step/ltv":                     obs_snapshot.get("ltv", 0),
        "step/ltv_cac_ratio":           obs_snapshot.get("ltv_cac_ratio", 0),
        "step/brand_awareness":         obs_snapshot.get("brand_awareness", 0),
        "step/pipeline_generated":      obs_snapshot.get("pipeline_generated", 0),

        # ── Founder state ──────────────────────────────────────────────────────
        "step/founder_trust":           obs_snapshot.get("founder_trust", 0),
        "step/founder_burnout":         obs_snapshot.get("founder_burnout", 0),
        "step/founder_confidence":      obs_snapshot.get("founder_confidence", 0),
        "step/investor_sentiment":      obs_snapshot.get("investor_sentiment", 0),

        # ── Context ────────────────────────────────────────────────────────────
        "step/true_phase":              true_phase,
        "step/action":                  action_type,
        "env/step":                     step,
        "env/episode":                  episode,
    }
    _wandb.log(log)


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
    extra: dict | None = None,
):
    """
    Per-episode summary with balanced scorecard.
    Expanded to track all multi-dimensional final state.
    """
    if not _enabled:
        return

    extra = extra or {}
    log = {
        # ── Core episode metrics ───────────────────────────────────────────────
        "episode/total_reward":             total_reward,
        "episode/length":                   episode_length,
        "episode/survived":                 int(survived),
        "episode/pivot_step":               pivot_step if pivot_step is not None else -1,
        "episode/pivot_timing_score":       pivot_timing_score,
        "episode/milestones_hit":           milestones_hit,
        "episode/founder_overrides_correct":founder_overrides_correct,
        "episode/scenario":                 scenario_name,

        # ── Final financial state ──────────────────────────────────────────────
        "episode/final_runway":             final_runway,
        "episode/final_revenue":            final_revenue,
        "episode/final_churn":              final_churn,
        "episode/total_raised":             extra.get("total_raised", 0),

        # ── Final product state ────────────────────────────────────────────────
        "episode/final_pmf":                extra.get("final_pmf", 0),
        "episode/final_tech_debt":          1 if extra.get("tech_debt") in ("high", "critical") else 0,
        "episode/tech_debt_severity":       extra.get("tech_debt", "low"),

        # ── Final team state ───────────────────────────────────────────────────
        "episode/final_morale":             extra.get("final_morale", 0),

        # ── Final unit economics ───────────────────────────────────────────────
        "episode/final_ltv_cac":            extra.get("final_ltv_cac", 0),

        # ── Final founder relationship ─────────────────────────────────────────
        "episode/final_founder_trust":      extra.get("final_trust", 0),

        # ── Balanced scorecard (composite score across all dimensions) ─────────
        "episode/balanced_score":           _compute_balanced_score(
                                                survived=survived,
                                                final_runway=final_runway,
                                                final_pmf=extra.get("final_pmf", 0),
                                                final_morale=extra.get("final_morale", 0.5),
                                                final_ltv_cac=extra.get("final_ltv_cac", 1.0),
                                                final_trust=extra.get("final_trust", 0.5),
                                                milestones_hit=milestones_hit,
                                                pivot_timing_score=pivot_timing_score,
                                            ),
        "env/episode":                      episode,
    }
    _wandb.log(log)


def _compute_balanced_score(
    survived: bool,
    final_runway: int,
    final_pmf: float,
    final_morale: float,
    final_ltv_cac: float,
    final_trust: float,
    milestones_hit: int,
    pivot_timing_score: float,
) -> float:
    """
    Composite score across all 6 dimensions of the CoFounder environment.
    Each dimension contributes 0-100 points; max = 600.
    Normalized to 0-100.
    """
    score = 0.0

    # Survival (0-100)
    score += 100 if survived else (final_runway * 5)  # partial credit for runway

    # Product health (0-100)
    score += final_pmf * 100

    # Team health (0-100)
    score += final_morale * 100

    # Unit economics (0-100)
    ratio_score = min(final_ltv_cac / 5.0, 1.0) * 100   # 5.0 = perfect
    score += ratio_score

    # Founder relationship (0-100)
    score += final_trust * 100

    # Business performance (0-100)
    score += milestones_hit * 33   # capped at ~100 for 3 milestones
    score += pivot_timing_score * 30

    return round(min(score / 6.0, 100.0), 1)


def finish():
    if _enabled and _wandb:
        _wandb.finish()
