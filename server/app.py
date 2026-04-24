"""
FastAPI entry point — uses openenv-core create_app for the RL interface,
then adds extra routes for the web UI, scenario support, and prompt endpoint.
"""
import json
import os
import pathlib
import sys
from fastapi import Query
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# Make project root importable (needed when running from Docker WORKDIR)
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from openenv.core.env_server.http_server import create_app

try:
    from models import PivotAction, PivotObservation
    from server.pivot_environment import ThePivotEnvironment
    from server.prompt_encoder import encode_observation
except ModuleNotFoundError:
    from models import PivotAction, PivotObservation          # type: ignore
    from server.pivot_environment import ThePivotEnvironment  # type: ignore
    from server.prompt_encoder import encode_observation       # type: ignore

import server.wandb_logger as wlog
from pydantic import BaseModel
from typing import Optional

SCENARIOS_DIR = pathlib.Path(__file__).parent.parent / "scenarios"
STATIC_DIR    = pathlib.Path(__file__).parent.parent / "static"

# ── W&B init ──────────────────────────────────────────────────────────────
if os.getenv("WANDB_API_KEY"):
    wlog.init(
        project=os.getenv("WANDB_PROJECT", "models-nexica-ai"),
        run_name=os.getenv("WANDB_RUN_NAME"),
        config={"env_version": "0.1.0", "max_steps": 60},
    )

# ── Shared env instance for the UI (separate from WebSocket sessions) ─────
_ui_env: ThePivotEnvironment | None = None
_last_obs: PivotObservation | None = None

# ── In-memory leaderboard (resets on server restart) ─────────────────────
_leaderboard: list[dict] = []


# ── Request models ─────────────────────────────────────────────────────────
class AdvisorRequest(BaseModel):
    mrr: float                    # Monthly recurring revenue (USD)
    burn: float                   # Monthly burn (USD)
    runway: int                   # Months of runway remaining
    nps: int                      # Net Promoter Score
    churn: float                  # Monthly churn rate (0–1)
    step: Optional[int] = 30      # Which month you're in
    name: Optional[str] = "Founder"  # Name for leaderboard

class LeaderboardEntry(BaseModel):
    name: str
    scenario: str
    total_reward: float
    survived: bool
    pivot_step: Optional[int] = None
    n_steps: int

class CounterfactualRequest(BaseModel):
    scenario: str = "b2c_saas"
    seed: int = 42
    pivot_at_step: int            # Simulate PIVOT at this specific step
    n_steps_ahead: int = 20       # How many steps to simulate after the pivot


def _load_scenario(name: str | None) -> dict | None:
    if not name or name == "default":
        return None
    path = SCENARIOS_DIR / f"{name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ── Create openenv-core app (handles /reset /step /state /ws /schema) ─────
app = create_app(
    ThePivotEnvironment,
    PivotAction,
    PivotObservation,
    env_name="the_pivot",
    max_concurrent_envs=4,
)

# ── Serve static files ────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Root + health ─────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    """Redirect root to the dashboard."""
    return RedirectResponse(url="/ui")


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"status": "ok", "service": "the-pivot", "env": "ThePivotEnvironment"}


@app.get("/debug/routes", include_in_schema=False)
def debug_routes():
    """List all registered routes — diagnostic for deployment issues."""
    return {"routes": sorted([getattr(r, "path", str(r)) for r in app.routes])}


# ── UI routes ─────────────────────────────────────────────────────────────
@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def serve_ui():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/ui/reset")
def ui_reset(scenario: str = Query(default="default")):
    global _ui_env, _last_obs
    sc = _load_scenario(scenario)
    _ui_env = ThePivotEnvironment(scenario=sc)
    obs = _ui_env.reset()
    _last_obs = obs
    return obs.model_dump()


@app.post("/ui/step")
def ui_step(action: PivotAction):
    global _last_obs
    if _ui_env is None:
        return {"error": "Call /ui/reset first"}
    obs = _ui_env.step(action)
    _last_obs = obs
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
        "info": obs.metadata,
    }


@app.get("/prompt")
def get_prompt():
    """Returns the current observation as the text the LLM would read."""
    if _last_obs is None:
        return {"prompt": "No observation yet. Call /reset first."}
    return {"prompt": encode_observation(_last_obs)}


@app.get("/scenarios")
def list_scenarios():
    """List all available scenario names."""
    names = [f.stem for f in SCENARIOS_DIR.glob("*.json")]
    return {"scenarios": sorted(names)}


@app.get("/compare")
def compare_baselines(scenario: str = Query(default="b2c_saas"), n_episodes: int = Query(default=10)):
    """
    Run the 3 baseline agents on one scenario and return their stats.
    Used by the Compare tab in the UI.
    """
    import sys, os
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    os.environ["WANDB_MODE"] = "disabled"
    try:
        from training.baseline_agent import RandomAgent, StubbornAgent, PanicAgent, run_episodes
    except ImportError:
        from training.baseline_agent import RandomAgent, StubbornAgent, PanicAgent, run_episodes  # type: ignore

    sc = _load_scenario(scenario)
    if sc is None:
        return {"error": f"Scenario '{scenario}' not found"}

    n_episodes = max(1, min(n_episodes, 50))
    agents = [RandomAgent(), StubbornAgent(), PanicAgent()]
    results = []
    for agent in agents:
        r = run_episodes(agent, sc, n_episodes)
        results.append(r)
    return {"scenario": scenario, "n_episodes": n_episodes, "results": results}


@app.post("/advisor")
def advisor_endpoint(req: AdvisorRequest):
    """
    Advisor mode: user pastes their real startup metrics, gets a rule-based
    strategic recommendation + reasoning. No LLM required server-side.
    """
    rec, reasoning = _rule_based_advice(req)
    return {
        "recommendation": rec,
        "reasoning": reasoning,
        "metrics": req.model_dump(),
    }


def _rule_based_advice(req: AdvisorRequest) -> tuple[str, str]:
    """Deterministic decision tree that mimics what a trained founder agent does."""
    burn_ratio  = req.burn / max(req.mrr, 1)
    growth_proxy = req.nps / 100.0  # rough proxy for traction quality

    if req.runway <= 2:
        return "SELL", (
            f"With only {req.runway} months of runway, you are in survival territory. "
            "Consider an acqui-hire or strategic sale before reaching zero. "
            "Investors won't write checks at this stage without traction proof."
        )
    if req.runway <= 4 and burn_ratio > 2.5:
        return "CUT_COSTS", (
            f"Burn is {burn_ratio:.1f}x revenue with {req.runway}mo runway. "
            "Cut aggressively to extend runway before fundraising or pivoting."
        )
    if req.churn > 0.20 and req.nps < 10:
        return "PIVOT", (
            f"Churn of {req.churn:.0%} combined with NPS of {req.nps} signals product-market fit failure. "
            "The market is telling you something. A strategic pivot is likely necessary."
        )
    if req.churn > 0.12 and req.step >= 30:
        return "RESEARCH", (
            f"Churn at {req.churn:.0%} is elevated and you're past month {req.step}. "
            "Invest in a research sprint to clarify which customer segment is churning and why."
        )
    if req.runway < 8 and req.nps > 30:
        return "FUNDRAISE", (
            f"NPS of {req.nps} shows traction but runway is {req.runway}mo. "
            "This is a reasonable moment to raise — investors like product signal backed by urgency."
        )
    if req.nps > 40 and growth_proxy > 0.4 and burn_ratio < 2.0:
        return "HIRE", (
            f"Strong NPS ({req.nps}), decent unit economics (burn {burn_ratio:.1f}x). "
            "This is a growth-mode signal. A strategic hire could accelerate velocity."
        )
    return "EXECUTE", (
        f"Metrics are mixed but not alarming. Runway: {req.runway}mo, NPS: {req.nps}, "
        f"churn: {req.churn:.0%}. Stay the course and monitor closely."
    )


@app.get("/leaderboard")
def get_leaderboard():
    """Return top 20 scores."""
    sorted_board = sorted(_leaderboard, key=lambda x: x["total_reward"], reverse=True)
    return {"leaderboard": sorted_board[:20], "total_entries": len(_leaderboard)}


@app.post("/leaderboard/submit")
def submit_leaderboard(entry: LeaderboardEntry):
    """Submit a score to the leaderboard."""
    _leaderboard.append(entry.model_dump())
    rank = sorted(_leaderboard, key=lambda x: x["total_reward"], reverse=True).index(entry.model_dump()) + 1
    return {"status": "submitted", "rank": rank, "total": len(_leaderboard)}


@app.post("/counterfactual")
def counterfactual_endpoint(req: CounterfactualRequest):
    """
    Counterfactual replay: simulate 'what if I had PIVOTed at step X?'
    Runs the environment from scratch with a forced PIVOT at pivot_at_step,
    then EXECUTEs for n_steps_ahead, and returns the trajectory.
    """
    from models import ActionType, PivotAction

    sc = _load_scenario(req.scenario)
    if sc is None:
        return {"error": f"Scenario '{req.scenario}' not found"}

    env = ThePivotEnvironment(scenario=sc, rng_seed=req.seed)
    obs = env.reset()
    trajectory = []
    total_reward = 0.0

    for step in range(min(req.pivot_at_step + req.n_steps_ahead, 60)):
        if step == req.pivot_at_step:
            action_type = ActionType.PIVOT
        else:
            action_type = ActionType.EXECUTE

        obs = env.step(PivotAction(action_type=action_type))
        total_reward += obs.reward or 0
        trajectory.append({
            "step": step + 1,
            "action": action_type.value,
            "reward": obs.reward,
            "runway": obs.runway_remaining,
            "revenue": round(obs.monthly_revenue),
            "churn": round(obs.churn_rate, 3),
            "done": obs.done,
        })
        if obs.done:
            break

    return {
        "scenario": req.scenario,
        "pivot_at_step": req.pivot_at_step,
        "total_reward": round(total_reward, 1),
        "survived": obs.runway_remaining > 0,
        "final_runway": obs.runway_remaining,
        "trajectory": trajectory,
    }


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
