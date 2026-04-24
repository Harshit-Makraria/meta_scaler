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


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
