"""
FastAPI entry point — uses openenv-core create_app for the RL interface,
then adds extra routes for the web UI, scenario support, and prompt endpoint.
"""
import json
import os
import pathlib
import sys
import math
from fastapi import Query
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# Make project root importable (needed when running from Docker WORKDIR)
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from openenv.core.env_server.http_server import create_app

try:
    from models import CoFounderAction, CoFounderObservation
    # Backward-compat aliases
    PivotAction      = CoFounderAction
    PivotObservation = CoFounderObservation
    from server.cofounder_environment import CoFounderEnvironment as ThePivotEnvironment
    from server.prompt_encoder import encode_observation
except ModuleNotFoundError:
    from models import CoFounderAction as PivotAction, CoFounderObservation as PivotObservation  # type: ignore
    from server.pivot_environment import ThePivotEnvironment  # type: ignore
    from server.prompt_encoder import encode_observation       # type: ignore

import server.wandb_logger as wlog
from pydantic import BaseModel
from typing import Optional, List
import threading

SCENARIOS_DIR = pathlib.Path(__file__).parent.parent / "scenarios"
STATIC_DIR    = pathlib.Path(__file__).parent.parent / "static"

# ── W&B init ──────────────────────────────────────────────────────────────
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "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")
if WANDB_API_KEY:
    try:
        import wandb
        wandb.login(key=WANDB_API_KEY)
        wlog.init(
            project=os.getenv("WANDB_PROJECT", "models-nexica-ai"),
            run_name=os.getenv("WANDB_RUN_NAME"),
            config={"env_version": "0.1.0", "max_steps": 60},
        )
    except Exception as e:
        print(f"[wandb] Login/Init failed: {e}")

# ── Shared env instance for the UI (separate from WebSocket sessions) ─────
_ui_env: ThePivotEnvironment | None = None
_last_obs: PivotObservation | None = None

# ── In-memory leaderboard (resets on server restart) ─────────────────────
_leaderboard: list[dict] = []

# ── Trained model (loaded lazily if MODEL_ID env var is set) ──────────────
_chat_model      = None
_chat_tokenizer  = None
_model_status    = "not_configured"   # not_configured | loading | ready | error
_model_error     = ""

SYSTEM_PROMPT = """You are a Strategist Co-Founder AI trained with reinforcement learning to navigate multi-dimensional startup challenges. You help founders manage Product, Team, Marketing, Finance, and Market simultaneously.

When given startup metrics, reply with RECOMMENDATION: followed by exactly one action word.
Valid actions: EXECUTE | PIVOT | RESEARCH | FUNDRAISE | HIRE | CUT_COSTS | SELL | LAUNCH_FEATURE | MARKETING_CAMPAIGN | SET_PRICING | FIRE | PARTNERSHIP

Give 2-3 sentences of clear strategic reasoning citing specific numbers. Be direct.

Actions:
- EXECUTE: stay the course
- PIVOT: change product direction (costs 3 months runway)
- RESEARCH: reduce signal noise before deciding
- FUNDRAISE: raise from investors (check milestones first)
- HIRE: add headcount — raises burn, may boost velocity
- CUT_COSTS: reduce burn — survival mode
- SELL: acqui-hire exit
- LAUNCH_FEATURE: ship product feature — improves PMF or adds tech debt
- MARKETING_CAMPAIGN: run campaign — only effective if PMF > 40%
- SET_PRICING: adjust pricing tier
- FIRE: layoff — saves burn but causes 3-month morale hangover
- PARTNERSHIP: channel deal — reduces CAC, boosts pipeline in 2 months"""


def _load_model_background(model_id: str):
    """Load the trained LoRA model in a background thread so server stays responsive."""
    global _chat_model, _chat_tokenizer, _model_status, _model_error
    try:
        _model_status = "loading"
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        base_id = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
        device  = "cuda" if __import__("torch").cuda.is_available() else "cpu"

        _chat_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        _chat_tokenizer.pad_token = _chat_tokenizer.eos_token

        load_kwargs = {"device_map": "auto", "trust_remote_code": True}
        if device == "cuda":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=__import__("torch").bfloat16,
            )
        else:
            load_kwargs["torch_dtype"] = __import__("torch").float32

        base = AutoModelForCausalLM.from_pretrained(base_id, **load_kwargs)
        _chat_model   = PeftModel.from_pretrained(base, model_id)
        _chat_model.eval()
        _model_status = "ready"
    except Exception as e:
        _model_status = "error"
        _model_error  = str(e)


# Start loading model in background if MODEL_ID env var is set
_MODEL_ID = os.getenv("MODEL_ID", "")
if _MODEL_ID:
    threading.Thread(target=_load_model_background, args=(_MODEL_ID,), daemon=True).start()


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

class DemoRequest(BaseModel):
    scenario: str = "b2c_saas"
    seed: int = 42

class ChatMessage(BaseModel):
    role: str    # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []


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
    CoFounderAction if 'CoFounderAction' in dir() else PivotAction,
    CoFounderObservation if 'CoFounderObservation' in dir() else PivotObservation,
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


# ── Real-data endpoints (drive the dashboard) ─────────────────────────────
@app.get("/api/metrics")
def api_metrics():
    """Return real training metrics from docs/plots/metrics.json + run summary."""
    metrics_path = _PROJECT_ROOT / "docs" / "plots" / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            metrics = {}
    return {
        "metrics": metrics,
        "model_status": _model_status,
        "model_id": _MODEL_ID or None,
        "leaderboard_size": len(_leaderboard),
    }


@app.get("/api/scenario/{name}")
def api_scenario_detail(name: str):
    """Return the full scenario JSON for the Environment tab."""
    sc = _load_scenario(name)
    if sc is None:
        return {"error": f"Scenario '{name}' not found"}
    return sc


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
        from training.baseline_agent import RandomAgent, StubbornAgent, PanicAgent, StrategistAgent, run_episodes
    except ImportError:
        from training.baseline_agent import RandomAgent, StubbornAgent, PanicAgent, StrategistAgent, run_episodes  # type: ignore

    sc = _load_scenario(scenario)
    if sc is None:
        return {"error": f"Scenario '{scenario}' not found"}

    n_episodes = max(1, min(n_episodes, 50))
    agents = [RandomAgent(), StubbornAgent(), PanicAgent(), StrategistAgent()]
    results = []
    for agent in agents:
        r = run_episodes(agent, sc, n_episodes)
        results.append(r)
    return {"scenario": scenario, "n_episodes": n_episodes, "results": results}


def _downsample_points(points: list[dict], max_points: int) -> list[dict]:
    if len(points) <= max_points:
        return points
    stride = max(1, math.ceil(len(points) / max_points))
    sampled = points[::stride]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled[:max_points]


@app.get("/wandb/history")
def wandb_history(
    entity: str | None = Query(default=None),
    project: str | None = Query(default=None),
    run_id: str | None = Query(default=None),
    metric_primary: str = Query(default="episode_reward"),
    metric_secondary: str = Query(default="survival_rate"),
    max_points: int = Query(default=200, ge=20, le=1000),
):
    """Fetch run history from Weights & Biases for dashboard analytics."""
    try:
        import wandb
    except Exception as exc:
        return {"error": f"wandb import failed: {exc}"}

    resolved_entity = entity or os.getenv("WANDB_ENTITY")
    resolved_project = project or os.getenv("WANDB_PROJECT", "models-nexica-ai")
    if not resolved_entity or not resolved_project:
        return {
            "error": "Missing W&B entity/project. Set WANDB_ENTITY and WANDB_PROJECT env vars or pass query params.",
            "entity": resolved_entity,
            "project": resolved_project,
        }

    try:
        api = wandb.Api()
        if run_id:
            run = api.run(f"{resolved_entity}/{resolved_project}/{run_id}")
        else:
            runs = api.runs(path=f"{resolved_entity}/{resolved_project}", per_page=1, order="-created_at")
            if not runs:
                return {
                    "error": "No runs found for project",
                    "entity": resolved_entity,
                    "project": resolved_project,
                }
            run = runs[0]

        keys = ["_step", metric_primary]
        if metric_secondary and metric_secondary != metric_primary:
            keys.append(metric_secondary)

        raw_points: list[dict] = []
        for row in run.scan_history(keys=keys):
            step = row.get("_step")
            if step is None:
                continue
            raw_points.append(
                {
                    "step": int(step),
                    "primary": row.get(metric_primary),
                    "secondary": row.get(metric_secondary) if metric_secondary else None,
                }
            )

        points = _downsample_points(raw_points, max_points=max_points)
        return {
            "entity": resolved_entity,
            "project": resolved_project,
            "run_id": getattr(run, "id", run_id),
            "run_name": getattr(run, "name", "unknown"),
            "metric_primary": metric_primary,
            "metric_secondary": metric_secondary,
            "points": points,
            "total_points": len(raw_points),
            "available_summary_keys": sorted(list((run.summary or {}).keys()))[:100],
            "wandb_url": getattr(run, "url", None),
        }
    except Exception as exc:
        return {
            "error": str(exc),
            "entity": resolved_entity,
            "project": resolved_project,
            "run_id": run_id,
        }


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


@app.post("/demo")
def demo_endpoint(req: DemoRequest):
    """
    Run a full 60-step episode using the rule-based advisor as the agent.
    Returns step-by-step trajectory: action taken, reasoning, reward, runway, revenue.
    Used by the Live Demo tab in the UI.
    """
    from models import ActionType

    sc = _load_scenario(req.scenario)
    if sc is None:
        return {"error": f"Scenario '{req.scenario}' not found"}

    env = ThePivotEnvironment(scenario=sc, rng_seed=req.seed)
    obs = env.reset()
    trajectory = []
    total_reward = 0.0
    pivot_steps = []

    for step in range(60):
        # Build advisor request from current observation
        adv_req = AdvisorRequest(
            mrr=obs.monthly_revenue,
            burn=obs.burn_rate,
            runway=obs.runway_remaining,
            nps=obs.nps_score,
            churn=obs.churn_rate,
            step=step,
        )
        action_str, reasoning = _rule_based_advice(adv_req)

        obs = env.step(PivotAction(action_type=ActionType(action_str)))
        reward = obs.reward or 0.0
        total_reward += reward

        if action_str == "PIVOT":
            pivot_steps.append(step + 1)

        trajectory.append({
            "step":      step + 1,
            "action":    action_str,
            "reasoning": reasoning[:80],   # truncate for UI display
            "reward":    round(reward, 1),
            "runway":    obs.runway_remaining,
            "revenue":   round(obs.monthly_revenue),
            "nps":       obs.nps_score,
            "churn":     round(obs.churn_rate * 100, 1),
            "done":      obs.done,
        })
        if obs.done:
            break

    return {
        "scenario":     req.scenario,
        "seed":         req.seed,
        "total_reward": round(total_reward, 1),
        "survived":     obs.runway_remaining > 0,
        "steps":        len(trajectory),
        "pivot_steps":  pivot_steps,
        "trajectory":   trajectory,
    }


@app.get("/model/status")
def model_status():
    """Check whether the trained LoRA model is loaded and ready for chat."""
    return {
        "status":     _model_status,
        "model_id":   _MODEL_ID or None,
        "error":      _model_error or None,
        "configured": bool(_MODEL_ID),
    }


@app.get("/chat", response_class=HTMLResponse, include_in_schema=False)
def chat_page():
    """Serve the standalone chat page."""
    return FileResponse(str(STATIC_DIR / "chat.html"))


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    """
    Chat with the trained startup advisor model.
    Requires MODEL_ID env var set and model loaded (check /model/status first).
    Maintains multi-turn conversation history.
    """
    if _model_status == "not_configured":
        return {
            "response": "⚠️ Model not configured. Set the MODEL_ID environment variable in your HF Space secrets to enable this feature.",
            "action": None, "ready": False,
        }
    if _model_status == "loading":
        return {
            "response": "⏳ Model is still loading (this takes 1-2 minutes on first start). Try again shortly.",
            "action": None, "ready": False,
        }
    if _model_status == "error":
        return {
            "response": f"❌ Model failed to load: {_model_error}",
            "action": None, "ready": False,
        }

    import torch

    # Build message list for chat template
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in req.history[-6:]:          # last 3 turns (6 messages)
        messages.append({"role": h.role, "content": h.content})
    messages.append({"role": "user", "content": req.message})

    text = _chat_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _chat_tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=1024).to(_chat_model.device)

    with torch.no_grad():
        out = _chat_model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=_chat_tokenizer.eos_token_id,
        )
    response = _chat_tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    # Extract action word from first line
    first_word = response.split()[0].upper().rstrip(".,!:") if response.split() else ""
    valid_actions = {
        "EXECUTE", "PIVOT", "RESEARCH", "FUNDRAISE", "HIRE", "CUT_COSTS", "SELL",
        "LAUNCH_FEATURE", "MARKETING_CAMPAIGN", "SET_PRICING", "FIRE", "PARTNERSHIP",
    }
    # Also handle "RECOMMENDATION: ACTION" format
    for line in response.split("\n"):
        if line.strip().upper().startswith("RECOMMENDATION:"):
            first_word = line.split(":", 1)[1].strip().upper().rstrip(".,!:")
            break
    action = first_word if first_word in valid_actions else None

    return {"response": response, "action": action, "ready": True}


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
