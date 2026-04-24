# The Pivot — Codebase Guide

A plain-English walkthrough of every file and what it does.

---

## How It All Fits Together

```
Agent (LLM)
    │
    │  step(action) → observation
    ▼
ThePivotEnv (client.py)           ← what your training code imports
    │  WebSocket
    ▼
FastAPI server (server/app.py)    ← runs on localhost:8000
    │
    ▼
ThePivotEnvironment               ← the brain — calls everything below
    ├── MarketSimulator           ← hidden truth: which phase is the market in?
    ├── SignalGenerator           ← adds noise so agent can't see truth directly
    ├── FounderAgent              ← Ghost Protocol: advice gets worse as money runs out
    ├── InvestorAgent             ← VC who silently changes what they want
    ├── RunwayTracker             ← tracks cash, revenue, burn rate
    └── RewardCalculator          ← scores the agent's decisions
```

---

## Root Files

### `openenv.yaml`
The project's ID card for the hackathon framework. Tells OpenEnv:
- What this environment is called (`the-pivot`)
- What class runs it (`ThePivotEnvironment`)
- How many steps per episode (60 = 60 months)
- Tags for judging themes

### `pyproject.toml`
Standard Python packaging config. Lists all dependencies.
Run `pip install -e .` to install in dev mode.

### `models.py`
**The shared data language** between agent and environment.

Two classes:
- **`PivotAction`** — what the agent sends each step. Has `action_type` (one of 6 choices) and optional `action_params`.
- **`PivotObservation`** — what the agent receives back. Has 15 fields covering financial state, noisy market signals, and founder/investor state.

Both inherit from `openenv-core` base classes (`Action`, `Observation`) so the framework can auto-validate and serialize them over WebSocket.

Key rule: **`PivotObservation` has `.reward` and `.done` embedded** — this is the openenv-core pattern (not a separate tuple).

### `client.py`
What training code imports to talk to the running server.

```python
async with ThePivotEnv("http://localhost:8000") as env:
    obs = await env.reset()
    obs = await env.step(PivotAction(action_type="PIVOT"))
```

Uses WebSocket (persistent connection) for low latency. For sync usage: `env.sync()`.

---

## `server/` — The Environment Engine

### `server/app.py`
The web server entry point.

Uses `openenv.core.env_server.http_server.create_app()` which auto-generates:
- `POST /reset` — start new episode
- `POST /step` — take one action
- `GET /state` — get full internal state (debug)
- `GET /schema` — shows action + observation schemas as JSON
- `GET /health` — server health check
- `WS /ws` — WebSocket for persistent agent sessions (lower latency)
- `GET /mcp` — MCP tool interface (bonus for judges)

Also initializes W&B on startup if `WANDB_API_KEY` is in environment.

**To start the server:**
```bash
uvicorn server.app:app --reload --port 8000
```

---

### `server/market.py`
**The hidden world — what the agent must figure out without being told.**

Three market phases:
| Phase | Steps | Revenue growth | Churn drift | Competitor activity |
|-------|-------|---------------|-------------|---------------------|
| GROWTH | 0–20 | +8%/month | low | 5% chance |
| SATURATION | 21–35 | +1%/month | medium | 25% chance |
| DECLINE | 36–60 | -4%/month | high | 55% chance |

The agent **never sees the phase directly**. It has to infer it from noisy signals.

Key methods:
- `get_phase(step)` — returns the true phase (used internally, never sent to agent)
- `get_config(step)` — returns the PhaseConfig (growth rates, complaint distributions)
- `check_pivot_necessity(step, pivot_step)` — was this pivot the right call?
- `get_optimal_pivot_window()` — returns `(39, 46)`: the best time to pivot

---

### `server/signals.py`
**The noise layer — makes the agent's job hard and realistic.**

Takes the true market state and adds realistic noise before the agent sees it:
- Revenue: ±15% Gaussian noise
- Churn: ±0.08 noise
- NPS: ±12 integer noise
- Complaints: probability distribution shifts with market phase but noisily
- Competitor events: 85% chance of seeing a real event, 5% false positive

When agent takes `RESEARCH` action → noise drops to 30% for 3 steps.

The complaint distribution is the biggest signal:
- GROWTH → mostly "missing feature", "slow performance"
- SATURATION → "too expensive", "competitor is better"
- DECLINE → "switching to X", "competitor is better"

---

### `server/founder.py`
**Ghost Protocol mechanic — the mentor who becomes unreliable.**

The founder starts giving good advice (85% accuracy) but decays as money runs low.

`decay_level` (hidden from agent):
- 0.0 at start (18 months runway)
- ~0.5 at 6 months runway
- 1.0 at 0 months runway

What the agent CAN observe:
- `founder_confidence` — drops with decay + noise
- `founder_desperation_signal` — noisy proxy for decay

What the agent CANNOT observe:
- `decay_level` (the raw number)

When `decay > 0.7`, founder defaults to "stay_course" even when market is clearly declining. The agent must learn to **detect when to stop listening to the founder** — this is the Ghost Protocol skill.

---

### `server/runway.py`
**The financial scoreboard.**

Tracks three numbers:
- `monthly_revenue` — how much the startup earns each month
- `burn_rate` — how much the startup spends each month
- `cash` — total cash in the bank

`runway_remaining` = `cash / (burn_rate - monthly_revenue)` — capped at 999 when profitable.

Effect of each action on finances:
| Action | Financial effect |
|--------|-----------------|
| EXECUTE | Revenue grows by market rate |
| PIVOT | -3 months runway, revenue resets to 60% |
| RESEARCH | -0.5 months runway |
| FUNDRAISE | +$500k / +$2M / +$5M (if approved) |
| HIRE | Burn +$20k/month |
| CUT_COSTS | Burn -$30k/month, revenue -20% |

---

### `server/investor.py`
**The VC who silently changes the rules.**

Three funding rounds with different (hidden) criteria:
| Round | Milestone shown to agent | Secret criteria |
|-------|--------------------------|-----------------|
| Seed (steps 0–20) | "Show 10% MoM revenue growth" | `revenue_delta_3m >= 0.10` |
| Series A (steps 21–40) | "3+ months runway and positive NPS" | `runway >= 3` AND `nps >= 0` |
| Series B (steps 41–60) | "Clear path to profitability" | `revenue - burn > -$20k` |

The milestone description is visible to the agent. The exact criteria are not.

`investor_sentiment` (0–1) rises when milestones are hit, drops when missed.

---

### `server/reward.py`
**How the agent is scored — 5 components.**

| Component | When | Amount |
|-----------|------|--------|
| Survival | Every step | -1 (time pressure) |
| Survival | Runway hits 0 | -200 (death) |
| Survival | Survived 60 steps | +150 |
| Growth | Per step | ±(revenue change %) × 50 |
| Pivot Timing | On PIVOT action | +50 × timing_score (0–1) if necessary, -20 if unnecessary |
| Milestone | On FUNDRAISE | +100 if approved |
| Founder Awareness | On any action | +30 if correctly overrode panicking founder |

The **pivot timing rubric** is the core learning signal. It peaks at 1.0 if the agent pivots right when DECLINE starts (step 39) and decays by 0.09 per step late.

---

### `server/wandb_logger.py`
**Tracks everything to your W&B dashboard.**

Two streams:
- **Per-step** (`step/reward`, `step/runway_remaining`, `step/true_phase`, etc.)
- **Per-episode** (`episode/total_reward`, `episode/survived`, `episode/pivot_timing_score`, etc.)

Activated by setting `WANDB_API_KEY` in `.env`. If the key is missing, W&B silently disables itself — the environment still works normally.

Dashboard: https://wandb.ai/models-nexica-ai

---

### `server/pivot_environment.py`
**The main file — the game engine.**

Connects all subsystems. Inherits from `openenv.core.env_server.interfaces.Environment`.

Three methods the framework calls:
- `reset(seed, episode_id)` → initializes all subsystems, returns initial observation
- `step(action)` → applies action, advances world 1 month, returns observation with reward embedded
- `state` (property) → returns full internal state for debugging

Internal flow of `step()`:
1. Apply action effects to runway/churn/NPS
2. Advance market (revenue grows/shrinks by phase rate, churn drifts, NPS drifts)
3. Roll for competitor event
4. Compute reward across all 5 rubrics
5. Log to W&B
6. If done: log episode summary to W&B, increment episode counter
7. Build observation (add noise to true state) and return it

---

## `.env` file (never commit this)
Contains secrets:
```
WANDB_API_KEY=...       ← your W&B key
WANDB_PROJECT=models-nexica-ai
WANDB_RUN_NAME=...      ← optional, auto-generated if omitted
```

Copy `.env.example` to `.env` and fill in values.

---

## Running Locally
```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Set environment variables
cp .env.example .env
# edit .env with your WANDB_API_KEY

# 3. Start the server
uvicorn server.app:app --reload --port 8000

# 4. Open interactive docs
# http://localhost:8000/docs

# 5. Or run a quick test episode
python -c "
from dotenv import load_dotenv; load_dotenv()
import sys; sys.path.insert(0, '.')
from server.pivot_environment import ThePivotEnvironment
from models import PivotAction, ActionType
env = ThePivotEnvironment()
obs = env.reset()
print('Runway:', obs.runway_remaining, 'months')
"
```
