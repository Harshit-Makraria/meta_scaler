# Claude Context — The Pivot Project

This doc is for Claude Code. Read this at the start of any session on this project.

---

## What This Project Is

**The Pivot** — an OpenEnv-compliant RL environment for the Meta PyTorch OpenEnv Hackathon 2026 (Scaler x PyTorch, Bangalore finale April 25–26).

It's a startup founder simulation where an LLM agent must detect hidden market phase shifts (without being told) and decide when to pivot strategy. Built for GRPO training with HuggingFace TRL.

Hackathon page: https://www.scaler.com/school-of-technology/meta-pytorch-hackathon
Framework repo: https://github.com/meta-pytorch/OpenEnv

---

## Framework: openenv-core (NOT the other "openenv" package)

- **Package**: `openenv-core` v0.2.3 — `pip install openenv-core`
- The `openenv` package (v0.1.13, by neo-derek) is a different unrelated package also installed — ignore it
- Key import paths: `openenv.core.env_server.interfaces.Environment`, `openenv.core.env_server.types.Action`, `openenv.core.env_server.types.Observation`, `openenv.core.EnvClient`

**Critical pattern**: `step()` returns just `Observation` — reward and done are embedded as `.reward` and `.done` fields on the observation object. NOT a tuple.

---

## Current Status (as of session 2026-04-24)

### Built and Working ✅

**Core Environment**
- `models.py` — PivotAction, PivotObservation (both inherit openenv-core base classes). Now includes `competitor_play` and `competitor_strength` fields.
- `server/market.py` — MarketSimulator with 3 hidden phases, scenario-aware
- `server/signals.py` — SignalGenerator with noise + RESEARCH noise reduction
- `server/founder.py` — FounderAgent with Ghost Protocol decay
- `server/runway.py` — RunwayTracker (runway capped at 999 when profitable)
- `server/investor.py` — InvestorAgent with 3 funding rounds silently shifting at steps 20 and 40
- `server/reward.py` — RewardCalculator with 5 rubric components
- `server/competitor.py` ✅ NEW — CompetitorAgent (rule-based rival). Picks from: DORMANT, LAUNCH_FEATURE, PRICE_WAR, TALENT_RAID, AGGRESSIVE_MKT based on your NPS/churn/burn/runway signals. Grows stronger over time (strength 0.2→1.0). Applies revenue steal + burn increase each step.
- `server/pivot_environment.py` — ThePivotEnvironment (full openenv-core compatible). Competitor wired in: `tick()` each step, revenue/burn penalties applied.
- `server/app.py` — Uses `create_app()` + extra routes: `/ui`, `/ui/reset`, `/ui/step`, `/prompt`, `/scenarios`, `/compare` (NEW — runs 3 baseline agents server-side for UI comparison tab)
- `server/wandb_logger.py` — Per-step and per-episode W&B logging
- `server/prompt_encoder.py` — Converts PivotObservation → natural language. Now includes competitor play narration so LLM sees rival moves in plain English.
- `client.py` — ThePivotEnv inheriting EnvClient (WebSocket)

**Scenarios (5 JSON files in scenarios/)**
- `b2c_saas.json` — easy, decline_start=36, pivot_window=(39,46)
- `enterprise_saas.json` — medium, decline_start=41, pivot_window=(44,50)
- `fintech.json` — medium-hard, decline_start=33, pivot_window=(36,42)
- `marketplace.json` — hard, decline_start=29, pivot_window=(32,38)
- `consumer_app.json` — very hard, decline_start=23, pivot_window=(26,30)

**Training**
- `training/baseline_agent.py` — RandomAgent, StubbornAgent, PanicAgent + `run_episodes()`
- `training/evaluate.py` — full eval pipeline: baseline + optional trained model + matplotlib plots + W&B
- `training/curriculum.py` ✅ NEW — AdaptiveCurriculum. Trains on easy scenarios first, unlocks harder tiers when mean_reward > threshold AND survival_rate > 45%. 20% replay from already-unlocked tiers. 5 tiers: b2c_saas → enterprise_saas → fintech → marketplace → consumer_app.

**Web UI (static/index.html)**
- Full dark dashboard with financial, market signals, founder/investor cards
- Competitor intelligence card (play + strength gauge) in Market Signals
- Episode replay table (full step-by-step timeline with competitor column)
- Export replay as JSON button
- Compare tab: runs 3 baseline agents server-side, shows reward/survival/pivot-rate cards
- 6 action buttons, Chart.js reward + revenue/burn charts, LLM prompt display, scenario selector

**Docs**
- `docs/CODEBASE_GUIDE.md` — human-readable guide to all files
- `docs/CLAUDE_CONTEXT.md` — this file

---

## W&B Setup

- **Project**: `models-nexica-ai`
- **Entity**: `harshitmakraria9` (user's email: harshitmakraria9@gmail.com)
- **API Key**: in `.env` (never commit, in `.gitignore`)
- **Dashboard**: https://wandb.ai/models-nexica-ai/models-nexica-ai
- W&B auto-disables if key not set — env works without it

---

## Architecture Decisions (don't change without reason)

1. **step() returns Observation, not tuple** — openenv-core requirement. All callers must use `obs.reward` and `obs.done`.

2. **Competitor wiring** — `CompetitorAgent.tick(snapshot)` called each step BEFORE runway.step(). Revenue and burn penalties applied directly to runway tracker. Competitor strength grows 0.2→1.0 over 60 steps regardless of resets (resets on env.reset()).

3. **Episode seeding** — Each reset uses `seed + episode_count` so consecutive episodes are different but reproducible.

4. **Runway capped at 999** — When revenue > burn rate (profitable), runway would be infinite. We cap at 999 months.

5. **Reward info in metadata** — The 5-component breakdown dict goes in `obs.metadata["survival"]`, `obs.metadata["growth"]`, etc. Also `obs.metadata["true_phase"]` for training analysis.

6. **Adaptive curriculum** — `training/curriculum.py` must be used in the Colab training notebook. `curriculum.sample_scenario()` returns a scenario dict. Call `curriculum.record_result(mean_reward, survived)` after each episode, then `curriculum.advance_tier()` if `curriculum.should_advance()`.

7. **Compare endpoint** — `/compare?scenario=b2c_saas&n_episodes=10` runs the 3 baseline agents synchronously. Capped at 50 episodes max to avoid timeout. Called by the UI Compare tab.

---

## Key Numbers (don't change without updating docs)

| Constant | Value | File |
|----------|-------|------|
| MAX_STEPS | 60 | pivot_environment.py |
| INITIAL_REVENUE | $45,000/mo | pivot_environment.py |
| INITIAL_BURN | $120,000/mo | pivot_environment.py |
| INITIAL_RUNWAY | 18 months | pivot_environment.py |
| GROWTH phase | steps 0–20 | market.py |
| SATURATION phase | steps 21–35 | market.py |
| DECLINE phase | steps 36–60 | market.py |
| Optimal pivot window (default) | steps 39–46 | market.py |
| PIVOT cost | 3 months runway + revenue resets to 60% | pivot_environment.py |
| Competitor seed | rng_seed + 99 | pivot_environment.py |
| Curriculum window | 20 episodes | curriculum.py |
| Curriculum survival gate | 45% | curriculum.py |
| Seed funding check size | $500,000 | investor.py |
| Series A check size | $2,000,000 | investor.py |
| Series B check size | $5,000,000 | investor.py |

---

## Testing Quickly

```python
import sys; sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv()
from server.pivot_environment import ThePivotEnvironment
from models import PivotAction, ActionType

env = ThePivotEnvironment()
obs = env.reset()
obs = env.step(PivotAction(action_type=ActionType.EXECUTE))
print(obs.reward, obs.done, obs.runway_remaining)
print(obs.competitor_play, obs.competitor_strength)
```

Test curriculum:
```python
from training.curriculum import AdaptiveCurriculum
c = AdaptiveCurriculum()
sc = c.sample_scenario()
print(sc['name'], c.status())
```

Start server:
```bash
uvicorn server.app:app --reload --port 8000
```
Then visit:
- http://localhost:8000/ui — web dashboard
- http://localhost:8000/docs — auto API docs
- http://localhost:8000/compare?scenario=b2c_saas&n_episodes=10 — baseline comparison

---

## What's Still Needed (priority order)

1. **`training/train_colab.ipynb`** — GRPO training notebook using TRL + Qwen2.5-3B-Instruct + AdaptiveCurriculum. **CRITICAL for submission.**
2. **`README.md`** — Project overview for hackathon judges. **CRITICAL for submission.**
3. HF Spaces deployment (see answer in codebase guide)
