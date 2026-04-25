# 🔄 The Pivot

> **An OpenEnv-compliant reinforcement learning environment where an LLM agent plays a startup founder navigating hidden market phase shifts — and must decide when to pivot before it's too late.**

Built for the **Meta PyTorch OpenEnv Hackathon 2026** · Scaler × PyTorch · Bangalore · April 25–26

---

## 🔗 Quick Links

| | |
|---|---|
| 🚀 **Live Demo (HF Space)** | [harshit-makraria-the-pivot.hf.space/ui](https://harshit-makraria-the-pivot.hf.space/ui) |
| 📓 **Training Notebook (Colab)** | [Open in Colab](https://colab.research.google.com/github/Harshit-Makraria/meta_scaler/blob/main/training/train_colab.ipynb) |
| 📊 **W&B Dashboard** | [wandb.ai/models-nexica-ai](https://wandb.ai/models-nexica-ai/models-nexica-ai) |
| 🧩 **OpenEnv Manifest** | [`openenv.yaml`](openenv.yaml) |
| 📝 **Full Technical Writeup** | [`docs/DETAILS.md`](docs/DETAILS.md) |
| 📝 **Submission Writeup** | [`docs/WRITEUP.md`](docs/WRITEUP.md) |

---

## 🎯 What Is This?

Most LLM benchmarks test if an agent can **follow a plan**. The Pivot tests something harder: **can it detect that the plan is no longer working — and decide when to abandon it?**

An LLM agent plays the role of a startup co-founder across **60 simulated months**. The market silently moves through three hidden phases (Growth → Saturation → Decline) with timing that is **never revealed**. The agent must:

- Read **noisy, contradictory signals** (NPS, churn, CAC/LTV, competitor moves)
- Compete against a **rule-based rival** that adapts to your weaknesses
- Manage **investor expectations** across 3 funding rounds
- Decide **when** to PIVOT — not too early (burns runway), not too late (company dies)
- Survive **random macro shocks** (funding winters, viral moments, key hires quitting)

---

## 📊 Training Results

| Metric | Value |
|---|---|
| Algorithm | GRPO + ε-greedy exploration + KL penalty |
| Base model | Qwen/Qwen2.5-0.5B-Instruct |
| PEFT | LoRA r=8 (q_proj, v_proj) + 4-bit nf4 |
| Training hardware | Google Colab T4 (15GB VRAM) |
| Episodes | 150 |
| Mean reward (last 20 ep) | −0.5 |
| Best episode reward | +7.0 |
| Curriculum tier reached | 2 / 5 |

### Reward Curve
![Reward Curve](docs/plots/reward_curve.png)

### GRPO Loss Curve
![Loss Curve](docs/plots/loss_curve.png)

### Episode Survival
![Survival](docs/plots/survival_curve.png)

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    ThePivotEnvironment                         │
│                                                                │
│  MarketSimulator      SignalGenerator      CompetitorAgent     │
│  (3 hidden phases,    (noisy KPIs,         (5 strategies,      │
│   scenario-timed)      RESEARCH reduces     strength 0.2→1.0)  │
│                        noise level)                            │
│                                                                │
│  InvestorAgent        FounderAgent         RunwayTracker       │
│  (3 funding rounds,   (Ghost Protocol      (revenue/burn/cash, │
│   shift at 20 & 40)    morale decay)        cap 999mo)         │
│                                                                │
│  RewardCalculator — 5-component rubric                         │
│  ShockEngine — 6 macro events (random, unannounced)            │
└────────────────────────────────────────────────────────────────┘
          ↓  step() returns Observation (NOT a tuple)
┌────────────────────────────────────────────────────────────────┐
│                  FastAPI Server (openenv-core)                 │
│  /reset  /step  /state  /ws  /schema  (OpenEnv standard)      │
│  /ui  /prompt  /compare  /advisor  /leaderboard               │
│  /counterfactual  /healthz  /debug/routes                      │
└────────────────────────────────────────────────────────────────┘
          ↓
┌────────────────────────────────────────────────────────────────┐
│              GRPO Training (Colab T4)                          │
│  Qwen2.5-0.5B + LoRA r=8 + 4-bit nf4                         │
│  ε-greedy exploration (30%) + return-to-go advantages          │
│  KL penalty (β=0.04) + AdaptiveCurriculum (5 tiers)           │
└────────────────────────────────────────────────────────────────┘
```

---

## 🎮 The 7 Actions

| Action | Effect | Best used when |
|---|---|---|
| `EXECUTE` | Normal operations, slow revenue growth | Market is growing, signals green |
| `PIVOT` | −3mo runway, revenue resets to 60% | Decline confirmed, in optimal window |
| `RESEARCH` | Reduces signal noise for 3 steps | Signals contradictory / confusing |
| `FUNDRAISE` | Triggers investor round if conditions met | Runway < 8mo, metrics strong |
| `HIRE` | +$20k burn, +product velocity | Growth phase, budget available |
| `CUT_COSTS` | −$30k burn, −morale, −revenue 20% | Distress, runway crisis |
| `SELL` | Acqui-hire exit — ends episode | Runway ≤ 2mo, survival play |

---

## 🌍 5 Scenarios (Difficulty Ladder)

| Tier | Scenario | Difficulty | Decline starts | Optimal pivot window |
|------|-----------|-----------|---------------|---------------------|
| 1 | `b2c_saas` | Easy | Month 36 | Months 39–46 |
| 2 | `enterprise_saas` | Medium | Month 41 | Months 44–50 |
| 3 | `fintech` | Medium-Hard | Month 33 | Months 36–42 |
| 4 | `marketplace` | Hard | Month 29 | Months 32–38 |
| 5 | `consumer_app` | Very Hard | Month 23 | Months 26–30 |

The adaptive curriculum unlocks tier N+1 only when:
- 20-episode moving average reward > threshold, **AND**
- Survival rate ≥ 45%

---

## 🏆 Reward System (5 Components)

| Component | What it measures | Max contribution |
|---|---|---|
| **Survival** | Being alive each step; +150 for completing 60 months | ~+150 |
| **Growth** | Revenue growth rate above baseline | Variable |
| **Pivot timing** | Pivoting inside the optimal window | +50 |
| **Efficiency** | Burn rate control relative to revenue | Variable |
| **Founder awareness** | Correctly overriding a panicking founder | +30 per override |
| **Board pressure** | Decisive action after step 40 with low runway | ±15 |
| **Acqui-hire** | Graceful SELL at the right moment | +50 |
| **Shock survival** | Staying alive through a macro shock event | +5 |

---

## ⚡ Macro Shock Events

Random events that hit with no warning, mid-episode:

| Event | Probability | Effect |
|---|---|---|
| `funding_winter` | 4%/step in Saturation/Decline | Burn +25%, revenue −8% |
| `viral_moment` | 3%/step in Growth | Revenue +35%, NPS +10 |
| `key_engineer_quits` | 5%/step in Saturation/Decline | Morale −15%, revenue −3% |
| `competitor_acquired` | 3%/step in Decline | Competitor strength +25%, NPS −7 |
| `regulatory_change` | 3%/step in Saturation | Burn +15%, revenue −4% |
| `key_customer_churns` | 4%/step in Saturation/Decline | Revenue −12%, NPS −5 |

---

## 🛠️ Run Locally in 3 Commands

```bash
git clone https://github.com/Harshit-Makraria/meta_scaler
cd meta_scaler
pip install openenv-core fastapi uvicorn pydantic numpy python-dotenv
uvicorn server.app:app --port 8000
```

Then open:
- **Dashboard**: http://localhost:8000/ui
- **API docs**: http://localhost:8000/docs
- **Advisor**: POST http://localhost:8000/advisor
- **Leaderboard**: http://localhost:8000/leaderboard

---

## 🧪 Quick Python Test

```python
from server.pivot_environment import ThePivotEnvironment
from models import PivotAction, ActionType

env = ThePivotEnvironment()
obs = env.reset()
print(f"Month 1 — Runway: {obs.runway_remaining}mo, Revenue: ${obs.monthly_revenue:,.0f}")

for step in range(60):
    # Simple rule: PIVOT if NPS < 10 and churn > 20%
    if obs.nps_score < 10 and obs.churn_rate > 0.20:
        action = ActionType.PIVOT
    else:
        action = ActionType.EXECUTE
    obs = env.step(PivotAction(action_type=action))
    if obs.done:
        print(f"Done at month {obs.step} | reward: {obs.reward:.1f} | runway: {obs.runway_remaining}mo")
        if obs.active_shock:
            print(f"Shock: {obs.shock_message}")
        break
```

---

## 🏋️ Train Your Own Policy

Open [`training/train_colab.ipynb`](training/train_colab.ipynb) in Colab with a **T4 GPU**. Run all 12 cells top to bottom. ~60–90 minutes for 150 episodes.

**Key features of the training setup:**
- Qwen2.5-0.5B-Instruct (fits in 15GB T4 VRAM with 4-bit quantization)
- LoRA r=8 on q_proj + v_proj only
- GRPO with return-to-go advantages (60 steps per episode = the group)
- ε-greedy exploration (ε=0.3) — critical for diverse completions with a small base model
- KL penalty (β=0.04) against frozen reference model — prevents reward hacking
- AdaptiveCurriculum: starts on easiest scenario, unlocks harder ones as model improves
- W&B logging of every step + episode

---

## 🧠 Advisor Mode

The `/advisor` endpoint takes your **real startup metrics** and returns a strategic recommendation:

```bash
curl -X POST https://harshit-makraria-the-pivot.hf.space/advisor \
  -H "Content-Type: application/json" \
  -d '{"mrr": 45000, "burn": 120000, "runway": 6, "nps": 15, "churn": 0.18, "step": 35}'
```

Response:
```json
{
  "recommendation": "CUT_COSTS",
  "reasoning": "Burn is 2.7x revenue with 6mo runway. Cut aggressively to extend runway before fundraising or pivoting."
}
```

---

## 🔀 Counterfactual Replay

Simulate "what if I had pivoted at month X?":

```bash
curl -X POST https://harshit-makraria-the-pivot.hf.space/counterfactual \
  -H "Content-Type: application/json" \
  -d '{"scenario": "b2c_saas", "pivot_at_step": 38, "n_steps_ahead": 20, "seed": 42}'
```

Returns full step-by-step trajectory of the alternate timeline.

---

## 📁 Repository Structure

```
meta_scaler/
├── models.py                    # PivotAction + PivotObservation (7 actions, 17 fields)
├── openenv.yaml                 # OpenEnv manifest
├── server/
│   ├── app.py                   # FastAPI server + all endpoints
│   ├── pivot_environment.py     # Main environment — wires all subsystems
│   ├── market.py                # MarketSimulator + shock event engine
│   ├── signals.py               # Noisy signal generator
│   ├── founder.py               # Ghost Protocol morale decay
│   ├── investor.py              # 3-round funding system
│   ├── competitor.py            # Rule-based rival (5 strategies)
│   ├── reward.py                # 8-component reward calculator
│   ├── runway.py                # Revenue/burn/cash tracker
│   ├── prompt_encoder.py        # Obs → natural language + multi-turn memory
│   └── wandb_logger.py          # W&B per-step + per-episode logging
├── training/
│   ├── train_colab.ipynb        # GRPO training notebook (12 cells)
│   ├── curriculum.py            # AdaptiveCurriculum (5 tiers)
│   ├── baseline_agent.py        # Random / Stubborn / Panic baselines
│   └── evaluate.py              # Evaluation pipeline
├── scenarios/
│   ├── b2c_saas.json            # Tier 1: easy
│   ├── enterprise_saas.json     # Tier 2: medium
│   ├── fintech.json             # Tier 3: medium-hard
│   ├── marketplace.json         # Tier 4: hard
│   └── consumer_app.json        # Tier 5: very hard
├── static/
│   └── index.html               # Full dashboard (charts, replay, advisor, leaderboard)
├── hf_space/
│   └── Dockerfile               # HF Space deployment
└── docs/
    ├── DETAILS.md               # Deep technical documentation
    ├── WRITEUP.md               # Submission writeup
    └── plots/                   # Training curves (committed post-training)
```

---

## 🤖 Agents in the System

| Agent | Type | Role |
|---|---|---|
| **Founder/CEO** | 🧠 Trained LLM (Qwen2.5-0.5B + LoRA) | Makes all 7 strategic decisions each month |
| **Competitor** | Rule-based | Picks DORMANT/LAUNCH_FEATURE/PRICE_WAR/TALENT_RAID/AGGRESSIVE_MKT based on your weakness |
| **Investor** | Rule-based | 3 rounds ($500K seed → $2M Series A → $5M Series B), requirements shift at steps 20 & 40 |
| **Internal team** | Rule-based | Ghost Protocol decay — advice reliability degrades under financial pressure |

Only the Founder/CEO is trained. The other three are deterministic NPCs that make the environment dynamic.

---

## 📜 License

MIT

---

*Built solo for the Meta PyTorch OpenEnv Hackathon 2026 by [Harshit Makraria](https://github.com/Harshit-Makraria)*
*Powered by [openenv-core](https://github.com/meta-pytorch/OpenEnv) · HuggingFace Transformers · PEFT · W&B*
