# 🔄 The Pivot

**An OpenEnv-compliant RL environment where an LLM founder agent must detect hidden market phase shifts and decide when to pivot — without being told.**

> Built for the **Meta PyTorch OpenEnv Hackathon 2026** (Scaler × PyTorch, Bangalore, April 25–26)

---

## 🔗 Deliverables

| What | Where |
|------|-------|
| 🚀 **HF Space (live demo)** | `<PASTE_HF_SPACE_URL_HERE>` |
| 📓 **Training notebook (Colab)** | [`training/train_colab.ipynb`](training/train_colab.ipynb) — [Open in Colab](https://colab.research.google.com/github/Harshit-Makraria/meta_scaler/blob/main/training/train_colab.ipynb) |
| 📝 **Writeup** | [`docs/WRITEUP.md`](docs/WRITEUP.md) |
| 🧩 **OpenEnv manifest** | [`openenv.yaml`](openenv.yaml) |
| 📊 **Training plots** | [`docs/plots/`](docs/plots/) |

---

## 📊 Training results

### Reward curve
![Reward curve](docs/plots/reward_curve.png)

### GRPO loss curve
![Loss curve](docs/plots/loss_curve.png)

### Episode length (survival)
![Survival](docs/plots/survival_curve.png)

*Trained on Qwen2.5-0.5B-Instruct + LoRA (r=8) with GRPO, 150 episodes on a T4 GPU. See [`docs/plots/metrics.json`](docs/plots/metrics.json) for exact numbers.*

---

## 🧠 The task

An LLM agent plays a startup founder across **60 simulated months**. Each step it chooses one of 6 actions:

`EXECUTE · PIVOT · RESEARCH · FUNDRAISE · HIRE · CUT_COSTS`

The market silently moves through **3 hidden phases**: GROWTH → SATURATION → DECLINE. The decline-start timing is **scenario-dependent and not revealed** — the founder must infer it from noisy NPS, churn, CAC, LTV, and competitor-move signals, and pivot in a narrow window (usually ±3 months).

Pivot too early → burn runway on a still-growing product. Pivot too late → die in the decline phase. The reward is a composable 5-part rubric (survival, growth, timing, efficiency, pivot quality).

## 🏗️ What's inside

```
server/
├── pivot_environment.py   # ThePivotEnvironment (OpenEnv-compliant)
├── market.py              # 3-phase hidden market with scenario-aware timing
├── signals.py             # Noisy signal generator (RESEARCH reduces noise)
├── founder.py             # Internal team dynamics (Ghost Protocol decay)
├── investor.py            # 3 funding rounds silently shifting
├── competitor.py          # Rule-based rival (5 strategies, strength grows 0.2→1.0)
├── reward.py              # 5-component rubric reward
└── app.py                 # FastAPI + OpenEnv HTTP/WebSocket server

training/
├── curriculum.py          # Adaptive 5-tier curriculum (easy → very-hard)
├── baseline_agent.py      # Random / Stubborn / Panic baselines
├── train_colab.ipynb      # GRPO training notebook (runnable in Colab)
└── evaluate.py            # Full eval pipeline with plots

static/
└── index.html             # Interactive dashboard with replay viewer
```

## 🎯 Agents

**One trained agent** — the Founder/CEO (LLM). The environment contains **three rule-based NPCs**:

| Role | Type |
|------|------|
| Founder/CEO | 🧠 **Trained (GRPO + LoRA)** |
| Competitor | Rule-based — DORMANT / LAUNCH_FEATURE / PRICE_WAR / TALENT_RAID / AGGRESSIVE_MKT |
| Investor | Rule-based — 3-round funding shifts |
| Internal team | Rule-based — Ghost Protocol morale decay |

## 🧬 Scenarios (curriculum)

| Tier | Scenario | Difficulty | Decline starts | Pivot window |
|------|----------|-----------|----------------|--------------|
| 1 | b2c_saas | easy | step 36 | 39–46 |
| 2 | enterprise_saas | medium | step 41 | 44–50 |
| 3 | fintech | medium-hard | step 33 | 36–42 |
| 4 | marketplace | hard | step 29 | 32–38 |
| 5 | consumer_app | very-hard | step 23 | 26–30 |

The adaptive curriculum unlocks tier N+1 only when the 20-episode moving average reward exceeds a threshold **AND** survival rate ≥ 45%.

## 🚀 Run it locally

```bash
git clone https://github.com/Harshit-Makraria/meta_scaler
cd meta_scaler
pip install openenv-core fastapi uvicorn pydantic numpy python-dotenv
uvicorn server.app:app --port 8000
```

Open:
- Dashboard: http://localhost:8000/ui
- API docs: http://localhost:8000/docs
- Baseline compare: http://localhost:8000/compare?scenario=b2c_saas&n_episodes=10

## 🏋️ Train your own policy

Open [`training/train_colab.ipynb`](training/train_colab.ipynb) in Colab with a T4 GPU and run all cells. ~60–90 minutes for 150 episodes.

The notebook uses:
- **Base model**: Qwen/Qwen2.5-0.5B-Instruct
- **PEFT**: LoRA r=8 (q_proj, v_proj only)
- **Quantization**: 4-bit nf4
- **Algorithm**: GRPO with ε-greedy exploration, return-to-go advantages
- **Memory**: Fits in 15GB T4 VRAM

## 🧪 Quick programmatic test

```python
from server.pivot_environment import ThePivotEnvironment
from models import PivotAction, ActionType

env = ThePivotEnvironment()
obs = env.reset()
for _ in range(60):
    obs = env.step(PivotAction(action_type=ActionType.EXECUTE))
    if obs.done: break
print('reward:', obs.reward, 'runway:', obs.runway_remaining)
print('competitor:', obs.competitor_play, obs.competitor_strength)
```

## 📝 License

MIT

## 🙏 Credits

Built solo for Meta PyTorch OpenEnv Hackathon 2026 by [@Harshit-Makraria](https://github.com/Harshit-Makraria).
Powered by [openenv-core](https://github.com/meta-pytorch/OpenEnv), 🤗 [Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft), and [W&B](https://wandb.ai).
