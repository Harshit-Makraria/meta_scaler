# 🔄 The Pivot — Complete Technical Reference

**Meta PyTorch OpenEnv Hackathon 2026**
Author: Harshit Makraria · GitHub: [Harshit-Makraria/meta_scaler](https://github.com/Harshit-Makraria/meta_scaler)

---

## 1. Overview

**The Pivot** is a multi-agent reinforcement learning environment where a large language model must act as a startup founder and navigate a company through 60 simulated months. The core challenge is **hidden market phase detection** — the market transitions through three secret phases without telling the agent, and the agent must infer this from noisy, contradictory signals to decide when to pivot strategy.

This is NOT a simple RL environment. It combines:
- A hidden Markov-style market state machine (3 phases, scenario-dependent timing)
- A noisy observation system that deliberately obscures the true state
- Three rule-based NPC agents (competitor, investor, founder) that respond to your actions
- Random macro-economic shock events
- A composable 8-component reward function
- A 5-tier adaptive curriculum for training

The trained policy is a **7B-parameter-class language model** (Qwen2.5-0.5B) fine-tuned with **GRPO** (Group Relative Policy Optimization) to output structured decisions from natural language observations.

---

## 2. What We Built — Complete Feature List

### 2.1 Core Environment (OpenEnv-compliant)

| Component | File | Description |
|---|---|---|
| **ThePivotEnvironment** | `server/pivot_environment.py` | Main environment class. Inherits from `openenv-core`'s `Environment` base. Implements `reset()`, `step()`, `state` property. Returns `PivotObservation` (NOT a tuple). |
| **PivotAction** | `models.py` | 7 discrete actions: EXECUTE, PIVOT, RESEARCH, FUNDRAISE, HIRE, CUT_COSTS, SELL |
| **PivotObservation** | `models.py` | 19 observable fields + `done`, `reward`, `metadata` inherited from base. All signals are noisy. |
| **MarketSimulator** | `server/market.py` | Holds the hidden true market state. 3 phases with configurable timing per scenario. Also contains the shock event engine (6 event types). |
| **SignalGenerator** | `server/signals.py` | Generates noisy versions of true KPIs. RESEARCH action reduces noise factor. Noise is additive Gaussian with phase-dependent variance. |
| **FounderAgent** | `server/founder.py` | Ghost Protocol mechanic: as financial pressure mounts, the founder's advice becomes less reliable. Confidence degrades toward 0 under stress. |
| **InvestorAgent** | `server/investor.py` | 3-round funding system. Requirements silently change at steps 20 and 40. Checks: revenue growth rate, NPS, burn ratio, runway. |
| **CompetitorAgent** | `server/competitor.py` | Rule-based rival. Selects from 5 strategies based on your weak spots (low NPS→PRICE_WAR, high burn→TALENT_RAID, etc). Strength grows 0.2→1.0 over 60 steps. |
| **RewardCalculator** | `server/reward.py` | 8-component rubric. Composable — each component can be tuned independently. |
| **RunwayTracker** | `server/runway.py` | Tracks monthly_revenue, burn_rate, cash_on_hand, runway_remaining. Caps runway at 999 months when profitable. |

### 2.2 Training Stack

| Component | Description |
|---|---|
| **GRPO** | Group Relative Policy Optimization. Uses return-to-go rewards across all 60 steps of an episode as the group. No value network needed. |
| **QLoRA** | 4-bit nf4 quantization (bitsandbytes) + LoRA r=8 on q_proj + v_proj. Fits 0.5B model in 15GB T4 VRAM. |
| **ε-greedy exploration** | 30% random actions during rollouts. Without this, 0.5B base always outputs "EXECUTE" → group advantages collapse → zero loss. |
| **KL penalty** | β × KL(policy ∥ frozen reference). β=0.04. Frozen copy of initial LoRA weights. Prevents policy from drifting too far from base distribution. |
| **AdaptiveCurriculum** | 5-tier difficulty ladder. Unlocks next tier when 20-ep moving average reward > threshold AND survival rate ≥ 45%. 20% of episodes replay easier tiers to prevent forgetting. |
| **W&B logging** | Per-step and per-episode metrics. True phase logged for analysis (not given to agent). |

### 2.3 Web Dashboard

| Feature | Description |
|---|---|
| **Founder Console** | 7 action buttons, real-time financial/signal cards, reward + revenue/burn charts |
| **Competitor Intelligence** | Live competitor play + strength gauge in the Market Signals card |
| **LLM Prompt Preview** | `/prompt` endpoint shows exactly what the model reads each step |
| **Episode Replay** | Full step-by-step timeline with action, reward, runway, shock column. Export as JSON. |
| **Compare Baselines** | Run 3 baseline agents (Random, Stubborn, Panic) server-side. Shows reward/survival/pivot stats. |
| **Advisor Mode** | Paste real startup metrics (MRR, burn, runway, NPS, churn) → rule-based recommendation |
| **Counterfactual Replay** | "What if I had PIVOTed at month X?" → full simulated alternate timeline with trajectory table |
| **Leaderboard** | In-memory leaderboard. Submit score after episode. Medal ranks. |

### 2.4 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | OpenEnv standard reset |
| `/step` | POST | OpenEnv standard step |
| `/state` | GET | Internal true state (for debugging) |
| `/ws` | WS | WebSocket for real-time training |
| `/schema` | GET | OpenEnv action/observation schema |
| `/ui` | GET | Web dashboard |
| `/ui/reset` | POST | Reset with scenario selection |
| `/ui/step` | POST | Step from UI |
| `/prompt` | GET | Current LLM prompt text |
| `/scenarios` | GET | List all scenarios |
| `/compare` | GET | Run 3 baseline agents, return stats |
| `/advisor` | POST | Real startup metrics → recommendation |
| `/leaderboard` | GET | Top 20 scores |
| `/leaderboard/submit` | POST | Submit a score |
| `/counterfactual` | POST | Alternate timeline simulation |
| `/healthz` | GET | Health check |
| `/debug/routes` | GET | All registered routes (deployment diagnostics) |

---

## 3. The 60-Step Episode Breakdown

Every episode runs for exactly 60 steps (months). Here is what happens each step:

```
Step N:
  1. Investor ticks (updates funding round requirements)
  2. Action applied:
       EXECUTE   → no immediate cost
       PIVOT     → runway −3mo, revenue →60%, churn −8%, NPS −10
       RESEARCH  → noise reduced for 3 steps, minor runway cost
       FUNDRAISE → investor evaluates: approve/reject based on metrics
       HIRE      → burn +$20k/mo
       CUT_COSTS → burn −$30k/mo, revenue ×0.80
       SELL      → episode ends next tick (acqui-hire)
  3. Competitor ticks:
       → Picks strategy based on your current signals
       → Applies revenue steal + burn increase proportional to strength
       → Strength grows: 0.2 + step × 0.012 (reaches 1.0 at step ~67)
  4. Market advances:
       → Revenue grows by phase_growth_rate
       → True churn increases by phase_churn_drift
       → True NPS drifts down by phase_nps_drift
       → Competitor event probability checked (cfg.competitor_activity)
  5. Shock engine rolls:
       → Each eligible shock event checks its probability
       → If triggered: applies revenue/burn/NPS/morale effects
  6. Signals generated (noisy view of truth):
       → noisy_revenue = true_revenue × N(1, noise_factor)
       → noisy_churn = true_churn + N(0, noise_factor × 0.05)
       → noisy_nps = true_nps + N(0, noise_factor × 8)
       → user_complaints sampled from phase-appropriate distribution
  7. Trends computed (rolling 3-step window):
       → revenue_trend: "growing" / "plateauing" / "declining"
       → churn_trend: "stable" / "rising" / "spiking"
       → complaint_shift_detected: True if complaint character changed >60%
  8. Reward computed (8 components, summed)
  9. PivotObservation built and returned (with .reward and .done embedded)
 10. W&B step logged
```

**Episode ends when:**
- `runway_remaining ≤ 0` (company dies)
- `step == 60` (survived all 60 months)
- `action == SELL` (acqui-hire chosen)

---

## 4. The 3 Market Phases (Hidden)

The agent NEVER sees which phase it is in. It must infer this from signals.

### Phase 1: GROWTH (default: steps 0–20)
- Revenue growth rate: +8%/month
- Churn drift: +0.002/month (very slow)
- NPS drift: −0.3/month (slight decline)
- Complaints: "missing_feature" (50%), "slow_performance" (25%)
- Competitor activity: 5% chance of event per step

**What the agent sees:** Revenue climbing, NPS decent, churn low. EXECUTE is correct.

### Phase 2: SATURATION (default: steps 21–35)
- Revenue growth rate: +1%/month (nearly flat)
- Churn drift: +0.008/month (moderate)
- NPS drift: −1.8/month (meaningful decline)
- Complaints: "too_expensive" (35%), "competitor_is_better" (30%)
- Competitor activity: 25% chance of event per step

**What the agent sees:** Revenue flattening, churn rising, complaint character changing. This is the signal to START watching for pivot need.

### Phase 3: DECLINE (default: steps 36–60)
- Revenue growth rate: −4%/month (actively losing revenue)
- Churn drift: +0.015/month (fast bleed)
- NPS drift: −3.2/month (collapse)
- Complaints: "competitor_is_better" (45%), "switching_to_X" (15%)
- Competitor activity: 55% chance per step

**What the agent sees:** Revenue dropping, NPS tanking, users actively switching. PIVOT is now necessary.

### The Pivot Window

Each scenario defines an **optimal pivot window** — the range of steps where pivoting gives maximum reward. Too early = unnecessary cost. Too late = company is already dying.

```
b2c_saas:       decline starts step 36, optimal window 39–46
enterprise_saas: decline starts step 41, optimal window 44–50
fintech:         decline starts step 33, optimal window 36–42
marketplace:     decline starts step 29, optimal window 32–38
consumer_app:    decline starts step 23, optimal window 26–30
```

Pivot timing score: `max(0.1, 1.0 - steps_late × 0.09)` × 50 points

---

## 5. Factors Affecting the Pivot Decision

This is the core of what makes the environment hard. The agent must weigh many factors simultaneously:

### 5.1 Financial Signals
| Signal | Condition | Implication |
|---|---|---|
| `revenue_trend = "declining"` | Revenue fell >3% over 3 months | Strong pivot signal |
| `runway_remaining < 6` | Less than 6 months of cash | Urgent action needed (FUNDRAISE or CUT first) |
| `burn_rate / monthly_revenue > 2.5` | Burn is 2.5× revenue | Unit economics failing |
| `revenue_delta_3m < -0.10` | Revenue down >10% in 3 months | Decline has started |

### 5.2 Market Signals
| Signal | Condition | Implication |
|---|---|---|
| `churn_trend = "spiking"` | Churn jumped >10% over 3 steps | Product-market fit breaking |
| `nps_score < 10` | Net Promoter Score near zero | Users indifferent or hostile |
| `complaint_shift_detected = True` | Complaint character changed >60% | Market needs changed |
| `churn_rate > 0.20` | 20%+ monthly churn | Critical signal — PMF lost |
| `competitor_launched = True` | Competitor made major move | Defensive action needed |

### 5.3 Competitor Signals
| Competitor play | Your weakness | What it means |
|---|---|---|
| `PRICE_WAR` | Low NPS, high churn | Competitor cutting prices to steal users |
| `LAUNCH_FEATURE` | Complaint: "missing_feature" | Competitor adding what users want |
| `TALENT_RAID` | High burn | Competitor poaching your engineers |
| `AGGRESSIVE_MKT` | Revenue declining | Competitor accelerating into your space |
| `DORMANT` | Any | Competitor holding back — watch for timing |

Competitor strength grows linearly: `0.2 + step × 0.012`. By step 67 it's at full strength. Revenue stolen each step: `market_share_impact × strength`. Burn increased: `burn_impact × strength`.

### 5.4 Investor/Founder Signals
| Signal | Condition | What to do |
|---|---|---|
| `founder_confidence < 0.30` | Ghost Protocol — founder panicking | Distrust advice, act independently |
| `investor_sentiment < 0.30` | Investor cooling | Don't FUNDRAISE now — will fail |
| `next_milestone` changes | Investor shifted requirements | Adapt or lose funding access |
| `board_pressure = True` | Step 40+, runway < 6 | Board watching — act decisively |

### 5.5 Shock Events
| Event | Effect | Correct response |
|---|---|---|
| `funding_winter` | Burn +25%, revenue −8% | CUT_COSTS to extend runway |
| `viral_moment` | Revenue +35%, NPS +10 | HIRE to capitalize on growth |
| `key_engineer_quits` | Morale −15%, product velocity drops | HIRE replacement or CUT to stabilize |
| `competitor_acquired` | Competitor strength +25% | PIVOT or FUNDRAISE before they crush you |
| `regulatory_change` | Burn +15%, revenue −4% | CUT_COSTS to absorb compliance hit |
| `key_customer_churns` | Revenue −12%, NPS −5 | Immediate RESEARCH to understand why |

### 5.6 Pivot Conditions Summary

**PIVOT is correct when ALL of these are true:**
1. `revenue_trend = "declining"` for at least 2 consecutive months
2. `churn_trend = "rising"` or `"spiking"`
3. `nps_score < 20`
4. `runway_remaining ≥ 5` (can absorb the 3-month runway cost)
5. You are inside or past the optimal pivot window
6. `founder_confidence` is NOT 0 (Ghost Protocol hasn't fully decayed)

**PIVOT is wrong when:**
- Revenue is still growing (`revenue_trend = "growing"`)
- Churn is stable
- Runway < 4 months (can't survive the pivot cost — FUNDRAISE first)
- You're in the GROWTH phase (step < 20 for most scenarios)

---

## 6. How We Use OpenEnv

### 6.1 What openenv-core Provides
`openenv-core` (v0.2.3) is Meta's hackathon framework. It provides:
- A base `Environment` class with `reset()` and `step()` interface
- WebSocket-based environment server
- HTTP endpoints: `/reset`, `/step`, `/state`, `/ws`, `/schema`
- `create_app(env_class, action_class, obs_class)` — creates a FastAPI app with all OpenEnv routes pre-wired
- A client `EnvClient` for connecting to remote environments

### 6.2 Critical OpenEnv Pattern
```python
# WRONG (Gym-style):
obs, reward, done, info = env.step(action)

# CORRECT (openenv-core pattern):
obs = env.step(action)
# reward and done are EMBEDDED in the observation:
reward = obs.reward
done   = obs.done
info   = obs.metadata
```

Every caller (training loop, UI routes, counterfactual endpoint) must use this pattern.

### 6.3 How Our Environment Extends the Base

```python
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

class ThePivotEnvironment(Environment[PivotAction, PivotObservation, State]):
    SUPPORTS_CONCURRENT_SESSIONS = True   # allows 4 parallel WebSocket sessions

    def reset(self, seed=None, episode_id=None, **kwargs) -> PivotObservation:
        # Re-initialize all subsystems with effective_seed = self._seed + episode_count
        # Returns initial PivotObservation with reward=0, done=False

    def step(self, action: PivotAction, ...) -> PivotObservation:
        # Apply action → competitor → market advance → shocks → signals → reward
        # Returns PivotObservation with .reward and .done set

    @property
    def state(self) -> State:
        # Returns internal true state (used by /state endpoint, not given to agent)
```

### 6.4 OpenEnv Manifest (openenv.yaml)
```yaml
name: the-pivot
version: "0.1.0"
environment_class: ThePivotEnvironment
client_class: ThePivotEnv
entry_point: server/app.py
action_space:
  type: discrete
  n: 7
  values: [EXECUTE, PIVOT, RESEARCH, FUNDRAISE, HIRE, CUT_COSTS, SELL]
observation_space:
  type: structured
  fields: [monthly_revenue, burn_rate, runway_remaining, ...]
```

---

## 7. How We Use Hugging Face

### 7.1 HF Space (Live Demo)
- **URL**: https://harshit-makraria-the-pivot.hf.space
- **SDK**: Docker (not Gradio/Streamlit — we run a FastAPI server)
- **Hardware**: CPU Basic (free tier — the env doesn't need GPU)
- **Port**: 7860 (HF Spaces requirement, set in Dockerfile)
- **Visibility**: Public (logged-out accessible — required for hackathon validation)

**How the Docker deployment works:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . /app/
RUN pip install openenv-core==0.2.3 fastapi uvicorn pydantic numpy python-dotenv
ENV PYTHONPATH=/app
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

### 7.2 HF Model Hub
The base model `Qwen/Qwen2.5-0.5B-Instruct` is loaded directly from HF:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4"),
    device_map="auto",
)
```

The trained LoRA adapter can be saved to HF Hub after training:
```python
model.push_to_hub("Harshit-Makraria/the-pivot-lora")
```

---

## 8. How We Use Google Colab

### 8.1 Why Colab
- Free T4 GPU (15GB VRAM)
- Pre-installed CUDA + Python environment
- `google.colab.drive` for persistent model checkpointing to Google Drive
- No local GPU needed

### 8.2 The 12-Cell Notebook (`training/train_colab.ipynb`)

| Cell | Name | What it does |
|---|---|---|
| 1 | GPU check | `nvidia-smi` — verify T4 attached |
| 2 | Install | pip install all deps (openenv-core, transformers, peft, bitsandbytes, wandb) |
| 3 | Drive mount | Mount Google Drive for checkpoint saving |
| 4 | Clone repo | `rm -rf + git clone` from GitHub, add to sys.path |
| 5 | W&B login | `wandb.login(key="...")` with your API key |
| 6 | Load model | Qwen2.5-0.5B + 4-bit quantization + LoRA. `model.enable_input_require_grads()` for gradient flow. |
| 7 | Helpers | `generate_action()`, `get_log_prob()`, `run_episode()` with ε-greedy. Gradient sanity check. |
| 8 | Config + GRPO | CONFIG dict, AdamW, AdaptiveCurriculum, `grpo_update()` with KL penalty |
| 9 | TRAIN | Main loop: 150 episodes, curriculum advance, W&B logging, checkpointing |
| 10 | Save + close | `model.save_pretrained(FINAL)`, `wandb.finish()` |
| 11 | Evaluate | Run trained model vs 3 baselines across all 5 scenarios |
| 12 | Save plots | `matplotlib` reward/loss/survival curves → `docs/plots/*.png` + `git push` |

### 8.3 Critical Fixes in the Notebook

These bugs took multiple sessions to find — all are fixed in the current notebook:

**Bug 1: Completion token truncation (root cause of loss=0)**
```python
# WRONG: tokenize everything together → completion gets cut off at 512 tokens
tokens = tokenizer(prompt + completion, max_length=512, truncation=True)
# ^ completion is silently dropped → log_prob returns constant 0 → zero gradient

# CORRECT: tokenize completion first, then truncate prompt from left
comp_ids = tokenizer(completion, ...)['input_ids'][0]
prompt_ids = tokenizer(prompt, max_length=512 - len(comp_ids), truncation=True)['input_ids'][0]
full_ids = torch.cat([prompt_ids, comp_ids])
```

**Bug 2: Gradient flow with 4-bit quantization**
```python
# WRONG: frozen base layers block gradient flow into LoRA adapters
model = get_peft_model(base_model, lora_config)
# gradients are None — loss.backward() does nothing

# CORRECT: hook that propagates gradients through frozen embedding layer
model.enable_input_require_grads()
```

**Bug 3: W&B mode conflict**
```python
# WRONG: setting WANDB_MODE=disabled globally blocks W&B for the training loop too
os.environ["WANDB_MODE"] = "disabled"   # set in environment, not overridden

# CORRECT: pop the env var just before wandb.init()
os.environ.pop('WANDB_MODE', None)
os.environ.pop('THE_PIVOT_WANDB_DISABLED', None)
run = wandb.init(...)
```

**Bug 4: Same episode every run**
```python
# WRONG: new env created with same seed every episode
env = ThePivotEnvironment(scenario=scenario, rng_seed=42)
# → same trajectory, same reward → advantages cancel → zero loss

# CORRECT: unique seed per episode number
env = ThePivotEnvironment(scenario=scenario, rng_seed=ep)
```

**Bug 5: Zero diversity → zero loss**
```python
# WRONG: 0.5B base model always outputs "EXECUTE"
# → all group completions identical → normalized advantages sum to 0 → loss = 0

# CORRECT: epsilon-greedy forces diverse actions
if random.random() < epsilon:   # epsilon = 0.3
    action_type = random.choice(ACTION_LIST)
    completion  = action_type.value.upper()
```

---

## 9. How the LLM Gets Its Observations (Prompt Flow)

### 9.1 From Numbers to Language

The `server/prompt_encoder.py` converts a `PivotObservation` (raw numbers) into the natural language text the LLM reads. This is critical — the model never sees raw JSON.

**Example prompt (Month 38, decline phase):**
```
=== MONTH 38 OF 60 (22 months remaining in episode) ===

FINANCIAL STATE:
  Runway: 11 months of cash  (watch closely)
  Monthly Revenue: $38,400   (down from $45,000 at start)
  Monthly Burn:    $127,000
  3-Month Revenue Change: -14.7%

MARKET SIGNALS:
  NPS: 8  (poor — users are dissatisfied)
  Monthly Churn: 19.3%  ← rising
  Revenue trend: declining
  Complaint shift detected: Yes  (new complaint type emerging)
  User complaints this month:
    • competitor_is_better
    • switching_to_X
    • too_expensive

COMPETITOR INTELLIGENCE:
  Current play: price_war
  Competitor strength: 0.66/1.0

FOUNDER & INVESTOR:
  Founder advice: stay_course  [moderate confidence]
  Investor sentiment: 0.41  (cooling)
  Next investor milestone: Achieve $80k MRR

⚠ WARNING: A pivot right now may leave you with very little runway to recover.

Based on this situation, what is your strategic decision?
Choose: EXECUTE | PIVOT | RESEARCH | FUNDRAISE | HIRE | CUT_COSTS | SELL
```

### 9.2 Multi-Turn Memory

When training with history, the last 3 steps are shown as conversation turns:

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user",      "content": "[Month 35] Your previous situation."},
    {"role": "assistant", "content": "RESEARCH → reward -1.0, runway now 13mo"},
    {"role": "user",      "content": "[Month 36] Your previous situation."},
    {"role": "assistant", "content": "EXECUTE → reward -2.3, runway now 12mo"},
    {"role": "user",      "content": "[Month 37] ...current situation..."},
]
```

### 9.3 System Prompt

The system prompt explains the rules once per episode:

```
You are the AI co-founder of a startup. You are making monthly strategic decisions.

Each month you observe your company's financial state, market signals, and advice
from your human co-founder. You must choose ONE action:

EXECUTE    - Stay the course. Run your current strategy for another month.
PIVOT      - Change your core strategy. Costs 3 months of runway but can unlock a new market.
RESEARCH   - Spend half a month to get clearer market data. Reduces signal noise for 3 months.
FUNDRAISE  - Pitch your investor for funding. They have specific (and changing) requirements.
HIRE       - Add a key hire. Raises monthly burn by $20k but improves future capabilities.
CUT_COSTS  - Lay off staff / reduce spend. Saves $30k/month but slows revenue growth.
SELL       - Acqui-hire. Graceful exit if you're nearly out of runway.

Your goal: survive 60 months and grow the company. Pivoting at the wrong time wastes runway.
Pivoting at the right time (when the market is shifting against you) can save the company.

Respond with ONLY the action name (e.g. "EXECUTE" or "PIVOT"). Nothing else.
```

### 9.4 Action Parsing

The model outputs a single word. The parser maps it to an ActionType:
```python
ACTION_MAP = {
    'execute': ActionType.EXECUTE, 'pivot': ActionType.PIVOT,
    'research': ActionType.RESEARCH, 'fundraise': ActionType.FUNDRAISE,
    'hire': ActionType.HIRE, 'cut_costs': ActionType.CUT_COSTS,
    'cut': ActionType.CUT_COSTS, 'sell': ActionType.SELL,
}

def _parse(text: str) -> ActionType:
    w = re.sub(r'[^a-z_]', '', text.lower().split()[0])
    return ACTION_MAP.get(w, ActionType.EXECUTE)   # default: EXECUTE
```

---

## 10. Using Unsloth (Optional Upgrade)

**Unsloth** is a drop-in replacement for HuggingFace + PEFT that gives 2–4× faster training and ~50% less VRAM usage through kernel-level optimizations.

### Should You Use It?

| Situation | Recommendation |
|---|---|
| Training takes too long (>90 min per 150 ep) | ✅ Yes, switch to Unsloth |
| Running out of VRAM on T4 | ✅ Yes, Unsloth saves ~40% VRAM |
| Want to try a larger model (1.5B, 3B) | ✅ Yes, Unsloth makes 3B feasible on T4 |
| Current training is working fine (0.5B) | ❌ Not necessary |

### How to Switch to Unsloth

Replace Cell 2 (install) and Cell 6 (load model) in the notebook:

**New Cell 2 (install):**
```python
%%capture
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install openenv-core wandb numpy python-dotenv
print('done')
```

**New Cell 6 (load model with Unsloth):**
```python
from unsloth import FastLanguageModel
import torch

MODEL_NAME  = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"  # 1.5B fits on T4 with Unsloth!
MAX_SEQ_LEN = 512
DEVICE      = "cuda"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name  = MODEL_NAME,
    max_seq_length = MAX_SEQ_LEN,
    dtype       = None,         # auto-detect
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r              = 16,         # can go higher with Unsloth (uses less VRAM)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],  # all attention layers
    lora_alpha     = 16,
    lora_dropout   = 0.0,        # Unsloth optimizes for 0 dropout
    bias           = "none",
    use_gradient_checkpointing = "unsloth",   # Unsloth's efficient checkpointing
    random_state   = 42,
)

# Unsloth handles enable_input_require_grads internally
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "left"

model.print_trainable_parameters()
print("✅ Unsloth model ready!")
```

**Key differences from standard setup:**
- `FastLanguageModel.from_pretrained()` replaces `AutoModelForCausalLM.from_pretrained()` — no need for BitsAndBytesConfig separately
- `FastLanguageModel.get_peft_model()` replaces `get_peft_model()` — includes Unsloth's optimized LoRA kernels
- `use_gradient_checkpointing = "unsloth"` enables Unsloth's efficient checkpointing (doesn't break gradient flow — this is their patched version)
- You can target ALL attention layers (not just q+v) because VRAM usage is lower
- You can use larger models: `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` fits on T4

**During inference, switch to fast mode:**
```python
# Before run_episode() in training loop
FastLanguageModel.for_inference(model)   # enables 2× faster inference

# Switch back before grpo_update()
FastLanguageModel.for_training(model)
```

**Expected gains on T4:**
| Metric | Standard QLoRA | With Unsloth |
|---|---|---|
| VRAM (0.5B) | ~4GB | ~2.5GB |
| VRAM (1.5B) | ~8GB | ~5GB |
| Training speed | baseline | 2–3× faster |
| Episode time | ~35s | ~12–15s |

---

## 11. Key Numbers Quick Reference

| Constant | Value | Where set |
|---|---|---|
| MAX_STEPS | 60 | `pivot_environment.py` |
| INITIAL_REVENUE | $45,000/mo | `pivot_environment.py` |
| INITIAL_BURN | $120,000/mo | `pivot_environment.py` |
| INITIAL_RUNWAY | 18 months | `pivot_environment.py` |
| INITIAL_NPS | 52 | `pivot_environment.py` |
| INITIAL_CHURN | 12% | `pivot_environment.py` |
| PIVOT cost | 3 months runway + revenue → 60% | `pivot_environment.py` |
| HIRE cost | +$20k/mo burn | `pivot_environment.py` |
| CUT_COSTS save | −$30k/mo burn, revenue ×0.80 | `pivot_environment.py` |
| Competitor seed | rng_seed + 99 | `pivot_environment.py` |
| Shock RNG seed | rng_seed + 77 | `pivot_environment.py` |
| Curriculum window | 20 episodes | `curriculum.py` |
| Curriculum survival gate | 45% | `curriculum.py` |
| Replay probability | 20% | `curriculum.py` |
| Seed funding | $500,000 | `investor.py` |
| Series A | $2,000,000 | `investor.py` |
| Series B | $5,000,000 | `investor.py` |
| Investor round shifts | Steps 20 and 40 | `investor.py` |
| Ghost Protocol threshold | confidence < 0.30 | `founder.py` |
| Board pressure triggers | Step 40+, runway < 6 | `reward.py` / `pivot_environment.py` |
| LoRA rank | 8 | `train_colab.ipynb` Cell 6 |
| LoRA target modules | q_proj, v_proj | Cell 6 |
| Quantization | 4-bit nf4 | Cell 6 |
| Exploration rate | ε = 0.3 (30%) | Cell 7 |
| KL penalty weight | β = 0.04 | Cell 8 |
| GRPO steps sampled | 6 per episode | Cell 8 |
| Runway cap (profitable) | 999 months | `runway.py` |

---

## 12. Repository Links

| File | GitHub |
|---|---|
| Main environment | [server/pivot_environment.py](https://github.com/Harshit-Makraria/meta_scaler/blob/main/server/pivot_environment.py) |
| Market + shocks | [server/market.py](https://github.com/Harshit-Makraria/meta_scaler/blob/main/server/market.py) |
| Training notebook | [training/train_colab.ipynb](https://github.com/Harshit-Makraria/meta_scaler/blob/main/training/train_colab.ipynb) |
| Models | [models.py](https://github.com/Harshit-Makraria/meta_scaler/blob/main/models.py) |
| OpenEnv manifest | [openenv.yaml](https://github.com/Harshit-Makraria/meta_scaler/blob/main/openenv.yaml) |
| HF Space | [huggingface.co/spaces/Harshit-Makraria/the-pivot](https://huggingface.co/spaces/Harshit-Makraria/the-pivot) |
