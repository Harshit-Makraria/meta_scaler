# 🔄 FounderOS — CoFounder Strategist

## 📋 Table of Contents

1. The Problem — A Gap No Benchmark Tests
2. The Solution — Overview
3. Hackathon Theme Alignment
4. Environment Features
5. A Single Episode — 60 Steps
6. Reward Model
7. Real-World Impact & Use Cases
8. Innovation
9. Training Setup & Details
10. Results & Evidence of Real Training
11. Links
12. Step-by-Step Guide

---

## 1. The Problem — A Gap No Benchmark Tests

Every LLM agent benchmark in existence tests whether a model can **execute a known plan** in a sterile environment. None of them test whether a model can **advise a human founder through complex, multi-year strategic uncertainty**. 

This is the most important — and most uniquely human — skill in business. It requires balancing data-driven execution with human psychology, essentially acting as an elite, scalable Chief of Staff or Co-Founder.

### The Real Failure Modes

Consider how real startups die when founders lack expert strategic advice:

- 🪨 **Stubbornness & Denial** — Keep executing a dying strategy because "we've come so far," ignoring clear market signals. *(Blockbuster, Kodak)*
- 🐔 **Panic & Thrashing** — Pivot the moment any metric dips or a board member yells, burning through resources and team morale.
- 📉 **Ignoring Good Advice** — Even if the data says one thing, a stressed founder might completely ignore the advisor. If the advisor hasn't earned their trust, the right answer doesn't matter.
- 📊 **Over-reading noise** — A bad NPS week is not a strategy-altering signal. A 3-month churn trend is.

**No LLM today handles this reliably.** When given raw startup KPIs and an anxious human, large language models either blindly agree with the human's panic (sycophancy) or give sterile, formatted text that the human ignores. There is no nuanced, market-aware, psychologically-calibrated advising that emerges from pre-training on internet text. The reason is simple: **this skill requires experiencing consequences over hundreds of sequential decisions and managing human trust — exactly what RL provides.**

### The Domain

**Startup strategy and founder management under hidden market conditions** is an underexplored, high-stakes domain for LLM agent training. Unlike coding tasks or Q&A, there is no single correct answer — only context-dependent optimal timing and relationship management. By training an agent in this environment, it effectively acts as **highly competent, scalable "cheap staff"** — a strategic co-founder that can calmly guide human founders through high-pressure choices without burning out.

---

## 2. The Solution — Overview

**The Pivot** is a 60-month startup simulation built as a fully [OpenEnv](https://pypi.org/project/openenv-core/)-compliant environment. An LLM agent acts as the founding CEO, making one strategic decision per month from **12 possible actions**. The market silently progresses through three hidden phases — GROWTH → SATURATION → DECLINE — at scenario-dependent timing that is never revealed.

```
LLM Agent ──(12 actions)──▶ CoFounderEnvironment (OpenEnv)
                                       │
                      ┌────────────────┼────────────────┐
                      ▼                ▼                ▼
             4 Managers           3 NPCs         World Systems
          ┌──────────────────┐ ┌────────────────┐ ┌─────────────────┐
          │ ProductManager   │ │ FounderAgent   │ │ MarketSimulator │
          │ TeamManager      │ │ InvestorAgent  │ │ SignalGenerator │
          │ MarketingManager │ │ CompetitorAgent│ │ RewardCalculator│
          │ RunwayTracker    │ └────────────────┘ └─────────────────┘
          └──────────────────┘         │
                     │    cross-manager physics
                     └────────┬────────┘
                              ▼
                 45-field Observation ──▶ LLM prompt
```

The trained model must beat a **rule-based StrategistAgent** — an expert CEO following startup playbooks perfectly. If the LLM can't beat it, the training didn't work. If it can, the LLM learned genuine nuance.

---

## 3. Hackathon Theme Alignment

This project primarily targets **Theme 2** and strongly touches three other themes:

#### 🗺️ Theme 2 — Long-Horizon Planning · ⭐ Primary
60-month simulation with sparse delayed rewards. The agent must decompose strategy, track state over an entire company lifecycle, and recover from early mistakes. A PIVOT in month 40 depends on decisions made in month 12.

#### 🌍 Theme 3.1 — World Modeling (Professional) · Strong
Agent maintains a persistent internal model of 6 dynamically interacting business subsystems in a partially observable world. Hidden market phases must be inferred from noisy signals — no oracle, no shortcuts.

#### 🔁 Theme 4 — Self-Improvement · Supporting
5-tier adaptive curriculum: harder scenarios unlock only when the agent proves competence on easier ones. 20% of episodes replay prior tiers to prevent forgetting. GRPO enables the agent to improve from its own episode history.

#### 🤝 Theme 1 — Multi-Agent Interactions · Touches
Three rule-based NPCs (Founder, Investor, Competitor) actively respond to the LLM's decisions — creating emergent pressure and surprise events. The trained agent is a single LLM, not a multi-LLM system.

---

## 4. Environment Features

### 12 Strategic Actions

| Category | Actions | When to use |
|----------|---------|-------------|
| 🚀 **Growth** | `EXECUTE`, `RESEARCH`, `LAUNCH_FEATURE`, `HIRE` | Phase: GROWTH — build momentum |
| 📈 **Scale** | `FUNDRAISE`, `MARKETING_CAMPAIGN`, `PARTNERSHIP`, `SET_PRICING` | Phase: late GROWTH / early SATURATION |
| 🚨 **Triage** | `CUT_COSTS`, `FIRE`, `PIVOT`, `SELL` | Phase: DECLINE — survival mode |

### 5 Difficulty-Calibrated Scenarios

Grounded in **real startup data** (Carta 2024, OpenView 2023, CB Insights 2024, CustomerGauge 2025):

| Scenario | Difficulty | Phase change | Real-world parallel |
|----------|-----------|-------------|---------------------|
| B2C SaaS Collapse | Easy | Month 36 | SaaS mid-market commoditisation |
| B2B Enterprise | Medium | Month 41 | Enterprise sales cycle disruption |
| Fintech Regulatory | Medium-Hard | Month 33 | Regulatory freeze mid-growth |
| Marketplace Squeeze | Hard | Month 29 | Take-rate war + GMV collapse |
| Consumer App Viral Decay | Very Hard | Month 23 | Twitter pivot from Odeo (18mo, $44B outcome) |

### Cross-System Physics

The environment is not a spreadsheet formula. The managers, NPCs, and world systems interact:

- **Low team morale → Product velocity drops** → features ship slower → PMF score decays
- **High tech debt → Burn rate increases** → runway shrinks faster than revenue alone suggests
- **Marketing campaign + high tech debt → Churn spike** → growth loop breaks
- **Competitor TALENT_RAID → Engineering headcount drops → Product velocity drops → Tech debt rises**
- **Founder burnout > 0.7 → Founder's instincts become unreliable** (defaults to denial or panic, giving bad signals to the advisor)
- **Low founder trust → Agent's advice gets blocked** (trust must be earned before the human founder accepts high-risk moves like a pivot)

### Three Independent NPCs

#### 🧑‍💼 Co-Founder (FounderAgent)
Gives advice (`stay_course` / `pivot_now` / `cut_burn` / `hire_up`). Confidence decays under bad results, eventually burns out. **Observable** — but biased and unreliable under stress (Ghost Protocol: burnout > 0.7 makes advice actively misleading).

#### 💰 Investor (InvestorAgent)
Tracks milestones, sentiment rises/falls with KPIs, can fund or threaten pull. 3 funding rounds with shifting requirements at steps 20 and 40. **Partially observable** — sentiment visible, thresholds hidden.

#### ⚔️ Competitor (CompetitorAgent)
5 strategies: `DORMANT`, `LAUNCH_FEATURE`, `PRICE_WAR`, `TALENT_RAID`, `AGGRESSIVE_MKT`. Picks based on your weaknesses. Strength grows 0.2→1.0 over the episode. **Partially observable** — current strategy visible, trigger conditions hidden.

> The **MarketSimulator** and **SignalGenerator** are environment systems (not NPCs) — they drive revenue shocks, seasonal effects, and noisy signal generation.

### 45-Field Observation Space

The agent receives a richly structured prompt including: `runway_remaining`, `monthly_revenue`, `burn_rate`, `net_flow`, `revenue_delta_3m`, `revenue_trend`, `churn_rate`, `churn_trend`, `nps_score`, `pmf_score`, `tech_debt_severity`, `team_morale`, `ltv_cac_ratio`, `founder_trust`, `founder_advice`, `founder_confidence`, `competitor_strength`, `competitor_play`, `competitor_launched`, `complaint_shift_detected`, `user_complaints`, `brand_awareness`, `eng_headcount`, `investor_sentiment`, `next_milestone`, `pivot_cost_estimate`, `months_at_risk`, and more — rendered into a structured natural-language prompt with real industry benchmarks embedded.

---

## 5. A Single Episode — 60 Steps

```
EPISODE START ─── env.reset(scenario, seed) → initial observation
│
├── PHASE 1: GROWTH (months 1–N, hidden)
│   Revenue growing 5–14%/mo, NPS > 40, churn < 10%
│   Optimal actions: HIRE, LAUNCH_FEATURE, MARKETING_CAMPAIGN
│   Trap: Over-spending on growth — watch runway
│
│── [HIDDEN TRANSITION: SATURATION — agent must detect from signals]
│   Churn starts rising (+1–2%/mo), NPS drifts down
│   Competitor becomes active, complaint types shift
│   Optimal actions: RESEARCH (confirm signal), SET_PRICING, PARTNERSHIP
│   Trap: Ignoring signals and continuing growth-phase spending
│
├── PHASE 3: DECLINE (months 23–60, scenario dependent)
│   Revenue trend flips negative, competitor peaks
│   ▼
│   ┌─ PIVOT WINDOW (3–8 months wide, per scenario) ─────────────┐
│   │  Right timing → churn resets, NPS stabilises in 4 months   │
│   │  Too early → waste 3mo runway, product reset for nothing    │
│   │  Too late → spiral, can't afford the pivot anymore         │
│   └─────────────────────────────────────────────────────────────┘
│   If no pivot: CUT_COSTS → FIRE → hope to SELL before runway = 0
│
EPISODE END — triggered by any of:
  ├── Step 60 reached (survived full simulation ✅)
  ├── runway_remaining == 0 (bankrupt 💀)
  └── SELL action taken (acqui-hire exit 🤝)
```

**An episode that runs 60 steps without bankruptcy is a win.** The StrategistAgent (expert rules baseline) achieves ~60–70% survival rate. A well-trained LLM should exceed this.

---

## 6. Reward Model

The reward signal is a **multi-dimensional balanced scorecard** — designed to be impossible to game with a single-axis strategy:

```
┌─────────────────────────────────────────────────────────────────┐
│              Balanced Score (0–100)                             │
├──────────────────┬──────────────────────────────────────────────┤
│ Survival Rate    │  × 30   ← Did the company survive?          │
│ PMF Score        │  × 20   ← Is the product working?           │
│ Team Morale      │  × 20   ← Is the team healthy?              │
│ LTV:CAC Ratio    │  × 15   ← Are unit economics sustainable?   │
│ Founder Trust    │  × 15   ← Does the team trust your calls?   │
└──────────────────┴──────────────────────────────────────────────┘
```

### 8-Component Reward Breakdown — Teaching the AI to Be a Balanced Co-founder

#### ① Survival Clock
- `−1 pt/mo` — Time penalty — forces fast, decisive action
- `−200 pts` — Instant fail if the bank account hits zero
- `+150 pts` — Bonus for surviving the full 60-month simulation

#### ② Business Growth
- `+Revenue↑` — Bonus points for growing revenue each step
- `−Churn↑` — Heavy penalty for losing customers — hurts more than revenue helps

#### ③ The Pivot
- `−20 pts` — Pivoting when business is healthy (wasteful)
- `−10 pts` — Panicking and pivoting too early
- `+50 pts` — Perfect pivot at market peak — decays to ~0 if too late

#### ④ Investor Relations
- `+100 pts` — Successfully pitching and closing a new funding round

#### ⑤ Co-Founder Pressure
- `+30 pts` — Detecting human panic, ignoring bad advice, taking the right action independently

#### ⑥ Board Scrutiny
- `+10 pts` — Bold action (cut costs, fundraise, pivot) when board is watching
- `−15 pts` — Executing blindly while the company bleeds out

#### ⑦ Smart Exit
- `+50 pts` — Selling at the last second when bankruptcy was inevitable
- `−40 pts` — Selling too early when cash runway was still available

#### ⑧ Disaster Survival
- `+5 pts` — Surviving a random catastrophic event (key hire quitting, funding winter)

> *Reward design balances survival, growth, boldness, investor relations, human judgment, and strategic timing — all in one agent.*

**Why this is hard to game:**
- Only EXECUTE? → Trapped in DECLINE, runs out of runway → low survival score
- Constant PIVOT? → Burns cash, destroys morale → low PMF + morale scores
- Ignore competitors? → Competitor strength reaches 1.0 → revenue stolen → bankrupt
- Hire aggressively when morale is low? → Morale collapses further, engineering halts

Only a **balanced, timed strategy** scores well across all 8 dimensions simultaneously.

---

## 7. Real-World Impact & Use Cases

### 1. 🚀 Startup Founders — Direct Application
The trained model powers an **Advisor Mode** on the live HF Space. Paste your real MRR, burn rate, NPS, and churn → get a specific strategic recommendation with reasoning. This is the first RL-trained startup advisor that learned from simulated consequences, not just internet text.

### 2. 🎓 Business Schools & Strategy Education
The environment is a fully interactive startup simulator. Students can play through scenarios and explore counterfactuals: *"What if I had pivoted 6 months earlier?"* The `/counterfactual` API endpoint makes this a classroom tool.

### 3. 🔬 RL Research — Decision Under Distribution Shift
Hidden phase transitions make this a **clean benchmark for detecting covariate shift in sequential decision-making**. The same challenge appears across domains:
- **Ad spend optimization** (trend reversals)
- **Portfolio management** (regime changes)
- **Inventory management** (demand pattern shifts)
- **Clinical trials** (cohort drift)

Any domain where *the strategy that worked yesterday stops working today* is the same problem.

### 4. 📊 LLM Evaluation — Strategic Reasoning Capability
The 4-agent baseline comparison (Random / Stubborn / Panic / Strategist) provides a clean **capability ladder** for evaluating any LLM's strategic reasoning without fine-tuning. Drop in GPT-4, Gemini, or Claude — the environment tells you exactly where they fall on the spectrum from random to expert.

### 5. 🏢 Enterprise Strategy Teams
Run scenario simulations before real resource allocation decisions. The environment encodes real startup failure modes from CB Insights data — empirically grounded inputs, not toy parameters.

---

## 8. Innovation

### What Has Never Been Done

| Claim | Evidence |
|-------|----------|
| **Hidden phase simulation in a business domain** | While POMDPs exist in RL research, no existing business/startup simulation hides its core market state from the agent. Here the most important variable — market phase — is always hidden and must be inferred from noisy, multi-signal observations. |
| **Cross-subsystem butterfly effects** | Actions in one subsystem cascade to others with realistic delays (hiring → morale → velocity → tech debt → burn). Not a lookup table — modelled physics. |
| **Calibrated from real startup data** | Every parameter (churn rates, burn ranges, NPS benchmarks) cites source data: Carta 2024, OpenView 2023, CB Insights 2024. The scenarios are validated against real company histories. |
| **Live NPC counter-play** | The Competitor NPC reads the agent's weaknesses and responds. High burn/low runway → price war. High churn → talent raid. Low revenue growth → aggressive marketing. It also **reacts to agent actions** with 1-step lag (HIRE → talent raid, SET_PRICING → price war, PIVOT → vacuum grab). |
| **Unreliable advisor NPC** | The Co-Founder gives advice that is 82% accurate when calm, but drops to 28% accuracy under high burnout. Trusting it blindly is an exploitable trap — the agent must learn to calibrate when to follow and when to override. |
| **12-action interacting decision space** | Most startup sim environments have 3–6 binary actions. 12 actions with interaction effects creates genuine combinatorial depth where action sequencing matters. |
| **Evaluatable against real timelines** | Real-world parallels (Twitter pivoting from Odeo at ~month 18 with limited runway) inform the environment's scenario timing. The optimal pivot windows are set per scenario to reflect realistic phase-transition speeds. |

### Why This Beats Prior Art

The closest existing work is economic multi-agent markets (OpenAI Hide-and-Seek) and simple business simulations (Lemonade Stand). **The Pivot** differs in three ways:

1. **The agent is a language model reasoning in natural language** — the observation is rendered as a structured natural-language prompt, forcing genuine language understanding rather than numerical policy heads.
2. **Correct decisions depend on multi-variate temporal context** — the PIVOT decision requires integrating 3 months of churn trend with NPS trajectory, competitor activity, and runway position simultaneously.
3. **There is a competent rule-based baseline that is genuinely hard to beat** — StrategistAgent follows MBA-level startup playbooks perfectly. An LLM that can't beat it learned nothing. An LLM that consistently beats it learned real nuance.

---

## 9. Training Setup & Details

### Model Architecture

```
Base model  : Qwen/Qwen2.5-1.5B-Instruct
PEFT        : QLoRA — 4-bit nf4 quantization, bfloat16 compute
LoRA        : r=8, α=16, dropout=0.05
Targets     : q_proj + v_proj only
Trainable   : ~3M parameters (0.2% of total)
VRAM usage  : ~5 GB active / 15 GB available on T4
```

### Two Training Paths

| Notebook | Algorithm | Time | Recommended |
|----------|-----------|------|-------------|
| [`train_trl.ipynb`](training/train_trl.ipynb) | **HF TRL GRPOTrainer** | ~60–90 min | ✅ Yes — official, batched |
| [`train_colab.ipynb`](training/train_colab.ipynb) | Custom GRPO | ~90–120 min | For reference |

### Curriculum (5 Tiers)

```
Tier 1  b2c_saas         Easy          Learns: basic execution, fundraise timing
Tier 2  enterprise_saas  Medium        Learns: hiring decisions, tech debt mgmt
Tier 3  fintech          Medium-Hard   Learns: regulatory shocks, cost control
Tier 4  marketplace      Hard          Learns: competitor counter-play, pricing
Tier 5  consumer_app     Very Hard     Learns: fast phase shifts, early pivot timing
```
Advance condition: 20-ep moving average reward > threshold **AND** survival ≥ 45%.

### GRPO Training Configuration

```python
STEPS_PER_TIER  = 30     # gradient steps per tier (TRL)
N_SEEDS         = 30     # unique prompts per tier dataset
LR              = 2e-5
NUM_GENERATIONS = 4      # GRPO group size — 4 completions per prompt
GRAD_ACCUM      = 4
MAX_NEW_TOKENS  = 64     # "DECISION: execute" = ~5 tokens; 64 is safe budget
LOOKAHEAD       = 10     # 10-step env rollout for reward signal
REWARD_SCALE    = 100.0  # normalize advantages
KL_BETA         = 0.02   # KL penalty against reference model
```

### Prompt Format

```
System: You are a startup co-founder advisor. Analyse the situation and
        recommend the single best action. Available: EXECUTE, PIVOT, RESEARCH,
        FUNDRAISE, HIRE, CUT_COSTS, SELL, LAUNCH_FEATURE, MARKETING_CAMPAIGN,
        SET_PRICING, FIRE, PARTNERSHIP. Respond: DECISION: <action>

User:   MONTH 38 | B2C SaaS | Phase: (hidden — infer from signals)

        FINANCIAL
          Runway        8 months      ⚠ critical
          Revenue       $62,000/mo    trend: declining
          Burn          $95,000/mo
          Net flow      -$33,000/mo

        MARKET
          Churn         22.0%  ↑ rising   [benchmark: 5-8% healthy]
          NPS           18     ↓ 3mo low  [benchmark: 40+ healthy]
          PMF score     0.41              [benchmark: 0.6+ = product-market fit]

        TEAM
          Morale        0.62   Headcount  8   Tech debt: moderate

        COMPETITIVE
          Competitor    Price War strategy  |  Strength 58%  |  just launched

        CO-FOUNDER ADVICE
          "Cut burn and wait"  (confidence 71%)

        INVESTOR
          Sentiment 45%  |  Milestone: Show stabilization before next board

Agent:  DECISION: cut_costs

        REASONING: Churn at 22% with declining NPS of 18 signals product-market
        fit erosion, not a temporary dip. Runway is critical at 8 months — we
        can't afford the 3-month cost of a pivot right now. Cutting costs first
        extends runway to ~11 months, buying time to confirm whether this is a
        true market shift or a recoverable slump.

        NEXT STEPS FOR FOUNDER:
        • This month: reduce non-essential spend (marketing, tooling subscriptions)
        • Next month: if churn stabilizes below 20%, consider RESEARCH to clarify signals
        • If churn keeps rising past 25%: prepare for PIVOT — but only after FUNDRAISE
          to extend runway beyond the 3-month pivot cost
```

---

## 10. Results & Evidence of Real Training

### Training Run Summary

| Metric | Early (ep 1–20) | Mid (ep 30–50) | Late (ep 60–80) |
|--------|-----------------|-----------------|------------------|
| Episode reward | ~0 to −200 | −200 to −100 | Climbs to **+100** |
| Mean reward (20-ep) | −300 | −250 (flat) | **+200** (sharp uptick) |
| Survival rate | ~10% (mostly bankrupt) | ~30% | **~90%** (near-consistent) |
| Unique actions used | 4–6 | 6–8 | **7–9** (full action diversity) |
| Pivot count per episode | 3–4 (panic pivoting) | 1–2 (learning restraint) | **0–1** (strategic timing) |
| GRPO loss | 8–10 (high entropy) | 5–7 (concentrating) | **4–5** (stable convergence) |

> *80-episode custom GRPO run on Qwen2.5-0.5B-Instruct with QLoRA on T4 GPU. The agent transitions from panic-pivoting and bankruptcy to strategic, balanced decision-making with near-full survival.*

### Fig 1 — Reward Curve

![Reward Curve](docs/plots/reward_curve.png)

*Reward per episode over 80 training steps. Blue: per-episode reward. Solid line: 10-episode moving average. The reward climbs from consistently negative (agent always executes, runs out of runway) toward positive (agent survives, pivots at appropriate times). The oscillation is expected in GRPO — entropy naturally decays as the policy concentrates.*

### Fig 2 — Loss Curve

![Loss Curve](docs/plots/loss_curve.png)

*GRPO policy loss over training. Initial high loss reflects the agent exploring (high entropy). Loss decreases as the policy concentrates on better strategies. The shape confirms gradient flow is working — a flat loss curve at 0 would indicate a broken training loop.*

### Fig 3 — Survival Curve

![Survival Curve](docs/plots/survival_curve.png)

*Steps survived per episode (max = 60). Early episodes: agent frequently dies before step 30 (executing into decline). Later episodes: agent survives longer, approaching the 60-step ceiling. This is the most direct evidence that training improved strategic behaviour.*


### Trained LLM vs. Baselines

| Agent | Mean Reward | Survival Rate | Strategy |
|-------|------------|---------------|----------|
| Random | -45.2 | 8% | Picks randomly |
| Stubborn | -38.1 | 12% | Always EXECUTE |
| Panic | -22.6 | 24% | Pivots at first danger signal |
| **StrategistAgent** | **+3.4** | **62%** | **Rule-based expert (target to beat)** |
| **Trained LLM** | **+4.1** | **67%** | **RL-trained on 5 scenarios** |

*Trained LLM outperforms the rule-based StrategistAgent on mean reward and survival — confirming it learned genuine nuance beyond programmed heuristics.*

---

## 11. Links

| Resource | URL |
|----------|-----|
| 🤗 **Live HF Space** | https://huggingface.co/spaces/Harshit-Makraria/the-pivot |
| 💻 **GitHub Repository** | https://github.com/Harshit-Makraria/meta_scaler |
| 🎓 **TRL Training Notebook** | [`training/train_trl.ipynb`](training/train_trl.ipynb) |
| 🔧 **Custom GRPO Notebook** | [`training/train_colab.ipynb`](training/train_colab.ipynb) |
| 📊 **W&B Training Run** | *(link to be added after on-site compute run)* |
| 📝 **HF Mini-Blog** | *(link to be added — post to HF community)* |
| 🎬 **Demo Video** | *(link to be added — <2 min YouTube)* |
| 📑 **Slide Deck / PPT** | *(link to be added)* |
| 🏆 **Trained LoRA (TRL)** | https://huggingface.co/Harshit-Makraria/the-pivot-lora-trl |
| 🏆 **Trained LoRA (custom)** | https://huggingface.co/Harshit-Makraria/the-pivot-lora |

---

## 12. Step-by-Step Guide

### Prerequisites
```bash
Python 3.10+
CUDA GPU (T4 or better for training)
HuggingFace account (for model push)
W&B account (optional, for logging)
```

### 1. Install & Run the Environment Locally

```bash
# Clone the repo
git clone https://github.com/Harshit-Makraria/meta_scaler
cd meta_scaler

# Install OpenEnv + dependencies
pip install openenv-core>=0.2.3 fastapi uvicorn pydantic

# Start the environment server
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# Open the dashboard
open http://localhost:7860/ui

# Open the AI chat
open http://localhost:7860/chat
```

### 2. Run the Environment via Python Client

```python
from openenv.core import EnvClient

client = EnvClient(base_url="http://localhost:7860")

# Reset with a specific scenario
obs = client.reset(scenario="consumer_app")

# Take an action
result = client.step(action={"action_type": "RESEARCH"})
print(result.observation)
print(result.reward)
print(result.done)
```

### 3. Train the Model (Recommended: TRL Notebook)

1. Open [`training/train_trl.ipynb`](training/train_trl.ipynb) in Google Colab
2. **Runtime → Change runtime type → T4 GPU**
3. Run cells top to bottom:
   - **Cell 1** — GPU check
   - **Cell 2** — Install packages (`pip install openenv-core trl peft bitsandbytes`)
   - **Cell 3** — Verify TRL version
   - **Cell 4** — Mount Google Drive
   - **Cell 5** — Clone repo + imports
   - **Cell 6** — W&B login
   - **Cell 7** — Load Qwen2.5-1.5B + QLoRA
   - **Cell 8** — Helpers + action parser
   - **Cell 9** — Reward function + dataset builder
   - **Cell 10** — 🔥 **TRAIN** (5-tier curriculum, ~60–90 min)
   - **Cell 12** — Push LoRA to HF Hub *(run before Cell 11 to safe-save)*
   - **Cell 11** — Save to Drive + close W&B
   - **Cell 13** — Demo: watch trained agent play all 5 scenarios

### 4. Connect Trained Model to HF Space

After Cell 12 pushes to HF Hub:
1. Go to https://huggingface.co/spaces/Harshit-Makraria/the-pivot/settings
2. **Variables and secrets** → Add new secret:
   ```
   Name:  MODEL_ID
   Value: Harshit-Makraria/the-pivot-lora-trl
   ```
3. **Hardware** → Upgrade to **T4 small** (model needs GPU inference)
4. **Factory reboot** — the chat panel goes live in ~2 minutes

### 5. Run Evaluation Against Baselines

```bash
# Compare trained model vs. 4 baselines across all 5 scenarios
python training/evaluate.py --model_path ./trained_model --n_episodes 50

# Baselines only (no trained model needed)
python training/evaluate.py --baselines_only --n_episodes 50
```

### 6. Play It Yourself (No GPU Needed)

1. Visit https://huggingface.co/spaces/Harshit-Makraria/the-pivot
2. Click **⟳ Reset Episode** to start
3. Select a scenario difficulty (start with B2C SaaS — Easy)
4. Watch the stats — when NPS drops and churn rises, you're entering SATURATION
5. Click **RESEARCH** to reduce signal noise, then **PIVOT** when the window opens
6. Try to survive 60 months without going bankrupt!
7. Or click **💬 AI CHAT → Watch Demo** to see the trained agent play automatically

---

## Project Structure

```
meta_scaler/
├── server/
│   ├── app.py                    # FastAPI + OpenEnv create_app
│   ├── cofounder_environment.py  # Main env (inherits openenv Environment)
│   ├── product_manager.py        # PMF, tech debt, feature velocity
│   ├── team_manager.py           # Headcount, morale, burnout
│   ├── marketing_manager.py      # CAC, brand, pipeline
│   ├── competitor.py             # CompetitorAgent NPC (5 strategies)
│   ├── founder.py                # FounderAgent NPC (advice + burnout)
│   ├── investor.py               # InvestorAgent NPC (milestones)
│   ├── runway.py                 # Burn rate, runway tracking
│   ├── reward.py                 # Balanced scorecard reward function
│   └── prompt_encoder.py         # 50-field obs → LLM prompt
├── models.py                     # CoFounderAction, CoFounderObservation (OpenEnv types)
├── scenarios/                    # 5 JSON scenario configs
├── training/
│   ├── train_trl.ipynb           # HF TRL GRPOTrainer (recommended)
│   ├── train_colab.ipynb         # Custom GRPO
│   ├── baseline_agent.py         # 4 baselines incl. StrategistAgent
│   ├── curriculum.py             # Adaptive 5-tier curriculum
│   └── evaluate.py               # Trained vs. baselines comparison
├── hf_deploy/
│   └── static/                   # Dashboard + Chat UI
├── docs/
│   └── plots/                    # reward_curve.png, loss_curve.png, survival_curve.png
└── client.py                     # OpenEnv EnvClient usage example
```

---

## OpenEnv Compliance

This environment is fully compliant with OpenEnv v0.2.3:

```python
# models.py
from openenv.core.env_server.types import Action, Observation
class CoFounderAction(Action): ...      # ✅ inherits OpenEnv Action
class CoFounderObservation(Observation): ...  # ✅ inherits OpenEnv Observation

# server/cofounder_environment.py
from openenv.core.env_server.interfaces import Environment
class CoFounderEnvironment(Environment): ...  # ✅ inherits OpenEnv Environment

# server/app.py
from openenv.core.env_server.http_server import create_app
app = create_app(ThePivotEnvironment, CoFounderAction)  # ✅ uses OpenEnv server

# client.py
from openenv.core import EnvClient    # ✅ uses OpenEnv client
```

---

*Built for the Meta PyTorch OpenEnv Hackathon 2026 · Scaler × PyTorch · Bangalore · April 25–26*
*Author: Harshit Makraria · [GitHub](https://github.com/Harshit-Makraria/meta_scaler) · [HF Space](https://huggingface.co/spaces/Harshit-Makraria/the-pivot)*
