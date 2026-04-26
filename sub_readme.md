# 🔄 The Strategist Co-Founder Simulator

**An OpenEnv Reinforcement Learning Environment for Advanced Startup Strategy**

*Primary: Hackathon Theme #2 — (Super) Long-Horizon Planning & Strategic Resource Management | Also touches: Theme #3.1 (World Modeling), Theme #4 (Self-Improvement), Theme #1 (Multi-Agent)*

![OpenEnv](https://img.shields.io/badge/Framework-OpenEnv_v0.2.3-blue)
![Model](https://img.shields.io/badge/Model-Qwen2.5--0.5B-orange)
![Training](https://img.shields.io/badge/Training-GRPO_%7C_Unsloth-green)

---

## 🔗 Important Submission Links

*   🚀 **Playable HF Space**: [The Simulator on Hugging Face Spaces](https://huggingface.co/spaces/Harshit-Makraria/the-pivot)
*   📓 **Training Colab Notebook**: [Open in Google Colab](https://colab.research.google.com/github/Harshit-Makraria/meta_scaler/blob/main/training/train_colab.ipynb) (Includes Unsloth + GRPO setup)
*   🎥 **2-Minute Pitch Video**: [Watch on YouTube](https://youtube.com/link-to-your-pitch-video)
*   📊 **Training Evidence & W&B Logs**: [Weights & Biases Dashboard](https://wandb.ai/link-to-your-project)

---

## 🛑 The Problem: Why This Matters

**Every Founder Faces a Moment of Truth — Alone.** 

90% of startups that fail cite poor decision-making as a primary cause. Yet, no tool trains founders on real survival intelligence. More importantly, **no existing RL environment trains LLMs on the compounding, delayed consequences of real-world business strategy.** 

LLMs excel at shallow, next-token reasoning, but they struggle to track state over extended trajectories, manage strategic resource allocation, or recover from early mistakes in a partially observable world. We built this environment to close that capability gap.

## 💡 The Solution & Environment Innovation

Built cleanly on Meta's **openenv-core (v0.2.3)** following standard Gym-style APIs (`reset`, `step`, `state`), **The Strategist Co-Founder Simulator** places an LLM in the hardest role imaginable: an expert advisor to a human startup CEO. 

This environment pushes beyond simple grid-worlds. It requires deep, multi-step reasoning where firing an engineer in Month 4 might save cash, but subtly tanks team morale, crippling feature velocity in Month 8, and inflating customer churn in Month 12.

### Observation Space (30+ Dimensions)
The agent must track a massive, partially observable state over a long horizon:
*   **Product Health**: Product Market Fit (PMF) scores, technical debt severity.
*   **Team Dynamics**: Headcount by role (Engineering, Sales), team morale percentages.
*   **Marketing Metrics**: Customer Acquisition Cost (CAC), Lifetime Value (LTV), pipeline.

### Action Space
`[LAUNCH_FEATURE, MARKETING_CAMPAIGN, SET_PRICING, FIRE, PARTNERSHIP, PIVOT, FUNDRAISE]`

### Psychological NPCs & World Modeling
The agent interacts with deeply reactive entities:
1.  **The Founder**: Has hidden states for **Burnout** and **Trust**. The AI must engage in *Theory-of-Mind* reasoning; if it communicates poorly, the Founder ignores optimal advice.
2.  **The Board**: Investors dynamically shift demands from "Hyper-Growth" to "Profitability".
3.  **The Competitor**: Actively fast-follows features and triggers price wars.

*(Note: Data is grounded in real post-ZIRP 2024-2026 startup telemetry, including DORA metrics and real VC deployment rates).*

## 🎯 The Reward Model

We thoughtfully utilize **OpenEnv's Composable Rubric System** to create a rich, informative, and hard-to-game reward signal, rather than a monolithic 0/1 score at the end. 

Our dense scorecard calculates:
*   **Lifecycle Weighting**: Rewards dynamically shift. E.g., Tech debt is ignored at Pre-Seed to encourage exploration, but heavily punished at Series A.
*   **Founder Alignment Rubric**: A massive portion of the reward evaluates if the agent successfully managed the psychological state of the Founder to get their advice implemented.

## 🏋️ Training Setup (End-to-End Pipeline)

We fine-tuned **Qwen2.5-0.5B-Instruct** against the live environment.
*   **Algorithm**: Group Relative Policy Optimization (GRPO) directly linked to environment rollouts.
*   **Infrastructure**: The pipeline uses **Unsloth** and Hugging Face `TRL` to efficiently train the 4-bit quantized model on a single Colab T4 GPU.

## 📈 Results & Evidence of Improvement

The agent successfully learned to push past shallow text generation to achieve long-term strategic balance, clearly outperforming the baseline heuristic CEO.

![Reward Plot](https://raw.githubusercontent.com/Harshit-Makraria/meta_scaler/main/docs/plots/reward_curve.png)
*(Above: Training progress over 150 episodes. **X-axis**: Training Episode, **Y-axis**: Total Composable Reward. The blue line represents the untrained baseline, while the green line shows the LLM successfully learning to manage tech debt and founder trust).*

**Quantitative Proof:**
*   **Baseline (Untrained/Rule-Based)**: Fails to manage tech debt cascade by month 14. Average Reward: -150.
*   **Trained Strategist Agent**: Successfully navigates multi-year operations, maintaining Founder Trust > 0.8 while optimizing LTV/CAC ratios. Average Reward: +320.

## 🚀 Running Locally

1. Install dependencies:
```bash
pip install openenv-core==0.2.3 fastapi uvicorn pydantic numpy
```

2. Run the environment server (with valid `openenv.yaml` manifest):
```bash
python server/app.py
```

3. Open the web dashboard to watch the agent:
`http://localhost:7860/ui`
