# 🔄 The Pivot — Writeup

**Meta PyTorch OpenEnv Hackathon 2026 submission**
Author: Harshit Makraria · [GitHub](https://github.com/Harshit-Makraria/meta_scaler)

---

## The idea in one line

Most LLM-agent benchmarks test whether an agent can *follow a plan*. **The Pivot** tests whether it can detect that the plan is no longer the right one — and decide when to abandon it.

## The setup

A founder LLM plays 60 simulated months at an imaginary startup. Each month it picks one of six actions (`EXECUTE`, `PIVOT`, `RESEARCH`, `FUNDRAISE`, `HIRE`, `CUT_COSTS`). The market silently walks through three hidden phases — GROWTH, SATURATION, DECLINE — with scenario-dependent timing that is never revealed. If the founder keeps executing into the decline, they run out of runway. If they pivot too early, they torch cash on a product that was still working. The sweet spot is a narrow window (usually 3–5 months) where noisy signals (NPS, churn, CAC/LTV, rival moves) start tipping.

Five scenarios span the difficulty ladder, from a gentle B2C SaaS decline at month 36 to a brutal consumer-app cliff at month 23.

## Why it's interesting

Three things make this more than a toy env:

1. **Hidden-phase detection.** The agent has to build an implicit world model — no oracle, no phase indicator, just noisy KPIs. RESEARCH actions reduce noise at the cost of a month.
2. **A live rival.** A rule-based competitor picks from 5 strategies (DORMANT, LAUNCH_FEATURE, PRICE_WAR, TALENT_RAID, AGGRESSIVE_MKT) based on your weakness, and grows from strength 0.2 → 1.0 over 60 steps. Decisions that looked safe in isolation start bleeding revenue when the rival counter-punches.
3. **Irreversibility.** PIVOT costs 3 months of runway and resets revenue to 60%. You can only afford to do it once or twice. The agent can't brute-force explore.

## Training

- **Base model**: Qwen2.5-0.5B-Instruct (0.5B fits QLoRA comfortably on a T4)
- **PEFT**: LoRA r=8, q_proj + v_proj only
- **Quantization**: 4-bit nf4 with bfloat16 compute
- **Algorithm**: GRPO with return-to-go advantages within each 60-step episode
- **Exploration**: ε-greedy (ε=0.3) forced during rollouts — without this, the base 0.5B model always emits `"execute"` and the group-relative advantage collapses to zero
- **Curriculum**: Adaptive 5-tier. Unlocks the next tier only when the 20-ep moving-average reward beats a threshold AND ≥45% survival. 20% of sampled episodes are replayed from already-unlocked tiers to prevent catastrophic forgetting.

Training runs in 60–90 minutes on a Colab T4. 150 episodes per run.

## What worked

- **Return-to-go over the whole episode** as the GRPO group gave the cleanest advantage signal. The original per-step group with a sampled-twin completion collapsed because both samples were identical with a 0.5B base model.
- **`enable_input_require_grads()`** was the fix for gradients flowing through 4-bit frozen base into LoRA. Gradient checkpointing + `prepare_model_for_kbit_training` broke the graph.
- **Tokenize completion first, then truncate the prompt from the left** — the single subtlest bug. With a 512-token cap and long prompts, completions were being silently chopped to zero tokens, so every `log_prob` returned a detached `tensor(0.0)` → loss was exactly 0 for hundreds of episodes before I caught it.

## What's still rough

- 150 episodes on 0.5B is barely enough for the curriculum to unlock past tier 2. The infrastructure is ready for 500–1000 episodes on a bigger model; I didn't have the compute budget in 24h.
- No KL penalty against a reference model yet. Would prevent reward hacking as training scales.
- Pivot quality is scored via a hand-tuned window match. A learned reward model would be more robust.

## Real-world application

The same shape applies to any long-horizon decision-under-drift problem: **ad spend allocation, portfolio rebalancing, roadmap planning, inventory management**. The core research contribution here is the eval framework — hidden-phase scenarios with a scored pivot window — which transfers to any of those domains with a new scenario file.

I plan to ship an "Advisor Mode" where a real founder pastes their actual MRR/burn/NPS numbers and the trained policy returns a recommendation plus its top-k action probabilities with reasoning. That's the follow-up past the hackathon deadline.

## Links

- **Live demo (HF Space)**: https://huggingface.co/spaces/Harshit-Makraria/the-pivot
- **Code**: https://github.com/Harshit-Makraria/meta_scaler
- **Training notebook**: [`training/train_colab.ipynb`](../training/train_colab.ipynb)
- **Plots**: [`docs/plots/`](plots/)
