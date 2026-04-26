---
title: "I Trained a 1.5B Model to Be a Co-Founder. Here's What It Learned."
thumbnail: /blog/assets/founderos/thumbnail.png
authors:
  - user: Harshit-Makraria
tags:
  - reinforcement-learning
  - openenv
  - grpo
  - qwen
  - lora
  - rl-environment
---

# I Trained a 1.5B Model to Be a Co-Founder. Here's What It Learned.

There's a strange asymmetry in how we evaluate language models. We have benchmarks that measure whether a model can write working code, solve math problems, follow multi-step instructions, and use 200 different tools. We don't have a benchmark that measures something most actual jobs require: **knowing when to keep going and when to completely change direction.**

That's the gap I tried to close at the Meta PyTorch OpenEnv Hackathon. The result is FounderOS — a startup simulation where an LLM plays the founding CEO for sixty months, makes one strategic call per month, and learns from the consequences. After about three hours of training on a free Colab T4, a 1.5B parameter Qwen model with a tiny LoRA adapter started outperforming a hand-coded MBA-style baseline that took me a week to tune. This post is about how that happened, what surprised me, and why I think this kind of environment matters beyond startups.

---

## The thing pre-training cannot teach

If you ask a frontier LLM "should I pivot my startup?", you get one of two answers. Either it nods along with whatever the founder is already thinking — sycophancy disguised as advice — or it gives you a beautifully formatted bullet list of considerations that nobody actually uses to make decisions. Neither is what a real co-founder does. A real co-founder sits across from you, looks at three months of churn data, glances at your face, and says *"this is the moment, and we both know it."*

That kind of advising is the product of years of seeing startups die in specific, observable ways. Founders read the same Paul Graham essays the LLM did, but founders also remember the company that ignored a 2% monthly churn drift for six months until it was a 22% cliff. They remember the team that pivoted three times in a year and lost everyone good. They remember the one acquihire that left at the right second.

You can't get that from text. You get it from consequences. Which means you can't get it from pre-training — but you can get it from RL.

---

## What FounderOS actually is

FounderOS is a 60-month simulation that conforms to the OpenEnv spec. The agent gets one decision per month and chooses from twelve actions, grouped roughly into three phases of company life:

```
GROWTH       EXECUTE   RESEARCH   LAUNCH_FEATURE   HIRE
SCALE        FUNDRAISE   MARKETING_CAMPAIGN   PARTNERSHIP   SET_PRICING
TRIAGE       CUT_COSTS   FIRE   PIVOT   SELL
```

The catch — and this is the whole point — is that the market silently progresses through three hidden phases (GROWTH → SATURATION → DECLINE) at scenario-dependent timing the agent can never see. It has to infer the phase from noisy, partial signals: a churn trend, an NPS drift, the kind of complaints starting to appear in the support inbox. There's no "you are in DECLINE now" signal. The agent has to feel it.

Five scenarios run on top of this engine, each calibrated against real startup data — Carta's 2024 burn-rate medians, OpenView's 2023 SaaS growth percentiles, CB Insights post-mortem patterns. The Consumer App scenario, the hardest one, has its phase change set at month 23 because that's roughly when Twitter pivoted from Odeo. The Marketplace scenario changes at month 29 to mirror two-sided take-rate compression we've seen in real GMV histories.

Three NPCs are alive in the simulation:

- A **Founder** who gives advice and burns out under bad results. When calm, the founder's advice is about 82% useful. When burnout exceeds 0.7, the same advice becomes actively misleading — the founder defaults to denial or panic. Trusting the human blindly is a learnable trap.
- An **Investor** who tracks milestones with shifting requirements at steps 20 and 40. Sentiment is visible; the actual funding thresholds are not. The agent has to figure out what makes the cheque clear.
- A **Competitor** that picks a strategy based on your weaknesses. High burn → price war. High churn → talent raid. PIVOT → it tries to grab the vacuum. It reacts to *your moves* with a one-step lag, which makes the action space feel adversarial in a way that's hard to write rules against.

These pieces don't sit in isolation. Hire too aggressively when morale is already low and morale collapses further. Ship a feature with high tech debt and the next marketing campaign drives more churn instead of less. The whole thing is modelled physics, not a spreadsheet — actions cascade across subsystems with realistic delays, which is what makes the agent's job non-trivial.

---

## Designing a reward you can't game

This is where most simulation environments quietly cheat. They reward survival, the agent learns to play defense, and you've trained a coward. Or they reward growth, the agent learns to ignore churn until it dies in month 38, and you've trained a sociopath.

I wanted a reward that would make the agent miserable if it tried any single-axis strategy. The balanced scorecard:

- 30% survival rate — did the company live to month 60?
- 20% PMF score — does the product actually work?
- 20% team morale — is the team healthy enough to keep building?
- 15% LTV:CAC — are unit economics sustainable?
- 15% founder trust — does the team accept your calls?

On top of the scorecard there's an event layer. A pivot at the right moment scores +50; a panic pivot in month 8 scores −10; a wasteful pivot when business is healthy scores −20. Selling the company too early when runway was fine scores −40, but selling at the last second when bankruptcy was inevitable scores +50. Bold moves under board scrutiny score; blindly executing while the company bleeds out is penalised. There's even a +5 reward for surviving random catastrophic events — a key hire quitting, a funding winter — because a co-founder who never had to handle disaster never proved anything.

The result is that "always EXECUTE" gets you trapped in DECLINE and dies. "Always PIVOT" burns cash and destroys morale. "Hire to grow" without watching morale collapses the product. Only a balanced, well-timed sequence scores. That's the whole game.

---

## Training: small model, small adapter, free GPU

Hardware: one T4 in Google Colab. Total cost: zero.

```
Base model      Qwen/Qwen2.5-1.5B-Instruct
PEFT            QLoRA — 4-bit nf4, bfloat16 compute
LoRA config     r=8, alpha=16, dropout=0.05
Targets         q_proj + v_proj
Trainable       ~3 million parameters (0.2% of total)
Algorithm       GRPO — Group Relative Policy Optimization
Curriculum      5 tiers, advance at 45% survival on 20-ep moving average
```

Five tiers of difficulty, from B2C SaaS at the easy end to Consumer App Viral Decay at the very hard end. The agent only unlocks the next tier after proving competence on the current one, and 20% of every batch replays earlier tiers to prevent forgetting. Total trainable parameters: about three million. This is comically small compared to the 1.5 billion parameters in the base — but that's the point. The base model already speaks fluent business; the LoRA adapter just steers it toward decisions that survive consequences.

Each GRPO step samples four completions per prompt, the environment runs a 10-step lookahead to compute a reward signal for each completion, and the policy moves toward the better-rewarded ones. KL penalty against the reference model keeps the agent from collapsing into a degenerate one-action policy. Reward scaling normalises the advantage so the gradient isn't dominated by single huge episodes.

Wall time: about 90 minutes for the TRL recipe, 2–3 hours for the custom GRPO loop in the bundled notebook.

---

## What the curves actually show

### Reward over training

![Reward Curve](https://raw.githubusercontent.com/Harshit-Makraria/meta_scaler/main/docs/plots/reward_curve.png)

Early episodes are deeply negative — the agent picks EXECUTE, EXECUTE, EXECUTE while the market quietly slides into DECLINE around it, runs out of runway in month 35, and gets the bankruptcy penalty. As GRPO concentrates the policy, the reward climbs into positive territory. The oscillation is not noise; it's entropy decaying as the policy concentrates on better strategies. A perfectly smooth curve here would mean GRPO had over-collapsed and was no longer exploring.

### Loss over training

![Loss Curve](https://raw.githubusercontent.com/Harshit-Makraria/meta_scaler/main/docs/plots/loss_curve.png)

Initial high loss reflects the agent exploring widely. As it figures out which actions correlate with reward in which contexts, the loss decreases and stabilises. The shape is the most important thing here — a flat zero-loss curve would mean the gradient isn't flowing through the LoRA adapter at all. This shape says training is working.

### Survival per episode

![Survival Curve](https://raw.githubusercontent.com/Harshit-Makraria/meta_scaler/main/docs/plots/survival_curve.png)

Early agent dies before month 30 most of the time — it executes its way into bankruptcy. Late agent survives much longer, frequently hitting the 60-step ceiling. This is the most direct evidence that training improved real strategic behaviour, not just reward-hacking some quirk of the score function.

---

## The honest comparison

I built four baselines specifically to make the trained model earn its place:

| Agent | Strategy | Mean Reward | Survival |
|---|---|---:|---:|
| Random | Picks an action uniformly | −45.2 | 8% |
| Stubborn | Always EXECUTE | −38.1 | 12% |
| Panic | Pivots at the first danger signal | −22.6 | 24% |
| Strategist | Hand-coded MBA-level rules | +3.4 | 62% |
| **Trained LLM** | **GRPO on 5 scenarios** | **+4.1** | **67%** |

The Strategist baseline is the one that matters. I spent a real week making it good — it knows to fundraise when runway dips below 8 months and NPS is healthy, it knows to research before pivoting under ambiguous signals, it knows to cut costs aggressively when burn-to-revenue ratio crosses 2.5x. It's a strong policy.

The trained LLM beats it. Not by a landslide — by about 8% on survival rate and a small margin on mean reward. But that small margin is the whole game. A model that can't beat a deterministic rule-based baseline didn't learn anything; it just memorised the most common action. A model that consistently outperforms it learned the kind of nuance that only emerges from experiencing consequences across hundreds of episodes — exactly what RL is supposed to teach.

The most interesting failure modes were also the most informative. Early in training, the model panic-pivoted at the first sign of trouble — three or four pivots per episode, burning cash, never recovering. Mid training, it swung the other way, becoming stubborn and refusing to pivot even when the data screamed for it. By the end, it was running zero or one pivot per episode, but at the *right* month — which is exactly what a real co-founder does. That trajectory, from panic to denial to discipline, is the same one human founders go through over the course of their careers. Watching a model traverse it in three hours of compute was the most surprising part of the project.

---

## What you can actually do with this

The code, the trained LoRA, and the live demo are all open. There's a six-tab dashboard you can play with — make your own 60 strategic decisions, watch the trained agent auto-play a full episode, paste your own startup metrics and get a recommendation, run counterfactual replays ("what if I had pivoted at month 25 vs month 50?"), see how the four baselines stack up against each other.

I think there are at least three groups of people who could use this:

**Founders** — paste real metrics in, get a specific recommendation with reasoning that cites your specific numbers. It's not a replacement for an experienced advisor, but at 2 a.m. when you need to decide whether the dip you're seeing is the moment or just a bad week, it's better than nothing.

**Business schools** — the counterfactual API lets students ask "what if?" and run it. The reward curve and survival rate make the consequences legible in a way that case-study discussions never quite manage.

**RL researchers** — the hidden phase transition is a clean covariate-shift problem. The same problem appears in ad-spend optimisation under trend reversals, portfolio management under regime changes, inventory under demand pattern shifts. If you want a benchmark for sequential decision-making under non-stationarity, FounderOS is one.

---

## What's next

The LoRA adapter is 12 megabytes. The base model is 1.5B parameters. The whole training run fits on a free GPU. None of this is the limiting factor.

The limiting factor is the environment. The next thing I want to add is a real conversational layer — instead of the agent emitting `DECISION: pivot`, I want it to talk to the founder NPC, explain its reasoning, watch trust go up or down based on whether the founder accepts the call, and have *that* trust be the thing that gates whether the recommendation actually executes. That's the part of co-founding that's hardest to model and most important to get right. A 3B advisor variant of the training notebook is in progress.

I also want to push the curriculum further — currently the 5 scenarios cover SaaS, enterprise, fintech, marketplace, consumer. Adding hardware, biotech, and infrastructure would test whether the policy generalises beyond software economics. And I want to compare the trained Qwen to GPT-4 and Claude on the same environment — not fine-tuned, just zero-shot. The four-baseline ladder gives you a clean place to drop them in.

If pre-training gives an LLM a library of every business book ever written, RL gives it the experience of running into walls. I think that's the more important capability for the kinds of decisions humans actually need help with. FounderOS is a small first step toward language models that aren't just informed about strategy — they're trained on its consequences.

---

*FounderOS was built solo at the Meta PyTorch OpenEnv Hackathon 2026 (Scaler × PyTorch · Bangalore · April 25–26). The trained LoRA adapter, training notebooks, environment code, baseline agents, and live demo are all open. Total compute spend across the project: zero — everything runs on Colab's free T4 tier.*
