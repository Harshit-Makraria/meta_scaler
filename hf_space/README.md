---
title: The Pivot
emoji: 🔄
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
tags:
  - openenv
  - reinforcement-learning
  - rl-environment
  - startup-simulation
  - grpo
  - meta-pytorch-hackathon-2026
---

# 🔄 The Pivot — OpenEnv Startup Founder Simulation

An OpenEnv-compliant RL environment where an LLM founder agent must detect hidden
market phase shifts and decide when to pivot strategy — without being told.

- **Dashboard**: visit `/ui`
- **API docs**: visit `/docs`
- **Baseline comparison**: `/compare?scenario=b2c_saas&n_episodes=10`
- **OpenEnv endpoints**: `/reset`, `/step`, `/state`, `/ws`, `/schema`

## Features
- 5 difficulty scenarios (b2c_saas → consumer_app)
- Rule-based competitor, investor, and internal team dynamics
- Noisy signal system with research-based noise reduction
- Adaptive curriculum for training

## Links
- GitHub: https://github.com/Harshit-Makraria/meta_scaler
- Training notebook: `training/train_colab.ipynb`

Built for the **Meta PyTorch OpenEnv Hackathon 2026** (Scaler × PyTorch).
