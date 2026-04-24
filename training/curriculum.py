"""
Adaptive Curriculum for GRPO Training.

What this does (plain English):
- Instead of training on all 5 scenarios randomly, we start with easy ones
  and only unlock harder ones once the model is performing well enough.
- This is called "curriculum learning" — the same way humans learn: easy first.
- It hits the T4 (Self-Improvement) hackathon theme directly.

Difficulty ladder (scenario name → difficulty tier):
  Tier 1 (easy):        b2c_saas
  Tier 2 (medium):      enterprise_saas
  Tier 3 (medium-hard): fintech
  Tier 4 (hard):        marketplace
  Tier 5 (very hard):   consumer_app

Unlock condition: mean_reward > threshold AND survival_rate > 0.5 in last N episodes.

Usage in training loop:
    curriculum = AdaptiveCurriculum()
    scenario = curriculum.sample_scenario()   # always returns a valid dict
    ...
    curriculum.record_result(mean_reward, survived)
    if curriculum.should_advance():
        curriculum.advance_tier()
"""
from __future__ import annotations
import json
import pathlib
import random
from dataclasses import dataclass, field

SCENARIOS_DIR = pathlib.Path(__file__).parent.parent / "scenarios"

# Ordered from easiest to hardest
DIFFICULTY_LADDER: list[str] = [
    "b2c_saas",        # tier 1 — easy
    "enterprise_saas", # tier 2 — medium
    "fintech",         # tier 3 — medium-hard
    "marketplace",     # tier 4 — hard
    "consumer_app",    # tier 5 — very hard
]

# Reward threshold to unlock the next tier
UNLOCK_THRESHOLDS: list[float] = [
    -30.0,   # advance from tier 1 when mean_reward > -30
    -10.0,   # advance from tier 2
    10.0,    # advance from tier 3
    30.0,    # advance from tier 4 (hardest gate)
]

WINDOW_SIZE = 20       # episodes to average before deciding to advance
SURVIVAL_GATE = 0.45   # must survive at least 45% of recent episodes


@dataclass
class AdaptiveCurriculum:
    """
    Tracks which difficulty tier the training run is on and decides when to advance.
    Replay buffer re-uses completed episodes so the model doesn't forget easy tiers.
    """
    seed: int = 42
    current_tier: int = 0                         # 0-indexed into DIFFICULTY_LADDER
    _recent_rewards: list[float] = field(default_factory=list, init=False)
    _recent_survived: list[bool] = field(default_factory=list, init=False)
    _tier_history: list[dict] = field(default_factory=list, init=False)
    _all_scenarios: dict[str, dict] = field(default_factory=dict, init=False)
    _rng: random.Random = field(init=False)

    def __post_init__(self):
        self._rng = random.Random(self.seed)
        self._load_scenarios()

    # ── Public API ────────────────────────────────────────────────────────────

    def sample_scenario(self) -> dict:
        """
        Return a scenario dict to train on.
        80% chance: current tier scenario.
        20% chance: replay from an already-unlocked (easier) tier.
        """
        if self.current_tier > 0 and self._rng.random() < 0.20:
            replay_tier = self._rng.randint(0, self.current_tier - 1)
            name = DIFFICULTY_LADDER[replay_tier]
        else:
            name = DIFFICULTY_LADDER[self.current_tier]

        return self._all_scenarios.get(name, list(self._all_scenarios.values())[0])

    def record_result(self, mean_reward: float, survived: bool):
        """Call after each training episode."""
        self._recent_rewards.append(mean_reward)
        self._recent_survived.append(survived)
        if len(self._recent_rewards) > WINDOW_SIZE:
            self._recent_rewards.pop(0)
            self._recent_survived.pop(0)

    def should_advance(self) -> bool:
        """True if we have enough data and performance is good enough to unlock next tier."""
        if self.current_tier >= len(DIFFICULTY_LADDER) - 1:
            return False
        if len(self._recent_rewards) < WINDOW_SIZE:
            return False
        mean_reward = sum(self._recent_rewards) / len(self._recent_rewards)
        survival_rate = sum(self._recent_survived) / len(self._recent_survived)
        threshold = UNLOCK_THRESHOLDS[self.current_tier]
        return mean_reward > threshold and survival_rate >= SURVIVAL_GATE

    def advance_tier(self) -> bool:
        """Advance to the next difficulty tier. Returns False if already at max."""
        if self.current_tier >= len(DIFFICULTY_LADDER) - 1:
            return False
        self._tier_history.append({
            "from_tier": self.current_tier,
            "scenario": DIFFICULTY_LADDER[self.current_tier],
            "mean_reward": sum(self._recent_rewards) / max(len(self._recent_rewards), 1),
            "survival_rate": sum(self._recent_survived) / max(len(self._recent_survived), 1),
        })
        self.current_tier += 1
        self._recent_rewards.clear()
        self._recent_survived.clear()
        return True

    def status(self) -> dict:
        """Human-readable status for logging."""
        tier_name = DIFFICULTY_LADDER[self.current_tier]
        mean_r = sum(self._recent_rewards) / max(len(self._recent_rewards), 1)
        surv = sum(self._recent_survived) / max(len(self._recent_survived), 1)
        threshold = UNLOCK_THRESHOLDS[self.current_tier] if self.current_tier < len(UNLOCK_THRESHOLDS) else None
        return {
            "tier": self.current_tier + 1,
            "max_tier": len(DIFFICULTY_LADDER),
            "scenario": tier_name,
            "episodes_in_window": len(self._recent_rewards),
            "window_mean_reward": round(mean_r, 1),
            "window_survival_rate": round(surv, 2),
            "unlock_threshold": threshold,
            "ready_to_advance": self.should_advance(),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_scenarios(self):
        for name in DIFFICULTY_LADDER:
            path = SCENARIOS_DIR / f"{name}.json"
            if path.exists():
                with open(path) as f:
                    self._all_scenarios[name] = json.load(f)
        if not self._all_scenarios:
            raise FileNotFoundError(f"No scenario JSONs found in {SCENARIOS_DIR}")
