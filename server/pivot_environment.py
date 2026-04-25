"""
pivot_environment.py — backward-compatibility shim.

The environment was renamed to CoFounderEnvironment in cofounder_environment.py
per plan.md Step 1. This file re-exports everything so existing imports don't break.

All new code should import from server.cofounder_environment directly.
"""
from server.cofounder_environment import (
    CoFounderEnvironment,
    CoFounderEnvironment as ThePivotEnvironment,   # legacy alias
    MAX_STEPS,
)

__all__ = ["ThePivotEnvironment", "CoFounderEnvironment", "MAX_STEPS"]
