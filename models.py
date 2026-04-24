"""
Data models for The Pivot environment.
Both classes inherit from openenv-core base types so the framework
can auto-serialize them, expose /schema, and validate WebSocket messages.
"""
from enum import Enum
from typing import Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class ActionType(str, Enum):
    EXECUTE = "EXECUTE"       # Keep running current strategy (default)
    PIVOT = "PIVOT"           # Change core strategy — costly but sometimes necessary
    RESEARCH = "RESEARCH"     # Spend half a month to get clearer market signals
    FUNDRAISE = "FUNDRAISE"   # Go to investor for money
    HIRE = "HIRE"             # Add headcount — raises burn, lowers future pivot cost
    CUT_COSTS = "CUT_COSTS"   # Cut spend — lowers burn, slows growth
    SELL = "SELL"             # Acqui-hire escape: graceful exit if runway < 3mo


class PivotAction(Action):
    """What the agent sends to the environment each step."""
    action_type: ActionType = Field(default=ActionType.EXECUTE)
    action_params: Optional[dict] = Field(default=None)


class PivotObservation(Observation):
    """
    What the agent receives after each step.
    Inherits `done`, `reward`, and `metadata` from Observation base.
    All market signals are noisy — the agent never sees true phase directly.
    """
    # Financial state
    runway_remaining: int = Field(default=18, description="Months of cash left at current burn")
    monthly_revenue: float = Field(default=45000.0, description="Revenue this month in USD")
    burn_rate: float = Field(default=120000.0, description="Monthly spend in USD")
    revenue_delta_3m: float = Field(default=0.0, description="Revenue % change over last 3 months")

    # Noisy market signals (true phase is hidden)
    churn_rate: float = Field(default=0.12, description="Fraction of users leaving per month")
    nps_score: int = Field(default=52, description="Net Promoter Score (-100 to 100)")
    user_complaints: list[str] = Field(default_factory=list, description="2–5 complaint strings this month")
    competitor_launched: bool = Field(default=False, description="Competitor made a major move this month")

    # Competitor intelligence
    competitor_play: str = Field(default="dormant", description="Competitor's current strategy: dormant/launch_feature/price_war/talent_raid/aggressive_mkt")
    competitor_strength: float = Field(default=0.3, description="How formidable the competitor is right now (0–1, grows over time)")

    # Founder state (Ghost Protocol — advice degrades under financial pressure)
    founder_advice: str = Field(default="stay_course", description="Founder's strategic recommendation")
    founder_confidence: float = Field(default=1.0, description="How confident the founder sounds (0–1)")
    investor_sentiment: float = Field(default=0.65, description="How positive the investor feels (0–1)")

    # Investor / milestone
    next_milestone: str = Field(default="Show 10% MoM revenue growth", description="What investor wants now")
    pivot_cost_estimate: int = Field(default=3, description="Months of runway a pivot would cost")

    # Trend signals — computed from last 3 steps, easier for LLM to reason about
    revenue_trend: str = Field(default="growing", description="'growing' / 'plateauing' / 'declining'")
    churn_trend: str = Field(default="stable", description="'stable' / 'rising' / 'spiking'")
    complaint_shift_detected: bool = Field(default=False, description="Complaint types changed character vs last month")
    months_at_risk: int = Field(default=0, description="Consecutive months where runway < 6")

    # Shock events (macro surprises — funding winter, viral moment, key hire quits, etc.)
    active_shock: str = Field(default="", description="Name of macro shock event active this step, or empty string")
    shock_message: str = Field(default="", description="Human-readable description of the shock event")

    # Board pressure (kicks in after step 40 when metrics are bad)
    board_pressure: bool = Field(default=False, description="True when board is applying ultimatum pressure (step>40, runway<6)")

    # Episode metadata
    step: int = Field(default=0, description="Current month number (0-indexed)")
    max_steps: int = Field(default=60, description="Total months in this episode")
