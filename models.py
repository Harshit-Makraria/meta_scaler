"""
Data models for the CoFounder Strategist environment.
Both classes inherit from openenv-core base types so the framework
can auto-serialize them, expose /schema, and validate WebSocket messages.

Expanded from the original Pivot-only model to a full 30+ field
multi-dimensional state space covering Product, Team, Marketing,
Financial, Market, and Competitor dynamics.
"""
from enum import Enum
from typing import Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class ActionType(str, Enum):
    # ── Original 7 actions (preserved) ───────────────────────────────────────
    EXECUTE  = "EXECUTE"       # Keep running current strategy (default)
    PIVOT    = "PIVOT"         # Change core strategy — costly but sometimes necessary
    RESEARCH = "RESEARCH"      # Spend half a month to get clearer market signals
    FUNDRAISE = "FUNDRAISE"    # Go to investor for money
    HIRE     = "HIRE"          # Add headcount — raises burn, lowers future pivot cost
    CUT_COSTS = "CUT_COSTS"    # Cut spend — lowers burn, slows growth
    SELL     = "SELL"          # Acqui-hire escape: graceful exit if runway < 3mo

    # ── New Strategist Co-Founder actions (Step 2 of plan.md) ────────────────
    LAUNCH_FEATURE     = "LAUNCH_FEATURE"      # Ship a product feature — uses eng capacity, can improve PMF or add tech debt
    MARKETING_CAMPAIGN = "MARKETING_CAMPAIGN"  # Run a campaign — costs burn, lowers CAC, raises brand awareness
    SET_PRICING        = "SET_PRICING"          # Adjust pricing tier — affects revenue, churn, and NPS
    FIRE               = "FIRE"                 # Lay off a team member — reduces burn but crushes morale
    PARTNERSHIP        = "PARTNERSHIP"          # Form a channel partner deal — reduces CAC, boosts pipeline


class CoFounderAction(Action):
    """What the agent (strategist) sends to the environment each step."""
    action_type: ActionType = Field(default=ActionType.EXECUTE)
    action_params: Optional[dict] = Field(default=None)


# Backward-compatibility alias
PivotAction = CoFounderAction


class CoFounderObservation(Observation):
    """
    What the agent receives after each step.
    Inherits `done`, `reward`, and `metadata` from Observation base.

    30+ fields spanning all dimensions the plan.md specifies:
      - Financial state (6 fields)
      - Market signals / noisy (8 fields)
      - Product health (5 fields)
      - Team dynamics (5 fields)
      - Marketing metrics (5 fields)
      - Founder NPC state (5 fields)
      - Investor / Board state (4 fields)
      - Episode metadata (3 fields)
    """

    # ── Financial state (6 fields) ────────────────────────────────────────────
    runway_remaining: int = Field(default=18,        description="Months of cash left at current burn")
    monthly_revenue:  float = Field(default=45000.0, description="Revenue this month in USD")
    burn_rate:        float = Field(default=120000.0, description="Monthly spend (payroll + infra + marketing) in USD")
    revenue_delta_3m: float = Field(default=0.0,     description="Revenue % change over last 3 months")
    net_cash_flow:    float = Field(default=0.0,     description="monthly_revenue - burn_rate; negative = burning cash")
    total_raised_usd: float = Field(default=0.0,     description="Cumulative funding received so far")

    # ── Noisy market signals (true phase is hidden) (8 fields) ───────────────
    churn_rate:               float      = Field(default=0.12,    description="Fraction of users leaving per month (noisy)")
    nps_score:                int        = Field(default=52,      description="Net Promoter Score -100 to 100 (noisy)")
    user_complaints:          list[str]  = Field(default_factory=list, description="2-5 complaint strings this month")
    competitor_launched:      bool       = Field(default=False,   description="Competitor made a major move this month")
    competitor_play:          str        = Field(default="dormant",description="Competitor strategy: dormant/launch_feature/price_war/talent_raid/aggressive_mkt")
    competitor_strength:      float      = Field(default=0.3,     description="Competitor formidability 0-1, grows over time")
    revenue_trend:            str        = Field(default="growing",description="'growing' / 'plateauing' / 'declining'")
    churn_trend:              str        = Field(default="stable", description="'stable' / 'rising' / 'spiking'")

    # ── Product health (5 fields) ─────────────────────────────────────────────
    pmf_score:              float = Field(default=0.55, description="Product-market fit score 0-1 (derived from NPS + complaints)")
    tech_debt_ratio:        float = Field(default=0.15, description="Tech debt as fraction of eng capacity 0-1 (higher = slower)")
    tech_debt_severity:     str   = Field(default="low",description="'low' / 'medium' / 'high' / 'critical'")
    features_shipped_last:  int   = Field(default=1,    description="Features shipped in last month (0 if team is overwhelmed)")
    feature_pipeline_depth: int   = Field(default=3,    description="Features queued and ready to build")

    # ── Team dynamics (5 fields) ──────────────────────────────────────────────
    team_size:        int   = Field(default=8,    description="Total headcount across all roles")
    eng_headcount:    int   = Field(default=4,    description="Engineering team size")
    sales_headcount:  int   = Field(default=2,    description="Sales team size")
    support_headcount:int   = Field(default=2,    description="Support team size")
    team_morale:      float = Field(default=0.75, description="Team morale 0-1; below 0.4 velocity drops sharply")

    # ── Marketing metrics (5 fields) ──────────────────────────────────────────
    cac:              float = Field(default=1500.0,description="Customer acquisition cost in USD")
    ltv:              float = Field(default=6000.0,description="Estimated customer lifetime value in USD")
    ltv_cac_ratio:    float = Field(default=4.0,   description="LTV / CAC ratio; >3 = healthy, <1 = unsustainable")
    brand_awareness:  float = Field(default=0.20,  description="Brand recognition score 0-1")
    pipeline_generated:float= Field(default=0.0,   description="New sales pipeline value created this month in USD")

    # ── Founder NPC state (5 fields) ──────────────────────────────────────────
    founder_advice:      str   = Field(default="stay_course",description="Founder's strategic recommendation")
    founder_confidence:  float = Field(default=1.0,          description="How confident the founder sounds 0-1")
    founder_trust:       float = Field(default=0.80,         description="Founder's trust in the advisor agent 0-1")
    founder_burnout:     float = Field(default=0.10,         description="Founder burnout level 0-1; above 0.7 impairs judgment")
    founder_stubbornness:float = Field(default=0.30,         description="Probability founder ignores advisor advice 0-1")

    # ── Investor / Board state (4 fields) ─────────────────────────────────────
    investor_sentiment: float     = Field(default=0.65,   description="How positive the investor feels 0-1")
    next_milestone:     str       = Field(default="Show 10% MoM revenue growth", description="What investor wants now")
    board_pressure:     bool      = Field(default=False,  description="True when board is applying ultimatum pressure (step>40, runway<6)")
    board_demands:      list[str] = Field(default_factory=list, description="Specific demands from the board this month")

    # ── Trend / shock signals (4 fields) ──────────────────────────────────────
    complaint_shift_detected: bool = Field(default=False, description="Complaint types changed character vs last month")
    months_at_risk:           int  = Field(default=0,     description="Consecutive months where runway < 6")
    active_shock:             str  = Field(default="",    description="Name of macro shock event active this step, or empty string")
    shock_message:            str  = Field(default="",    description="Human-readable description of the shock event")

    # ── Pivot economics (2 fields) ────────────────────────────────────────────
    pivot_cost_estimate: int   = Field(default=3,     description="Months of runway a pivot would cost")
    pivot_cost_months:   float = Field(default=3.0,   description="Precise pivot cost including team ramp-up time")

    # ── Episode metadata (3 fields) ───────────────────────────────────────────
    step:      int = Field(default=0,  description="Current month number (0-indexed)")
    max_steps: int = Field(default=60, description="Total months in this episode")
    scenario:  str = Field(default="default", description="Name of the active scenario")


# Backward-compatibility alias
PivotObservation = CoFounderObservation
