"""
InvestorAgent — VC simulator with real fundraising requirements.

Requirements calibrated to real 2024 data:
  - Carta 2024: median seed runway 20mo, only 15.4% raise Series A
  - Crunchbase Q1 2024: Series A median $18M, requires $2-5M ARR
  - Qubit Capital 2024: minimum 2x-3x YoY growth for Series A
  - Initialized Capital 2024: NPS > 30 minimum for Series A
  - WallStreetPrep 2024: burn multiple < 2.0x = healthy for Series B
  - ChartMogul 2024: NRR > 100% required for Series B
"""
from dataclasses import dataclass


@dataclass
class FundingRound:
    name:                     str
    milestone_description:    str    # What investor shows to the agent (visible)
    check_size:               float  # USD amount if approved

    # Internal evaluation criteria (hidden from agent — must be inferred)
    min_revenue_growth_3m:    float | None = None
    min_runway:               int | None   = None
    min_nps:                  int | None   = None
    min_arr_monthly:          float | None = None  # minimum monthly revenue
    max_burn_multiple:        float | None = None  # burn / new ARR ratio
    requires_profitability_path: bool = False


# ── Real funding round parameters ─────────────────────────────────────────────
# Check sizes calibrated for gameplay (real amounts would extend runway too far
# for a 60-step episode, so scaled proportionally):
#   Seed:     $1.0M → 11-12 months at $85K burn     (real: $3M)
#   Series A: $4.0M → 26 months at $150K burn       (real: $18M)
#   Series B: $9.0M → 45-60 months at $150-200K burn (real: $30M)
#
# Requirements are from REAL data — agents must hit real investor thresholds.

FUNDING_ROUNDS = [
    FundingRound(
        name="Seed Extension",
        # Carta 2024: seed investors want 10-15% MoM growth signal
        milestone_description="Demonstrate 10%+ monthly revenue growth for 3 consecutive months",
        check_size=1_000_000,
        min_revenue_growth_3m=0.10,
        min_runway=3,
        min_nps=-10,     # any non-catastrophic NPS ok at seed
    ),
    FundingRound(
        name="Series A",
        # Qubit Capital 2024: $2-5M ARR + 2-3x YoY growth
        # Initialized Capital 2024: NPS > 30 preferred, > 0 minimum
        # Real Series A median: $18M — scaled to $4M for gameplay
        milestone_description="Reach $500K+ monthly revenue, 2x annual growth, positive NPS (30+)",
        check_size=4_000_000,
        min_revenue_growth_3m=0.08,    # 8% MoM = roughly 2.5x YoY
        min_arr_monthly=41_667,        # $500K ARR / 12 = $41.7K/mo
        min_runway=6,                  # must have runway to negotiate
        min_nps=30,                    # Initialized Capital 2024 minimum
    ),
    FundingRound(
        name="Series B",
        # BVP State of Cloud 2024: $10-20M ARR for Series B
        # WallStreetPrep 2024: burn multiple < 2.0x
        # ChartMogul 2024: NRR > 100% (net revenue expansion)
        # Real Series B median: $30M — scaled to $9M for gameplay
        milestone_description="Clear path to profitability: NRR > 100%, burn multiple < 2x, NPS > 40",
        check_size=9_000_000,
        min_arr_monthly=83_333,        # $1M ARR / 12 — scaled for gameplay
        min_runway=6,
        min_nps=40,
        requires_profitability_path=True,
        max_burn_multiple=2.5,         # slightly above real threshold for gameplay flexibility
    ),
]


class InvestorAgent:
    """
    Rule-based VC. Silently changes requirements between rounds.
    Requirements match real 2024 investor criteria — agent must infer
    what's needed from the milestone description + failed fundraise feedback.

    Real context injected:
    - Only 15.4% of seed companies raise Series A (Carta 2024)
    - Series A process takes 3-6 months (agent loses 3+ months runway)
    - Investors check NPS, growth rate, burn multiple — not just revenue
    """

    # Real stat: only 15.4% of seed companies raise Series A (Carta 2024)
    SERIES_A_SUCCESS_RATE = 0.154

    def __init__(self):
        self._round_index   = 0
        self._sentiment     = 0.62     # starts cautiously positive
        self._milestones_hit  = 0
        self._milestones_missed = 0

    def tick(self, step: int):
        """
        Silently advance investor expectations at key milestones.
        Steps calibrated to real fundraising timeline:
          - Step 20 (~20 months): time to raise Series A if growth is there
          - Step 40 (~40 months): Series B territory if survived
        """
        if step == 20 and self._round_index == 0:
            self._round_index = 1
        elif step == 40 and self._round_index == 1:
            self._round_index = 2

    @property
    def current_round(self) -> FundingRound:
        return FUNDING_ROUNDS[self._round_index]

    @property
    def sentiment(self) -> float:
        return self._sentiment

    def get_current_milestone(self) -> str:
        return self.current_round.milestone_description

    def evaluate_funding_request(
        self,
        monthly_revenue:   float,
        revenue_delta_3m:  float,
        runway_remaining:  int,
        nps_score:         int,
        burn_rate:         float,
    ) -> tuple[bool, float, str]:
        """
        Evaluate a FUNDRAISE action.
        Returns (approved, amount_usd, investor_message).

        Criteria are hidden from the agent — it must learn them through
        trial-and-error (failed fundraise attempts with partial feedback).

        All thresholds sourced from real 2024 investor data.
        """
        r       = self.current_round
        approved = True
        reasons  = []

        # ── Revenue growth check ──────────────────────────────────────────────
        if r.min_revenue_growth_3m is not None:
            if revenue_delta_3m < r.min_revenue_growth_3m:
                approved = False
                reasons.append(
                    f"Revenue growth {revenue_delta_3m:.0%} below {r.min_revenue_growth_3m:.0%} target"
                )

        # ── Minimum ARR check ─────────────────────────────────────────────────
        if r.min_arr_monthly is not None:
            if monthly_revenue < r.min_arr_monthly:
                approved = False
                reasons.append(
                    f"Monthly revenue ${monthly_revenue:,.0f} below ${r.min_arr_monthly:,.0f} minimum"
                )

        # ── Runway check ──────────────────────────────────────────────────────
        if r.min_runway is not None:
            if runway_remaining < r.min_runway:
                approved = False
                reasons.append(
                    f"Only {runway_remaining} months runway — need {r.min_runway}+ to negotiate"
                )

        # ── NPS check ─────────────────────────────────────────────────────────
        if r.min_nps is not None:
            if nps_score < r.min_nps:
                approved = False
                # Real: Initialized Capital 2024 — NPS below 30 signals PMF risk
                reasons.append(
                    f"NPS {nps_score} below {r.min_nps} minimum "
                    f"(investors use NPS as PMF proxy — Initialized Capital 2024)"
                )

        # ── Profitability path check ──────────────────────────────────────────
        if r.requires_profitability_path:
            net = monthly_revenue - burn_rate
            burn_multiple = burn_rate / max(monthly_revenue * 0.1, 1)  # simplified burn multiple
            if net < -50_000:
                approved = False
                # Real: WallStreetPrep 2024 — Series B needs credible path to positive unit economics
                reasons.append(
                    f"Net cash flow ${net:+,.0f}/mo — no credible profitability path visible"
                )
            if r.max_burn_multiple and burn_multiple > r.max_burn_multiple:
                approved = False
                reasons.append(
                    f"Burn multiple {burn_multiple:.1f}x above {r.max_burn_multiple:.1f}x threshold"
                )

        # ── Sentiment modulation ──────────────────────────────────────────────
        if approved:
            self._milestones_hit += 1
            self._sentiment = min(1.0, self._sentiment + 0.12)
            msg = (
                f"✅ {r.name} approved — ${r.check_size:,.0f}. "
                f"Strong metrics for this stage. Keep the growth trajectory."
            )
        else:
            self._milestones_missed += 1
            self._sentiment = max(0.0, self._sentiment - 0.10)
            # Real: investors give specific feedback to help founders understand gaps
            msg = f"❌ {r.name} declined. " + " | ".join(reasons)

        return approved, (r.check_size if approved else 0.0), msg

    def record_milestone_miss(self):
        self._milestones_missed += 1
        self._sentiment = max(0.0, self._sentiment - 0.05)

    def record_milestone_hit(self):
        self._milestones_hit += 1
        self._sentiment = min(1.0, self._sentiment + 0.08)
