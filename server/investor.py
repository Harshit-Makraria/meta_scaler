from dataclasses import dataclass


@dataclass
class FundingRound:
    name: str
    milestone_description: str        # What investor says they want (shown to agent)
    check_size: float                  # USD amount if approved
    # Internal evaluation criteria (hidden from agent):
    min_revenue_growth_3m: float | None = None
    min_runway: int | None = None
    min_nps: int | None = None
    requires_profitability_path: bool = False


FUNDING_ROUNDS = [
    FundingRound(
        name="Seed",
        milestone_description="Show 10% MoM revenue growth",
        check_size=500_000,
        min_revenue_growth_3m=0.10,
    ),
    FundingRound(
        name="Series A",
        milestone_description="3+ months runway and positive NPS",  # silently shifts at step 20
        check_size=2_000_000,
        min_runway=3,
        min_nps=0,
    ),
    FundingRound(
        name="Series B",
        milestone_description="Clear path to profitability",  # silently shifts at step 40
        check_size=5_000_000,
        requires_profitability_path=True,
        min_runway=6,
    ),
]


class InvestorAgent:
    """
    Rule-based VC. Silently changes requirements between rounds — agent must infer this
    from new milestone descriptions and evaluation outcomes.
    """

    def __init__(self):
        self._round_index = 0
        self._sentiment = 0.65      # starts cautiously optimistic
        self._milestones_hit = 0
        self._milestones_missed = 0

    def tick(self, step: int):
        """Silently advance round at key steps."""
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
        monthly_revenue: float,
        revenue_delta_3m: float,
        runway_remaining: int,
        nps_score: int,
        burn_rate: float,
    ) -> tuple[bool, float, str]:
        """
        Returns (approved, amount, investor_message).
        Criteria vary by round — agent doesn't know the exact rules.
        """
        r = self.current_round
        approved = True
        reasons = []

        if r.min_revenue_growth_3m is not None and revenue_delta_3m < r.min_revenue_growth_3m:
            approved = False
            reasons.append(f"Revenue growth {revenue_delta_3m:.0%} below {r.min_revenue_growth_3m:.0%} target")

        if r.min_runway is not None and runway_remaining < r.min_runway:
            approved = False
            reasons.append(f"Only {runway_remaining} months runway, need {r.min_runway}+")

        if r.min_nps is not None and nps_score < r.min_nps:
            approved = False
            reasons.append(f"NPS {nps_score} is negative")

        if r.requires_profitability_path:
            net = monthly_revenue - burn_rate
            if net < -20_000:
                approved = False
                reasons.append("No credible path to profitability visible")

        if approved:
            self._milestones_hit += 1
            self._sentiment = min(1.0, self._sentiment + 0.12)
            msg = f"Approved {r.name} ({r.check_size:,.0f} USD). Well done."
        else:
            self._milestones_missed += 1
            self._sentiment = max(0.0, self._sentiment - 0.10)
            msg = f"Declined. " + " | ".join(reasons)

        amount = r.check_size if approved else 0.0
        return approved, amount, msg

    def record_milestone_miss(self):
        self._milestones_missed += 1
        self._sentiment = max(0.0, self._sentiment - 0.05)

    def record_milestone_hit(self):
        self._milestones_hit += 1
        self._sentiment = min(1.0, self._sentiment + 0.08)
