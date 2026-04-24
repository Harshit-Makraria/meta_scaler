class RunwayTracker:
    """
    Tracks financial state: revenue, burn rate, cash in bank, runway.
    Updated each step based on market conditions and agent actions.
    """

    def __init__(
        self,
        initial_revenue: float = 45_000,
        initial_burn: float = 120_000,
        initial_runway_months: int = 18,
    ):
        self.monthly_revenue = initial_revenue
        self.burn_rate = initial_burn
        self.cash = initial_burn * initial_runway_months  # total cash on hand
        self._revenue_history: list[float] = [initial_revenue]

    def step(self, revenue_growth_rate: float):
        """Advance one month: grow revenue, spend burn, update cash."""
        self.monthly_revenue *= (1 + revenue_growth_rate)
        net_cash_flow = self.monthly_revenue - self.burn_rate
        self.cash += net_cash_flow
        self._revenue_history.append(self.monthly_revenue)

    def apply_pivot(self, cost_months: int = 3):
        """Pivot costs runway and resets revenue momentum."""
        self.cash -= self.burn_rate * cost_months
        self.monthly_revenue *= 0.60   # revenue resets to 60% — new market, lower initial traction

    def apply_fundraise(self, amount: float):
        self.cash += amount

    def apply_hire(self, monthly_cost: float = 20_000):
        self.burn_rate += monthly_cost

    def apply_cut_costs(self, monthly_savings: float = 30_000):
        self.burn_rate = max(20_000, self.burn_rate - monthly_savings)

    def apply_research(self):
        """Research costs half a month of burn."""
        self.cash -= self.burn_rate * 0.5

    @property
    def runway_remaining(self) -> int:
        """Months of runway at current net burn. Capped at 999 when profitable."""
        if self.cash <= 0:
            return 0
        net_burn = self.burn_rate - self.monthly_revenue
        if net_burn <= 0:
            return 999  # cash-flow positive — effectively infinite runway
        return min(999, int(self.cash / net_burn))

    @property
    def revenue_delta_3m(self) -> float:
        """3-month revenue % change. Returns 0 if not enough history."""
        if len(self._revenue_history) < 4:
            return 0.0
        old = self._revenue_history[-4]
        new = self._revenue_history[-1]
        if old == 0:
            return 0.0
        return (new - old) / old
