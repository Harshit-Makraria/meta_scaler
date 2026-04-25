"""
MarketingManager — tracks growth metrics for the CoFounder environment.

Manages:
  - CAC (Customer Acquisition Cost): how much you spend to win one customer
  - LTV (Lifetime Value): expected revenue per customer over their lifetime
  - LTV:CAC ratio: the single most important unit economics metric for SaaS
  - Brand Awareness: cumulative reach/recognition (0-1)
  - Sales Pipeline: monthly qualified revenue potential created

Real-world data sources (post-ZIRP, 2024):
  - HockeyStack 2024: B2B SaaS median CAC $1,450 seed stage
  - Benchmarkit 2024: LTV:CAC > 3.0 = healthy; < 1.5 = dying
  - OpenView 2024: CAC payback period now 18-24 months (up from 8-12 in ZIRP era)
  - Gainsight 2024: NRR > 100% = expansion revenue (users buy more over time)
  - Crunchbase: 38% of marketplace failures due to unit economics never improving

Cross-manager interactions (resolved in cofounder_environment.py):
  - PMF < 0.4 → CAC rises (hard to acquire users who don't love product)
  - Competitor AGGRESSIVE_MKT → CAC spikes (bidding war on paid channels)
  - High churn_rate → LTV drops (users leave faster than expected)
  - PARTNERSHIP → CAC drops by 15-25% (channel sales is cheaper than direct)
"""
from __future__ import annotations
import random


class MarketingManager:
    """
    Tracks unit economics: CAC, LTV, brand awareness, and sales pipeline.
    Called each step by CoFounderEnvironment.
    """

    def __init__(
        self,
        initial_cac:    float = 1500.0,
        initial_arpu:   float = 500.0,   # average revenue per user per month
        initial_churn:  float = 0.10,
        initial_brand:  float = 0.20,
        rng_seed: int = 42,
    ):
        self._cac            = float(initial_cac)
        self._arpu           = float(initial_arpu)
        self._base_churn     = float(initial_churn)
        self._brand          = float(initial_brand)
        self._pipeline       = 0.0          # USD value of qualified pipeline
        self._campaign_boost = 0            # steps remaining for active campaign boost
        self._rng            = random.Random(rng_seed)
        self._total_campaign_spend = 0.0

    # ── Main step update ──────────────────────────────────────────────────────

    def tick(
        self,
        churn_rate: float,
        monthly_revenue: float,
        pmf_score: float,
        competitor_play: str,
        phase_name: str,
        sales_headcount: int,
    ) -> None:
        """
        Update CAC, LTV, brand, and pipeline each month.
        Called BEFORE action processing.
        """
        self._base_churn = churn_rate
        self._update_cac(pmf_score, competitor_play, phase_name)
        self._update_ltv()
        self._update_brand(phase_name)
        self._update_pipeline(monthly_revenue, sales_headcount)

        if self._campaign_boost > 0:
            self._campaign_boost -= 1

    # ── Action handlers ───────────────────────────────────────────────────────

    def handle_marketing_campaign(
        self,
        monthly_burn: float,
        pmf_score: float,
    ) -> dict:
        """
        MARKETING_CAMPAIGN action.
        Spend 15% of monthly burn on a campaign burst:
          - High PMF (>0.6): campaign is effective, CAC drops, pipeline surges
          - Low PMF (<0.4): campaign brings in wrong users, CAC stays high, churn spikes
        Returns effects dict.
        """
        campaign_spend = monthly_burn * 0.15
        self._total_campaign_spend += campaign_spend
        self._campaign_boost = 2   # 2-month effect window

        if pmf_score > 0.60:
            # Good product + marketing = compounding growth
            cac_reduction = self._rng.uniform(0.10, 0.20)
            pipeline_boost = campaign_spend * self._rng.uniform(3.0, 5.0)
            brand_boost    = 0.04 + self._rng.uniform(0, 0.02)
            self._cac    = max(300, self._cac * (1.0 - cac_reduction))
            self._pipeline += pipeline_boost
            self._brand   = min(1.0, self._brand + brand_boost)
            return {
                "spend": campaign_spend,
                "cac_delta": -cac_reduction,
                "pipeline_boost": pipeline_boost,
                "brand_boost": brand_boost,
                "message": f"Campaign effective (PMF {pmf_score:.0%}). CAC down {cac_reduction:.0%}, pipeline +${pipeline_boost:,.0f}.",
            }
        else:
            # Weak PMF: expensive leads who churn
            pipeline_boost = campaign_spend * self._rng.uniform(0.5, 1.5)
            brand_boost    = 0.01
            self._pipeline += pipeline_boost
            self._brand    = min(1.0, self._brand + brand_boost)
            return {
                "spend": campaign_spend,
                "cac_delta": 0.0,
                "pipeline_boost": pipeline_boost,
                "brand_boost": brand_boost,
                "message": f"Campaign weak (PMF {pmf_score:.0%}). Leads acquired but churn likely. Fix product first.",
            }

    def handle_partnership(self) -> dict:
        """
        PARTNERSHIP action: form a channel partnership.
        Reduces CAC by 15-25% (channel sales cheaper than direct outbound).
        Boosts pipeline by 2x sales headcount × avg deal size.
        Takes 1-2 months to show effect.
        """
        cac_reduction  = self._rng.uniform(0.15, 0.25)
        pipeline_boost = self._arpu * 12 * self._rng.uniform(2, 5)   # 2-5 deal referrals

        self._cac      = max(300, self._cac * (1.0 - cac_reduction))
        self._pipeline += pipeline_boost
        self._brand    = min(1.0, self._brand + 0.03)

        return {
            "cac_reduction": cac_reduction,
            "pipeline_boost": pipeline_boost,
            "message": f"Partnership signed. CAC down {cac_reduction:.0%}. Pipeline +${pipeline_boost:,.0f}.",
        }

    def handle_set_pricing(self, increase: bool = True) -> dict:
        """
        SET_PRICING action.
        - Price increase: ARPU up 15-25%, but churn risk +3-5% next month
        - Price decrease: ARPU down 10-15%, churn risk drops, volume may rise
        """
        if increase:
            arpu_change = self._rng.uniform(0.15, 0.25)
            self._arpu  *= (1.0 + arpu_change)
            return {
                "arpu_delta": arpu_change,
                "churn_risk_delta": +0.03,
                "message": f"Pricing raised {arpu_change:.0%}. Watch for churn from price-sensitive customers.",
            }
        else:
            arpu_change = -self._rng.uniform(0.10, 0.15)
            self._arpu  *= (1.0 + arpu_change)
            return {
                "arpu_delta": arpu_change,
                "churn_risk_delta": -0.02,
                "message": f"Pricing reduced {abs(arpu_change):.0%}. Churn risk eases. Volume should increase.",
            }

    def handle_competitor_aggressive_mkt(self) -> None:
        """Competitor flooding paid channels raises your CAC."""
        self._cac    = min(self._cac * 1.15, self._cac + 500)
        self._brand  = max(0.0, self._brand - 0.02)

    def handle_pivot(self) -> None:
        """Pivot resets brand and pipeline (new market = start over)."""
        self._pipeline = 0.0
        self._brand    = max(0.05, self._brand * 0.30)
        self._cac      = self._cac * 1.30   # new market: higher CAC initially

    # ── Private helpers ───────────────────────────────────────────────────────

    def _update_cac(self, pmf_score: float, competitor_play: str, phase_name: str) -> None:
        """
        CAC trends:
          - PMF improving → CAC drops (word-of-mouth, better conversion)
          - DECLINE → CAC rises (market saturated)
          - Competitor PRICE_WAR or AGGRESSIVE_MKT → CAC spikes
        """
        # Phase trend
        phase_adj = {"GROWTH": -0.01, "SATURATION": +0.02, "DECLINE": +0.04}.get(phase_name, 0)
        pmf_adj   = -0.02 if pmf_score > 0.65 else (+0.01 if pmf_score < 0.40 else 0)

        # Competitor pressure
        comp_adj = 0.0
        if competitor_play in ("price_war", "aggressive_mkt"):
            comp_adj = +0.03

        total_adj = phase_adj + pmf_adj + comp_adj
        self._cac = max(200, self._cac * (1 + total_adj))

    def _update_ltv(self) -> None:
        """
        LTV = ARPU / churn_rate  (standard SaaS formula).
        ARPU slowly increases with NRR (net revenue retention from upgrades).
        """
        if self._base_churn > 0:
            ltv = self._arpu / self._base_churn
        else:
            ltv = self._arpu * 24   # cap at 24-month LTV when churn → 0
        self._ltv = min(ltv, self._arpu * 36)   # sanity cap at 3yr LTV

    def _update_brand(self, phase_name: str) -> None:
        """Brand decays slightly each month unless maintained."""
        decay = {"GROWTH": 0.002, "SATURATION": 0.003, "DECLINE": 0.005}.get(phase_name, 0.003)
        self._brand = max(0.01, self._brand - decay)

    def _update_pipeline(self, monthly_revenue: float, sales_headcount: int) -> None:
        """
        Pipeline generated by sales team each month.
        sales_headcount × avg_deal_size × close_rate
        """
        avg_deal       = self._arpu * 8         # 8 months ARPU per deal (typical SaaS ACV)
        close_rate     = 0.15 + self._brand * 0.10  # 15-25% close rate depending on brand
        pipeline_gen   = sales_headcount * avg_deal * close_rate

        # Pipeline decays 30% each month (deals go cold, slow cycles)
        self._pipeline = self._pipeline * 0.70 + pipeline_gen

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def cac(self) -> float:
        return round(self._cac, 2)

    @property
    def ltv(self) -> float:
        if not hasattr(self, "_ltv"):
            self._update_ltv()
        return round(self._ltv, 2)

    @property
    def ltv_cac_ratio(self) -> float:
        if self._cac <= 0:
            return 0.0
        return round(self.ltv / self._cac, 2)

    @property
    def brand_awareness(self) -> float:
        return round(self._brand, 3)

    @property
    def pipeline_generated(self) -> float:
        return round(self._pipeline, 2)

    @property
    def arpu(self) -> float:
        return round(self._arpu, 2)

    @property
    def total_campaign_spend(self) -> float:
        return self._total_campaign_spend

    def snapshot(self) -> dict:
        return {
            "cac":               self.cac,
            "ltv":               self.ltv,
            "ltv_cac_ratio":     self.ltv_cac_ratio,
            "brand_awareness":   self.brand_awareness,
            "pipeline_generated":self.pipeline_generated,
        }
