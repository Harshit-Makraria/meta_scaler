"""
CoFounderEnvironment — the complete Strategist Co-Founder simulator.

Renamed from ThePivotEnvironment (plan.md Step 1).
Wires all 6 subsystem managers and implements cross-manager physics
(the butterfly-effect cascades described in plan.md Section 5).

Cross-manager physics (end-of-step propagation):
  1. TeamManager low morale → ProductManager velocity drops
  2. ProductManager high tech debt → RunwayTracker burn rate increases (infra overhead)
  3. MarketingManager campaign success + high tech debt → churn spike
  4. Competitor TALENT_RAID → TeamManager loses 1 eng, ProductManager velocity drops
  5. Competitor VACUUM_GRAB after PIVOT → revenue stolen from pivot reset period
  6. Founder burnout > 0.7 → advice unreliability increases, may block good actions
  7. Low founder trust → stubbornness blocks optimal moves (trust management required)
"""
from __future__ import annotations
import random
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import ActionType, CoFounderAction, CoFounderObservation
from server.market import MarketSimulator, sample_shock
from server.signals import SignalGenerator
from server.founder import FounderAgent
from server.investor import InvestorAgent
from server.runway import RunwayTracker
from server.reward import RewardCalculator
from server.competitor import CompetitorAgent, CompetitorPlay
from server.product_manager import ProductManager
from server.team_manager import TeamManager
from server.marketing_manager import MarketingManager
import server.wandb_logger as wlog

MAX_STEPS = 60

# ── Defaults (used when no scenario is provided) ──────────────────────────────
_DEF = {
    "revenue": 45_000.0,
    "burn_rate": 120_000.0,
    "runway": 18,
    "nps": 52.0,
    "churn_rate": 0.12,
    "pmf_score": 0.55,
    "base_cac": 1500.0,
    "team_size": 8,
    "eng": 4,
    "sales": 2,
    "support": 2,
}


class CoFounderEnvironment(Environment[CoFounderAction, CoFounderObservation, State]):
    """
    OpenEnv RL environment: Strategist Co-Founder simulator.
    The LLM agent acts as an expert advisor to the simulated founder NPC.
    Manages Product, Team, Marketing, Financial, Market, and Competitor dimensions.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, scenario: dict | None = None, rng_seed: int = 42):
        super().__init__()
        self._seed          = rng_seed
        self._scenario      = scenario
        self._scenario_name = scenario.get("name", "default") if scenario else "default"
        self._episode       = 0
        self._state         = State(episode_id=str(uuid4()), step_count=0)
        self._total_raised  = 0.0

        self._init_subsystems(rng_seed)

    # ─── openenv-core required API ────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> CoFounderObservation:
        effective_seed = seed if seed is not None else (self._seed + self._episode)
        random.seed(effective_seed)

        self._init_subsystems(effective_seed)
        self._total_raised  = 0.0
        self._state         = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._reset_rubric()
        self._internal_snapshot = self._build_snapshot()
        return self._build_obs(done=False, reward=0.0)

    def step(
        self,
        action: CoFounderAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> CoFounderObservation:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        prev_snapshot = dict(self._internal_snapshot)
        self._investor.tick(self._step_num)
        self._last_milestone_hit = False

        # ── Notify competitor of agent's action (1-step lag reaction) ────────
        self._competitor.notify_agent_action(action.action_type.value)

        # ── Route action to appropriate manager ──────────────────────────────
        action_effects = self._route_action(action)

        # ── Competitor move ───────────────────────────────────────────────────
        competitor_play = self._competitor.tick(self._internal_snapshot)
        self._apply_competitor_effects(competitor_play)

        # ── Advance world state (market + financial) ──────────────────────────
        cfg         = self._market.get_config(self._step_num)
        phase_name  = self._market.get_phase(self._step_num).value
        growth_rate = cfg.revenue_growth_rate
        if action.action_type == ActionType.CUT_COSTS:
            growth_rate *= 0.8

        self._runway.step(revenue_growth_rate=growth_rate)
        self._true_churn = min(0.95, self._true_churn + cfg.churn_drift)
        self._true_nps   = max(-100, self._true_nps + cfg.nps_drift)
        competitor_event = random.random() < cfg.competitor_activity

        # ── Macro shock ───────────────────────────────────────────────────────
        self._active_shock = sample_shock(self._step_num, phase_name, self._shock_rng)
        if self._active_shock:
            ev = self._active_shock
            self._runway.monthly_revenue *= ev.get("revenue_multiplier", 1.0)
            self._runway.burn_rate       *= ev.get("burn_multiplier", 1.0)
            self._true_nps = max(-100, self._true_nps + ev.get("nps_delta", 0))
            self._competitor.strength = min(1.0,
                self._competitor.strength + ev.get("competitor_strength_boost", 0.0))

        # ── Update subsystem managers ─────────────────────────────────────────
        self._update_all_managers(phase_name)

        # ── Cross-manager physics ─────────────────────────────────────────────
        self._resolve_cross_manager_effects(phase_name)

        self._signals.tick()
        self._step_num      += 1
        self._state.step_count = self._step_num

        survived_episode  = self._step_num >= MAX_STEPS
        ran_out_of_money  = self._runway.runway_remaining <= 0
        acqui_hired       = action.action_type == ActionType.SELL
        done              = ran_out_of_money or survived_episode or acqui_hired

        self._internal_snapshot = self._build_snapshot(competitor_event)

        # ── Advisor outcome tracking (for founder trust) ──────────────────────
        was_positive = (
            self._internal_snapshot["monthly_revenue"] >= prev_snapshot.get("monthly_revenue", 0)
            and self._internal_snapshot["runway_remaining"] >= prev_snapshot.get("runway_remaining", 0) - 1
        )
        self._founder.record_advisor_outcome(was_positive)

        # ── Compute reward ────────────────────────────────────────────────────
        optimal_pivot_start = self._market.get_optimal_pivot_window()[0]
        board_pressure      = self._step_num >= 40 and self._runway.runway_remaining < 6
        episode_data = {
            "step":                   self._step_num,
            "done":                   done,
            "phase":                  phase_name,
            "pivot_was_necessary":    self._market.check_pivot_necessity(self._step_num, self._pivot_step),
            "optimal_pivot_start":    optimal_pivot_start,
            "investor_milestone_hit": self._last_milestone_hit,
            "founder_decay":          self._founder.decay_level,
            "founder_advice":         prev_snapshot.get("founder_advice", ""),
            "shock_active":           self._active_shock is not None,
            "board_pressure":         board_pressure,
        }
        reward, breakdown = self._reward_calc.compute(
            state=prev_snapshot,
            action_type=action.action_type,
            next_state=self._internal_snapshot,
            episode_data=episode_data,
        )
        self._episode_total_reward += reward
        if episode_data["founder_decay"] > 0.7 and breakdown.get("founder_awareness", 0) > 0:
            self._founder_overrides_correct += 1

        # ── W&B logging ───────────────────────────────────────────────────────
        true_phase = self._market.get_phase(self._step_num).value
        wlog.log_step(
            step=self._step_num,
            episode=self._episode,
            reward=reward,
            reward_breakdown=breakdown,
            obs_snapshot=self._internal_snapshot,
            true_phase=true_phase,
            action_type=action.action_type.value,
        )

        if done:
            self._done = True
            pivot_timing_score = self._compute_pivot_timing_score(optimal_pivot_start)
            wlog.log_episode(
                episode=self._episode,
                total_reward=self._episode_total_reward,
                episode_length=self._step_num,
                survived=survived_episode,
                pivot_step=self._pivot_step,
                final_runway=self._runway.runway_remaining,
                final_revenue=self._runway.monthly_revenue,
                final_churn=self._true_churn,
                pivot_timing_score=pivot_timing_score,
                milestones_hit=self._investor._milestones_hit,
                founder_overrides_correct=self._founder_overrides_correct,
                scenario_name=self._scenario_name,
                extra={
                    "final_pmf":     self._product.pmf_score,
                    "final_morale":  self._team.morale,
                    "final_ltv_cac": self._marketing.ltv_cac_ratio,
                    "final_trust":   self._founder.true_trust,
                    "tech_debt":     self._product.tech_debt_severity,
                    "total_raised":  self._total_raised,
                },
            )
            self._episode += 1

        return self._build_obs(
            done=done,
            reward=float(reward),
            metadata={**breakdown, "true_phase": true_phase},
            competitor_event=competitor_event,
        )

    @property
    def state(self) -> State:
        return State(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
            true_phase=self._market.get_phase(self._step_num).value,
            true_churn=self._true_churn,
            true_nps=self._true_nps,
            founder_decay=self._founder.decay_level,
            pivot_step=self._pivot_step,
            episode=self._episode,
            done=self._done,
        )

    def get_metadata(self):
        from openenv.core.env_server.types import EnvironmentMetadata
        return EnvironmentMetadata(
            name="CoFounderEnvironment",
            description="Strategist Co-Founder simulator — 30+ dimensional multi-objective RL environment.",
            version="2.0.0",
            author="Team Meta Scaler",
        )

    # ─── Action routing ───────────────────────────────────────────────────────

    def _route_action(self, action: CoFounderAction) -> dict:
        """Route action to the correct manager(s). Returns effects dict."""
        at = action.action_type
        effects = {}

        if at == ActionType.EXECUTE:
            pass   # world advances naturally

        elif at == ActionType.PIVOT:
            self._pivot_step = self._step_num
            self._runway.apply_pivot(cost_months=3)
            self._true_churn = max(0.05, self._true_churn - 0.08)
            self._true_nps   = max(-30, self._true_nps - 10)
            self._product.handle_pivot()
            self._marketing.handle_pivot()
            effects["pivot_applied"] = True

        elif at == ActionType.RESEARCH:
            self._signals.reduce_noise(factor=0.3, duration=3)
            self._runway.apply_research()
            self._product.handle_research()

        elif at == ActionType.FUNDRAISE:
            approved, amount, msg = self._investor.evaluate_funding_request(
                monthly_revenue=self._runway.monthly_revenue,
                revenue_delta_3m=self._runway.revenue_delta_3m,
                runway_remaining=self._runway.runway_remaining,
                nps_score=int(self._true_nps),
                burn_rate=self._runway.burn_rate,
            )
            if approved:
                self._runway.apply_fundraise(amount)
                self._total_raised += amount
                self._last_milestone_hit = True
                self._investor.record_milestone_hit()
                self._team.handle_fundraise_success()
                effects["funded"] = amount
            else:
                self._investor.record_milestone_miss()
                effects["funded"] = 0

        elif at == ActionType.HIRE:
            result = self._team.handle_hire(role="eng")
            self._runway.apply_hire(monthly_cost=result["burn_increase"])
            effects.update(result)

        elif at == ActionType.CUT_COSTS:
            self._runway.monthly_revenue *= 0.80
            self._runway.apply_cut_costs(monthly_savings=30_000)
            self._team.handle_cut_costs()
            effects["cut_costs"] = True

        elif at == ActionType.SELL:
            pass   # acqui-hire; episode ends

        elif at == ActionType.LAUNCH_FEATURE:
            result = self._product.handle_launch_feature(
                eng_headcount=self._team.eng_headcount,
                team_morale=self._team.morale,
            )
            # PMF improvement may reduce churn slightly
            if result.get("pmf_delta", 0) > 0:
                self._true_churn = max(0.01, self._true_churn - result["pmf_delta"] * 0.3)
            effects.update(result)

        elif at == ActionType.MARKETING_CAMPAIGN:
            result = self._marketing.handle_marketing_campaign(
                monthly_burn=self._runway.burn_rate,
                pmf_score=self._product.pmf_score,
            )
            # Campaign costs come out of runway
            self._runway.cash -= result.get("spend", 0)
            effects.update(result)

        elif at == ActionType.SET_PRICING:
            result = self._marketing.handle_set_pricing(increase=True)
            # Price change immediately affects revenue
            self._runway.monthly_revenue *= (1 + result.get("arpu_delta", 0))
            effects.update(result)

        elif at == ActionType.FIRE:
            result = self._team.handle_fire(role="eng")
            if "burn_decrease" in result:
                self._runway.apply_cut_costs(monthly_savings=result["burn_decrease"])
            effects.update(result)

        elif at == ActionType.PARTNERSHIP:
            result = self._marketing.handle_partnership()
            effects.update(result)

        return effects

    # ─── Competitor effect application ────────────────────────────────────────

    def _apply_competitor_effects(self, play: CompetitorPlay) -> None:
        """Apply competitor play effects to all managers."""
        if self._competitor.market_share_impact > 0:
            self._runway.monthly_revenue *= (1.0 - self._competitor.market_share_impact)
        if self._competitor.burn_impact > 0:
            self._runway.burn_rate *= (1.0 + self._competitor.burn_impact)

        if play == CompetitorPlay.TALENT_RAID:
            raid_result = self._team.handle_competitor_talent_raid()
            if raid_result.get("poached"):
                self._product.handle_competitor_talent_raid()

        elif play == CompetitorPlay.AGGRESSIVE_MKT:
            self._marketing.handle_competitor_aggressive_mkt()

    # ─── Manager updates ──────────────────────────────────────────────────────

    def _update_all_managers(self, phase_name: str) -> None:
        """Tick all subsystem managers with current world state."""
        self._team.tick(
            runway_remaining=self._runway.runway_remaining,
            monthly_revenue=self._runway.monthly_revenue,
            burn_rate=self._runway.burn_rate,
            phase_name=phase_name,
        )
        self._product.tick(
            nps_score=int(self._true_nps),
            churn_rate=self._true_churn,
            complaint_types=[],   # complaints resolved in _build_obs
            team_morale=self._team.morale,
            eng_headcount=self._team.eng_headcount,
            phase_name=phase_name,
        )
        self._marketing.tick(
            churn_rate=self._true_churn,
            monthly_revenue=self._runway.monthly_revenue,
            pmf_score=self._product.pmf_score,
            competitor_play=self._competitor.current_play.value,
            phase_name=phase_name,
            sales_headcount=self._team.sales_headcount,
        )
        self._founder.tick(
            runway_remaining=self._runway.runway_remaining,
            team_morale=self._team.morale,
            step=self._step_num,
        )

    # ─── Cross-manager physics ────────────────────────────────────────────────

    def _resolve_cross_manager_effects(self, phase_name: str) -> None:
        """
        The butterfly-effect cascades from plan.md Section 5.
        These run AFTER all managers have ticked individually.
        """

        # 1. Low morale → higher burn (infra unreliability, context switching)
        if self._team.morale < 0.40:
            self._runway.burn_rate *= 1.02   # 2% burn overhead from low morale

        # 2. Critical tech debt → burn increase (server incidents, constant firefighting)
        if self._product.tech_debt_severity == "critical":
            self._runway.burn_rate *= 1.05   # 5% extra infra/on-call overhead
        elif self._product.tech_debt_severity == "high":
            self._runway.burn_rate *= 1.02

        # 3. Campaign success + high tech debt → churn spike
        #    (Marketing brings users in; broken product chases them out)
        if (self._product.tech_debt_ratio > 0.50 and
                self._marketing.pipeline_generated > self._runway.monthly_revenue * 0.3):
            self._true_churn = min(0.95, self._true_churn + 0.02)

        # 4. Decline + low PMF → accelerated NPS decay
        if phase_name == "DECLINE" and self._product.pmf_score < 0.35:
            self._true_nps = max(-100, self._true_nps - 2.0)

        # 5. Growing PMF → churn dampening
        if self._product.pmf_score > 0.70:
            self._true_churn = max(0.01, self._true_churn - 0.005)

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _init_subsystems(self, seed: int) -> None:
        """Initialize or reset all subsystems from scenario config."""
        ini  = self._scenario["initial_state"] if self._scenario else {}
        ini2 = self._scenario.get("initial_state", {}) if self._scenario else {}

        self._market    = MarketSimulator(scenario=self._scenario)
        self._signals   = SignalGenerator(rng_seed=seed)
        self._founder   = FounderAgent(rng_seed=seed + 1)
        self._investor  = InvestorAgent()
        self._runway    = RunwayTracker(
            initial_revenue=ini.get("revenue", _DEF["revenue"]),
            initial_burn=ini.get("burn_rate", _DEF["burn_rate"]),
            initial_runway_months=ini.get("runway", _DEF["runway"]),
        )
        self._reward_calc = RewardCalculator()
        self._competitor  = CompetitorAgent(rng_seed=seed + 99)

        # ── New subsystem managers ────────────────────────────────────────────
        team_size  = ini.get("team_size", _DEF["team_size"])
        eng        = ini.get("eng_headcount",     max(1, team_size // 2))
        sales      = ini.get("sales_headcount",   max(1, team_size // 4))
        support    = ini.get("support_headcount",  max(1, team_size - eng - sales))

        self._product   = ProductManager(
            initial_pmf_score=ini.get("pmf_score",   _DEF["pmf_score"]),
            initial_tech_debt=ini.get("tech_debt",   0.15),
            initial_pipeline=ini.get("pipeline",     3),
            rng_seed=seed + 10,
        )
        self._team      = TeamManager(
            initial_eng=eng,
            initial_sales=sales,
            initial_support=support,
            initial_morale=ini.get("team_morale",   0.75),
            rng_seed=seed + 20,
        )
        self._marketing = MarketingManager(
            initial_cac=ini.get("base_cac",     _DEF["base_cac"]),
            initial_arpu=ini.get("arpu",         ini.get("revenue", _DEF["revenue"]) / max(ini.get("customer_count", 30), 1)),
            initial_churn=ini.get("churn_rate",  _DEF["churn_rate"]),
            initial_brand=ini.get("brand_awareness", 0.20),
            rng_seed=seed + 30,
        )

        # ── Episode state ─────────────────────────────────────────────────────
        self._step_num           = 0
        self._done               = False
        self._true_nps           = float(ini.get("nps", _DEF["nps"]))
        self._true_churn         = float(ini.get("churn_rate", _DEF["churn_rate"]))
        self._pivot_step: int | None = None
        self._last_milestone_hit = False
        self._episode_total_reward   = 0.0
        self._founder_overrides_correct = 0
        self._churn_history:     list[float] = []
        self._complaint_history: list[set]   = []
        self._months_at_risk     = 0
        self._active_shock: dict | None = None
        self._shock_rng          = random.Random(seed + 77)
        self._internal_snapshot: dict = {}

    def _compute_pivot_timing_score(self, optimal_pivot_start: int) -> float:
        if self._pivot_step is None:
            return 0.0
        steps_late = self._pivot_step - optimal_pivot_start
        if steps_late < 0:
            return 0.0
        return max(0.0, 1.0 - steps_late * 0.09)

    def _build_snapshot(self, competitor_event: bool = False) -> dict:
        """Builds the internal dict used by reward + wlog."""
        phase_name = self._market.get_phase(self._step_num).value
        return {
            "step":               self._step_num,
            "monthly_revenue":    self._runway.monthly_revenue,
            "burn_rate":          self._runway.burn_rate,
            "runway_remaining":   self._runway.runway_remaining,
            "revenue_delta_3m":   self._runway.revenue_delta_3m,
            "churn_rate":         self._true_churn,
            "nps_score":          int(self._true_nps),
            "competitor_event":   competitor_event,
            "competitor_play":    self._competitor.current_play.value,
            "competitor_strength":round(self._competitor.strength, 2),
            "founder_advice":     self._founder.get_advice(
                                    self._market.get_phase(self._step_num),
                                    self._step_num,
                                    self._runway.runway_remaining,
                                  ),
            "founder_confidence": self._founder.get_confidence(),
            "founder_trust":      self._founder.get_trust(),
            "founder_burnout":    self._founder.get_burnout_signal(),
            "founder_stubbornness": self._founder.get_stubbornness(),
            "investor_sentiment": self._investor.sentiment,
            # Product
            "pmf_score":          self._product.pmf_score,
            "tech_debt_ratio":    self._product.tech_debt_ratio,
            "tech_debt_severity": self._product.tech_debt_severity,
            "features_shipped_last": self._product.features_shipped_last,
            "feature_pipeline_depth": self._product.feature_pipeline_depth,
            # Team
            "team_size":          self._team.team_size,
            "eng_headcount":      self._team.eng_headcount,
            "sales_headcount":    self._team.sales_headcount,
            "support_headcount":  self._team.support_headcount,
            "team_morale":        self._team.morale,
            # Marketing
            "cac":                self._marketing.cac,
            "ltv":                self._marketing.ltv,
            "ltv_cac_ratio":      self._marketing.ltv_cac_ratio,
            "brand_awareness":    self._marketing.brand_awareness,
            "pipeline_generated": self._marketing.pipeline_generated,
        }

    def _compute_trends(self, noisy_churn: float, complaints: list[str]) -> dict:
        self._churn_history.append(noisy_churn)
        self._complaint_history.append(set(complaints))
        if len(self._churn_history) > 3:
            self._churn_history.pop(0)
        if len(self._complaint_history) > 3:
            self._complaint_history.pop(0)

        delta = self._runway.revenue_delta_3m
        if delta > 0.05:
            revenue_trend = "growing"
        elif delta > -0.03:
            revenue_trend = "plateauing"
        else:
            revenue_trend = "declining"

        if len(self._churn_history) >= 2:
            churn_rise = self._churn_history[-1] - self._churn_history[0]
            churn_trend = "spiking" if churn_rise > 0.10 else ("rising" if churn_rise > 0.03 else "stable")
        else:
            churn_trend = "stable"

        complaint_shift = False
        if len(self._complaint_history) >= 2:
            prev    = self._complaint_history[-2]
            curr    = self._complaint_history[-1]
            overlap = len(prev & curr) / max(len(prev | curr), 1)
            complaint_shift = overlap < 0.4

        if self._runway.runway_remaining < 6:
            self._months_at_risk += 1
        else:
            self._months_at_risk = 0

        return {
            "revenue_trend":          revenue_trend,
            "churn_trend":            churn_trend,
            "complaint_shift_detected": complaint_shift,
            "months_at_risk":         self._months_at_risk,
        }

    def _build_obs(
        self,
        done: bool,
        reward: float,
        metadata: dict | None = None,
        competitor_event: bool = False,
    ) -> CoFounderObservation:
        cfg  = self._market.get_config(self._step_num)
        sig  = self._signals.generate_observation(
            true_revenue=self._runway.monthly_revenue,
            true_churn=self._true_churn,
            true_nps=self._true_nps,
            true_competitor_event=competitor_event,
            phase_config=cfg,
            step=self._step_num,
        )
        trends = self._compute_trends(sig["noisy_churn"], sig["user_complaints"])

        board_pressure = self._step_num >= 40 and self._runway.runway_remaining < 6
        board_demands: list[str] = []
        if board_pressure:
            board_demands = ["PIVOT or CUT_COSTS", "Present recovery plan", "Extend runway 6+ months"]
        elif self._step_num >= 40:
            board_demands = ["Show path to profitability"]

        shock_name = self._active_shock["name"]    if self._active_shock else ""
        shock_msg  = self._active_shock.get("message", "") if self._active_shock else ""

        snap = self._internal_snapshot
        net_flow = self._runway.monthly_revenue - self._runway.burn_rate

        return CoFounderObservation(
            done=done,
            reward=reward,
            metadata=metadata or {},

            # Financial
            runway_remaining=self._runway.runway_remaining,
            monthly_revenue=sig["noisy_revenue"],
            burn_rate=self._runway.burn_rate,
            revenue_delta_3m=self._runway.revenue_delta_3m,
            net_cash_flow=round(net_flow, 2),
            total_raised_usd=self._total_raised,

            # Market signals
            churn_rate=sig["noisy_churn"],
            nps_score=sig["noisy_nps"],
            user_complaints=sig["user_complaints"],
            competitor_launched=sig["competitor_launched"],
            competitor_play=self._competitor.current_play.value,
            competitor_strength=self._competitor.strength,
            revenue_trend=trends["revenue_trend"],
            churn_trend=trends["churn_trend"],

            # Product
            pmf_score=snap.get("pmf_score", 0.55),
            tech_debt_ratio=snap.get("tech_debt_ratio", 0.15),
            tech_debt_severity=snap.get("tech_debt_severity", "low"),
            features_shipped_last=snap.get("features_shipped_last", 1),
            feature_pipeline_depth=snap.get("feature_pipeline_depth", 3),

            # Team
            team_size=snap.get("team_size", 8),
            eng_headcount=snap.get("eng_headcount", 4),
            sales_headcount=snap.get("sales_headcount", 2),
            support_headcount=snap.get("support_headcount", 2),
            team_morale=snap.get("team_morale", 0.75),

            # Marketing
            cac=snap.get("cac", 1500.0),
            ltv=snap.get("ltv", 6000.0),
            ltv_cac_ratio=snap.get("ltv_cac_ratio", 4.0),
            brand_awareness=snap.get("brand_awareness", 0.20),
            pipeline_generated=snap.get("pipeline_generated", 0.0),

            # Founder
            founder_advice=snap.get("founder_advice", "stay_course"),
            founder_confidence=snap.get("founder_confidence", 1.0),
            founder_trust=snap.get("founder_trust", 0.80),
            founder_burnout=snap.get("founder_burnout", 0.10),
            founder_stubbornness=snap.get("founder_stubbornness", 0.30),

            # Investor / Board
            investor_sentiment=self._investor.sentiment,
            next_milestone=self._investor.get_current_milestone(),
            board_pressure=board_pressure,
            board_demands=board_demands,

            # Trends / shocks
            complaint_shift_detected=trends["complaint_shift_detected"],
            months_at_risk=trends["months_at_risk"],
            active_shock=shock_name,
            shock_message=shock_msg,

            # Pivot economics
            pivot_cost_estimate=3,
            pivot_cost_months=3.0,

            # Episode metadata
            step=self._step_num,
            max_steps=MAX_STEPS,
            scenario=self._scenario_name,
        )


# ── Backward-compatibility alias ──────────────────────────────────────────────
ThePivotEnvironment = CoFounderEnvironment
