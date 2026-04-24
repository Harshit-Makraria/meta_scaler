"""
Main ThePivotEnvironment — inherits from openenv-core Environment base class.
Connects all subsystems: market, signals, founder, investor, runway, reward.
step() returns PivotObservation with .reward and .done embedded (openenv-core pattern).
"""
import random
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import ActionType, PivotAction, PivotObservation
from server.market import MarketSimulator
from server.signals import SignalGenerator
from server.founder import FounderAgent
from server.investor import InvestorAgent
from server.runway import RunwayTracker
from server.reward import RewardCalculator
from server.competitor import CompetitorAgent, CompetitorPlay
import server.wandb_logger as wlog

MAX_STEPS = 60
INITIAL_REVENUE = 45_000.0
INITIAL_BURN = 120_000.0
INITIAL_RUNWAY = 18
INITIAL_NPS = 52.0
INITIAL_CHURN = 0.12


class ThePivotEnvironment(Environment[PivotAction, PivotObservation, State]):
    """
    OpenEnv RL environment: startup founder pivot simulation.
    The agent must detect hidden market phase shifts and decide when to pivot.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, scenario: dict | None = None, rng_seed: int = 42):
        super().__init__()
        self._seed = rng_seed
        self._scenario = scenario
        self._scenario_name = scenario.get("name", "default") if scenario else "default"
        self._episode = 0
        self._state = State(episode_id=str(uuid4()), step_count=0)

        self._market = MarketSimulator(scenario=scenario)
        self._signals = SignalGenerator(rng_seed=rng_seed)
        self._founder = FounderAgent(rng_seed=rng_seed + 1)
        self._investor = InvestorAgent()
        self._runway = RunwayTracker(
            initial_revenue=scenario["initial_state"]["revenue"] if scenario else INITIAL_REVENUE,
            initial_burn=scenario["initial_state"].get("burn_rate", INITIAL_BURN) if scenario else INITIAL_BURN,
            initial_runway_months=scenario["initial_state"].get("runway", INITIAL_RUNWAY) if scenario else INITIAL_RUNWAY,
        )
        self._reward_calc = RewardCalculator()
        self._competitor = CompetitorAgent(rng_seed=rng_seed + 99)

        self._step_num = 0
        self._done = False
        self._true_nps = scenario["initial_state"]["nps"] if scenario else INITIAL_NPS
        self._true_churn = scenario["initial_state"]["churn_rate"] if scenario else INITIAL_CHURN
        self._pivot_step: int | None = None
        self._last_milestone_hit = False
        self._episode_total_reward = 0.0
        self._founder_overrides_correct = 0
        self._internal_snapshot: dict = {}
        # Trend tracking (rolling history)
        self._churn_history: list[float] = []
        self._complaint_history: list[set] = []
        self._months_at_risk: int = 0

    # ─── openenv-core required API ────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> PivotObservation:
        """Start a new episode. Returns initial PivotObservation."""
        effective_seed = seed if seed is not None else (self._seed + self._episode)
        random.seed(effective_seed)

        ini = self._scenario["initial_state"] if self._scenario else {}
        self._market = MarketSimulator(scenario=self._scenario)
        self._signals = SignalGenerator(rng_seed=effective_seed)
        self._founder = FounderAgent(rng_seed=effective_seed + 1)
        self._investor = InvestorAgent()
        self._runway = RunwayTracker(
            initial_revenue=ini.get("revenue", INITIAL_REVENUE),
            initial_burn=ini.get("burn_rate", INITIAL_BURN),
            initial_runway_months=ini.get("runway", INITIAL_RUNWAY),
        )
        self._competitor.reset()
        self._step_num = 0
        self._done = False
        self._true_nps = ini.get("nps", INITIAL_NPS)
        self._true_churn = ini.get("churn_rate", INITIAL_CHURN)
        self._pivot_step = None
        self._last_milestone_hit = False
        self._episode_total_reward = 0.0
        self._founder_overrides_correct = 0
        self._churn_history = []
        self._complaint_history = []
        self._months_at_risk = 0

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._reset_rubric()
        self._internal_snapshot = self._build_snapshot()
        return self._build_obs(done=False, reward=0.0)

    def step(
        self,
        action: PivotAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> PivotObservation:
        """
        Process one action (= 1 month).
        Returns PivotObservation with .reward and .done set (openenv-core pattern).
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        prev_snapshot = dict(self._internal_snapshot)
        self._investor.tick(self._step_num)
        self._last_milestone_hit = False

        # ── Apply action ──────────────────────────────────────────────
        if action.action_type == ActionType.EXECUTE:
            pass

        elif action.action_type == ActionType.PIVOT:
            self._pivot_step = self._step_num
            self._runway.apply_pivot(cost_months=3)
            self._true_churn = max(0.05, self._true_churn - 0.08)
            self._true_nps = max(-30, self._true_nps - 10)

        elif action.action_type == ActionType.RESEARCH:
            self._signals.reduce_noise(factor=0.3, duration=3)
            self._runway.apply_research()

        elif action.action_type == ActionType.FUNDRAISE:
            approved, amount, msg = self._investor.evaluate_funding_request(
                monthly_revenue=self._runway.monthly_revenue,
                revenue_delta_3m=self._runway.revenue_delta_3m,
                runway_remaining=self._runway.runway_remaining,
                nps_score=int(self._true_nps),
                burn_rate=self._runway.burn_rate,
            )
            if approved:
                self._runway.apply_fundraise(amount)
                self._last_milestone_hit = True
                self._investor.record_milestone_hit()
            else:
                self._investor.record_milestone_miss()

        elif action.action_type == ActionType.HIRE:
            self._runway.apply_hire(monthly_cost=20_000)

        elif action.action_type == ActionType.CUT_COSTS:
            self._runway.monthly_revenue *= 0.80
            self._runway.apply_cut_costs(monthly_savings=30_000)

        # ── Competitor move ───────────────────────────────────────────
        competitor_play = self._competitor.tick(self._internal_snapshot)
        if self._competitor.market_share_impact > 0:
            self._runway.monthly_revenue *= (1.0 - self._competitor.market_share_impact)
        if self._competitor.burn_impact > 0:
            self._runway.burn_rate *= (1.0 + self._competitor.burn_impact)

        # ── Advance world state ───────────────────────────────────────
        cfg = self._market.get_config(self._step_num)
        growth_rate = cfg.revenue_growth_rate
        if action.action_type == ActionType.CUT_COSTS:
            growth_rate *= 0.8
        self._runway.step(revenue_growth_rate=growth_rate)

        self._true_churn = min(0.95, self._true_churn + cfg.churn_drift)
        self._true_nps = max(-100, self._true_nps + cfg.nps_drift)
        competitor_event = random.random() < cfg.competitor_activity

        self._signals.tick()
        self._step_num += 1
        self._state.step_count = self._step_num

        survived_episode = self._step_num >= MAX_STEPS
        ran_out_of_money = self._runway.runway_remaining <= 0
        done = ran_out_of_money or survived_episode

        self._internal_snapshot = self._build_snapshot(competitor_event)

        # ── Compute reward ────────────────────────────────────────────
        optimal_pivot_start = self._market.get_optimal_pivot_window()[0]
        episode_data = {
            "step": self._step_num,
            "done": done,
            "pivot_was_necessary": self._market.check_pivot_necessity(self._step_num, self._pivot_step),
            "optimal_pivot_start": optimal_pivot_start,
            "investor_milestone_hit": self._last_milestone_hit,
            "founder_decay": self._founder.decay_level,
            "founder_advice": prev_snapshot.get("founder_advice", ""),
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

        # ── W&B logging ───────────────────────────────────────────────
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
            )
            self._episode += 1

        # reward and done go INTO the observation (openenv-core pattern)
        return self._build_obs(
            done=done,
            reward=float(reward),
            metadata={**breakdown, "true_phase": true_phase},
            competitor_event=competitor_event,
        )

    @property
    def state(self) -> State:
        """Internal state — for debugging, not sent to agent during training."""
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
            name="ThePivotEnvironment",
            description="Startup founder pivot simulation — detect hidden market shifts and pivot before runway hits zero.",
            version="0.1.0",
            author="Team Meta Scaler",
        )

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _compute_pivot_timing_score(self, optimal_pivot_start: int) -> float:
        if self._pivot_step is None:
            return 0.0
        steps_late = self._pivot_step - optimal_pivot_start
        if steps_late < 0:
            return 0.0
        return max(0.0, 1.0 - steps_late * 0.09)

    def _build_snapshot(self, competitor_event: bool = False) -> dict:
        return {
            "step": self._step_num,
            "monthly_revenue": self._runway.monthly_revenue,
            "burn_rate": self._runway.burn_rate,
            "runway_remaining": self._runway.runway_remaining,
            "revenue_delta_3m": self._runway.revenue_delta_3m,
            "churn_rate": self._true_churn,
            "nps_score": int(self._true_nps),
            "competitor_event": competitor_event,
            "competitor_play": self._competitor.current_play.value,
            "competitor_strength": round(self._competitor.strength, 2),
            "founder_advice": self._founder.get_advice(
                self._market.get_phase(self._step_num), self._step_num, self._runway.runway_remaining
            ),
            "founder_confidence": self._founder.get_confidence(),
            "investor_sentiment": self._investor.sentiment,
        }

    def _compute_trends(self, noisy_churn: float, complaints: list[str]) -> dict:
        """Compute rolling trend signals over the last 3 steps."""
        self._churn_history.append(noisy_churn)
        self._complaint_history.append(set(complaints))
        if len(self._churn_history) > 3:
            self._churn_history.pop(0)
        if len(self._complaint_history) > 3:
            self._complaint_history.pop(0)

        # Revenue trend from 3-month delta
        delta = self._runway.revenue_delta_3m
        if delta > 0.05:
            revenue_trend = "growing"
        elif delta > -0.03:
            revenue_trend = "plateauing"
        else:
            revenue_trend = "declining"

        # Churn trend from rolling window
        if len(self._churn_history) >= 2:
            churn_rise = self._churn_history[-1] - self._churn_history[0]
            if churn_rise > 0.10:
                churn_trend = "spiking"
            elif churn_rise > 0.03:
                churn_trend = "rising"
            else:
                churn_trend = "stable"
        else:
            churn_trend = "stable"

        # Complaint shift: did the complaint set change character?
        complaint_shift = False
        if len(self._complaint_history) >= 2:
            prev = self._complaint_history[-2]
            curr = self._complaint_history[-1]
            overlap = len(prev & curr) / max(len(prev | curr), 1)
            complaint_shift = overlap < 0.4   # less than 40% overlap = shift

        # Months at risk
        if self._runway.runway_remaining < 6:
            self._months_at_risk += 1
        else:
            self._months_at_risk = 0

        return {
            "revenue_trend": revenue_trend,
            "churn_trend": churn_trend,
            "complaint_shift_detected": complaint_shift,
            "months_at_risk": self._months_at_risk,
        }

    def _build_obs(
        self,
        done: bool,
        reward: float,
        metadata: dict | None = None,
        competitor_event: bool = False,
    ) -> PivotObservation:
        cfg = self._market.get_config(self._step_num)
        sig = self._signals.generate_observation(
            true_revenue=self._runway.monthly_revenue,
            true_churn=self._true_churn,
            true_nps=self._true_nps,
            true_competitor_event=competitor_event,
            phase_config=cfg,
            step=self._step_num,
        )
        trends = self._compute_trends(sig["noisy_churn"], sig["user_complaints"])

        return PivotObservation(
            done=done,
            reward=reward,
            metadata=metadata or {},
            runway_remaining=self._runway.runway_remaining,
            monthly_revenue=sig["noisy_revenue"],
            burn_rate=self._runway.burn_rate,
            revenue_delta_3m=self._runway.revenue_delta_3m,
            churn_rate=sig["noisy_churn"],
            nps_score=sig["noisy_nps"],
            user_complaints=sig["user_complaints"],
            competitor_launched=sig["competitor_launched"],
            competitor_play=self._competitor.current_play.value,
            competitor_strength=self._competitor.strength,
            founder_advice=self._internal_snapshot.get("founder_advice", "stay_course"),
            founder_confidence=self._founder.get_confidence(),
            investor_sentiment=self._investor.sentiment,
            next_milestone=self._investor.get_current_milestone(),
            pivot_cost_estimate=3,
            revenue_trend=trends["revenue_trend"],
            churn_trend=trends["churn_trend"],
            complaint_shift_detected=trends["complaint_shift_detected"],
            months_at_risk=trends["months_at_risk"],
            step=self._step_num,
            max_steps=MAX_STEPS,
        )
