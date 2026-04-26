"""
Converts CoFounderObservation (30+ fields) into natural language the LLM reads.

Plan.md Section 6: Separate short-term tactical alerts (low runway, competitor
poaching) from long-term strategic metrics (degrading PMF, burned-out CEO).

Two outputs:
  - encode_observation(obs, sector) → full situation briefing
  - SYSTEM_PROMPT → explains the Strategist Co-Founder role + output format
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import CoFounderObservation

# Import real benchmark data — falls back gracefully if module not found
try:
    from training.market_data import (
        get_benchmarks,
        churn_vs_benchmark,
        nps_vs_benchmark,
        runway_vs_benchmark,
        find_similar_pivot_case,
        infer_sector_from_scenario,
        get_fundraise_context,
        EMPIRICAL_RULES,
    )
    _BENCHMARKS_AVAILABLE = True
except ImportError:
    _BENCHMARKS_AVAILABLE = False


# ─────────────────���───────────────────────────────────────────────────────────
# SYSTEM PROMPT — Strategist Co-Founder Advisor
# ──────────────────────────────────────────────���──────────────────────────────
SYSTEM_PROMPT = """You are a highly experienced and trusted Co-Founder advisor, speaking directly to the founding CEO. You are not a specialized 'pivot expert' but a long-term strategic partner.

Each month you analyze the startup's state and give strategic advice in a natural, conversational format.

Your response MUST follow this logic:
1. Greet the CEO conversationally and recap recent events (what happened based on your previous memory of the situation).
2. Discuss the current status naturally, noting key metrics (runway, PMF, morale).
3. Offer your guidance and reasoning for the best path forward, warning of tradeoffs like a co-founder would.
4. Conclude your conversation by clearly stating your recommendation on a new line like this:
RECOMMENDATION: [action]

Valid action words: EXECUTE | PIVOT | RESEARCH | FUNDRAISE | HIRE | CUT_COSTS | SELL | LAUNCH_FEATURE | MARKETING_CAMPAIGN | SET_PRICING | FIRE | PARTNERSHIP

Rules:
- Be conversational and supportive, speaking as a co-founder. Do NOT output rigid structural blocks (like SITUATION: or OPTIONS:).
- RECOMMENDATION must be on its own line starting with "RECOMMENDATION:"
- Cite specific numbers naturally.
- Warn the CEO naturally about consequences, such as morale hangovers from FIRE or runway costs from PIVOT.
- If founder trust is low, work to rebuild it conversationally."""


# ───────────────────────────────────────────��─────────────────────────────────
# Main encoder
# ────────────────────────────────────────────────���────────────────────────────

def encode_observation(obs: CoFounderObservation, sector: str = "b2b_enterprise") -> str:
    """
    Convert a CoFounderObservation (30+ fields) into a comprehensive briefing.
    Structured into short-term tactical alerts and long-term strategic metrics.
    """
    lines = []

    # ── Header ────────────────���───────────────────────────────────────────────
    months_left = obs.max_steps - obs.step
    lines.append(f"=== MONTH {obs.step + 1} OF {obs.max_steps} | SCENARIO: {obs.scenario.upper()} | {months_left} months remaining ===")
    lines.append("")

    # ── FINANCIAL STATE ──────────────────────────────���───────────────────���─────
    lines.append("--- FINANCIAL STATE ---")
    runway_urgency = _runway_label(obs.runway_remaining)
    lines.append(f"  Runway        : {obs.runway_remaining} months  {runway_urgency}")
    if obs.months_at_risk > 0:
        lines.append(f"  WARNING: Under 6-month runway for {obs.months_at_risk} consecutive months")
    net = obs.net_cash_flow
    sign = "+" if net >= 0 else ""
    lines.append(f"  Revenue       : ${obs.monthly_revenue:,.0f}/mo  |  Burn: ${obs.burn_rate:,.0f}/mo  |  Net: {sign}${net:,.0f}/mo")
    trend_emoji = {"growing": "up", "plateauing": "flat", "declining": "down"}
    lines.append(f"  Revenue trend : {obs.revenue_trend} ({trend_emoji.get(obs.revenue_trend, '')})")
    if obs.revenue_delta_3m != 0:
        sign2 = "+" if obs.revenue_delta_3m > 0 else ""
        lines.append(f"  3-mo change   : {sign2}{obs.revenue_delta_3m:.1%}")
    if obs.total_raised_usd > 0:
        lines.append(f"  Total raised  : ${obs.total_raised_usd:,.0f}")
    lines.append("")

    # ── PRODUCT HEALTH ────────────────────────────────────────────────────────���
    lines.append("--- PRODUCT HEALTH ---")
    lines.append(f"  PMF Score     : {obs.pmf_score:.0%}  {_pmf_label(obs.pmf_score)}")
    lines.append(f"  Tech Debt     : {obs.tech_debt_severity.upper()} ({obs.tech_debt_ratio:.0%} of eng capacity)")
    if obs.tech_debt_severity in ("high", "critical"):
        lines.append("  WARNING: High tech debt is slowing feature velocity and inflating burn.")
    lines.append(f"  Features      : {obs.features_shipped_last} shipped last month  |  {obs.feature_pipeline_depth} queued")
    lines.append("")

    # ── TEAM DYNAMICS ──────────────────────────────────────────────────────────
    lines.append("--- TEAM DYNAMICS ---")
    lines.append(f"  Team Morale   : {obs.team_morale:.0%}  {_morale_label(obs.team_morale)}")
    if obs.team_morale < 0.40:
        lines.append("  MORALE CRITICAL: Feature velocity 50% slower. Engineers may quit.")
    lines.append(f"  Headcount     : {obs.team_size} total  [ Eng: {obs.eng_headcount}  |  Sales: {obs.sales_headcount}  |  Support: {obs.support_headcount} ]")
    lines.append("")

    # ── MARKETING & UNIT ECONOMICS ─────────────────────────��───────────────────
    lines.append("--- MARKETING & UNIT ECONOMICS ---")
    lines.append(f"  LTV : CAC     : {obs.ltv_cac_ratio:.1f}x  (LTV ${obs.ltv:,.0f}  |  CAC ${obs.cac:,.0f})")
    if obs.ltv_cac_ratio < 1.5:
        lines.append("  UNIT ECONOMICS BROKEN: Every customer acquired costs more than they return.")
    lines.append(f"  Brand         : {obs.brand_awareness:.0%} awareness  |  Pipeline: ${obs.pipeline_generated:,.0f} this month")
    lines.append("")

    # ── MARKET SIGNALS ─────────────────────────────────────────────────────────
    lines.append("--- MARKET SIGNALS (noisy readings) ---")
    lines.append(f"  Churn         : {obs.churn_rate:.1%}/mo  — {obs.churn_trend}")
    lines.append(f"  NPS           : {obs.nps_score}  {_nps_label(obs.nps_score)}")
    if obs.complaint_shift_detected:
        lines.append("  ALERT: Customer complaint types changed significantly this month.")
    lines.append(f"  Complaints    : {', '.join(obs.user_complaints)}")

    _comp_play_text = {
        "dormant":        "Competitor is quiet.",
        "launch_feature": "Rival launched a competing feature.",
        "price_war":      "Competitor slashing prices to undercut.",
        "talent_raid":    "Rival poaching your engineers.",
        "aggressive_mkt": "Competitor flooding paid marketing channels.",
        "vacuum_grab":    "Competitor grabbing users you left behind after your pivot.",
    }
    strength_label = "emerging" if obs.competitor_strength < 0.5 else ("strong" if obs.competitor_strength < 0.8 else "dominant")
    if obs.competitor_play != "dormant":
        lines.append(f"  COMPETITOR ({strength_label}, {obs.competitor_strength:.0%}): {_comp_play_text.get(obs.competitor_play, '')}")
    if obs.competitor_launched:
        lines.append("  ALERT: Competitor made a major market move.")
    lines.append("")

    # ── INDUSTRY BENCHMARKS ──────────────────────────────��─────────────────────
    if _BENCHMARKS_AVAILABLE:
        bench = get_benchmarks(sector)
        lines.append(f"--- INDUSTRY BENCHMARKS ({bench['label']}) ---")
        lines.append(f"  Churn    : {churn_vs_benchmark(obs.churn_rate, sector)}")
        lines.append(f"  NPS      : {nps_vs_benchmark(obs.nps_score, sector)}")
        lines.append(f"  Runway   : {runway_vs_benchmark(obs.runway_remaining, sector)}")
        mom_growth    = obs.revenue_delta_3m / 3.0 if obs.revenue_delta_3m else 0.0
        fundraise_ctx = get_fundraise_context(sector, obs.monthly_revenue, mom_growth, obs.nps_score)
        lines.append(f"  Series A : {fundraise_ctx}")
        churn_elevated = obs.churn_rate > bench["avg_monthly_churn"] * 1.5
        nps_poor       = obs.nps_score < bench.get("poor_nps", 10)
        if churn_elevated or nps_poor:
            case = find_similar_pivot_case(
                runway_remaining=obs.runway_remaining,
                churn_rate=obs.churn_rate,
                sector=sector,
                step=obs.step,
            )
            if case:
                lines.append(f"  REAL CASE: {case['company']} pivoted from {case['from_product']} to {case['to_product']}")
                lines.append(f"     Month {case['pivot_month']}, {case['runway_at_pivot']}mo runway | Outcome: {case['outcome']}")
                lines.append(f"     Lesson: {case['lesson']}")
        lines.append("")

    # ── FOUNDER / CEO STATE ────────────────────────────────────────────────────
    lines.append("--- FOUNDER / CEO STATE ---")
    lines.append(f"  Advice        : '{obs.founder_advice.replace('_', ' ')}'")
    lines.append(f"  Confidence    : {obs.founder_confidence:.0%}  {_confidence_label(obs.founder_confidence)}")
    lines.append(f"  Trust in You  : {obs.founder_trust:.0%}  {_trust_label(obs.founder_trust)}")
    if obs.founder_trust < 0.30:
        lines.append("  TRUST CRITICAL: Founder may override your advice. Actions need to rebuild trust.")
    lines.append(f"  Burnout       : {obs.founder_burnout:.0%}  {_burnout_label(obs.founder_burnout)}")
    if obs.founder_burnout > 0.70:
        lines.append("  WARNING: CEO judgment impaired by burnout.")
    lines.append(f"  Stubbornness  : {obs.founder_stubbornness:.0%} probability of ignoring advice")
    lines.append("")

    # ── INVESTOR / BOARD STATE ─────────────────────────────────────────────────
    lines.append("--- INVESTOR / BOARD STATE ---")
    lines.append(f"  Sentiment     : {obs.investor_sentiment:.0%}  {_sentiment_label(obs.investor_sentiment)}")
    lines.append(f"  Next milestone: \"{obs.next_milestone}\"")
    if obs.board_demands:
        lines.append(f"  Board demands : {' | '.join(obs.board_demands)}")
    lines.append("")

    # ── PIVOT ECONOMICS ───────────────────────────────────────────���────────────
    lines.append("--- PIVOT ECONOMICS ---")
    lines.append(f"  Cost          : ~{obs.pivot_cost_estimate} months runway + revenue resets to ~60%")
    if obs.runway_remaining <= obs.pivot_cost_estimate + 2:
        lines.append("  WARNING: Pivoting now may leave insufficient runway to recover.")
    lines.append("")

    # ── SHOCK EVENT ───────────────────────────────────────────────────────────
    if obs.shock_message:
        lines.append(f"MACRO EVENT THIS MONTH: {obs.shock_message}")
        lines.append("")

    # ── BOARD ULTIMATUM ────────────────────────────────────────────��──────────
    if obs.board_pressure:
        lines.append("BOARD ULTIMATUM: Past month 40 with under 6 months runway.")
        lines.append("Acceptable actions: PIVOT | CUT_COSTS | FUNDRAISE | SELL | FIRE")
        lines.append("")

    # ── Decision prompt ──────────────────���──────────────────────────────��─────
    lines.append("Review the metrics casually. Note short-term details and long-term concerns.")
    lines.append("Respond as a co-founder conversationally and wrap up with your action call: RECOMMENDATION / ACTION.")

    return "\n".join(lines)


def encode_to_messages(
    obs: CoFounderObservation,
    history: list[dict] | None = None,
    sector: str = "b2b_enterprise",
) -> list[dict]:
    """Returns conversation in HuggingFace chat format for training loop."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        for h in history[-3:]:
            reward_str = f"{h.get('reward', 0):+.1f}"
            shock_str  = f" [SHOCK: {h['shock']}]" if h.get("shock") else ""
            messages.append({"role": "user", "content": f"[Month {h['step']+1}]: We were dealing with {shock_str}."})
            messages.append({"role": "assistant", "content": f"Hey, in regards to that, my RECOMMENDATION: {h['action']}. (Our results yielded reward {reward_str}, bringing our runway to {h['runway']}mo)."})

    messages.append({"role": "user", "content": encode_observation(obs, sector=sector)})
    return messages


# ── Label helpers ──────────────────────────────��───────────────────────���──────

def _runway_label(months: int) -> str:
    if months >= 18:   return "(comfortable)"
    elif months >= 12: return "(healthy)"
    elif months >= 6:  return "(watch closely)"
    elif months >= 3:  return "CRITICAL"
    else:              return "EMERGENCY - imminent shutdown"

def _pmf_label(pmf: float) -> str:
    if pmf >= 0.75:   return "(strong PMF - users love the product)"
    elif pmf >= 0.55: return "(moderate PMF - room to improve)"
    elif pmf >= 0.35: return "(weak PMF - product not resonating)"
    else:             return "(PMF FAILURE - pivot signal)"

def _nps_label(nps: int) -> str:
    if nps >= 50:    return "(excellent)"
    elif nps >= 20:  return "(good)"
    elif nps >= 0:   return "(neutral)"
    elif nps >= -20: return "(poor)"
    else:            return "(very poor)"

def _morale_label(morale: float) -> str:
    if morale >= 0.75:  return "high - team energized"
    elif morale >= 0.55: return "moderate"
    elif morale >= 0.40: return "low - productivity slipping"
    elif morale >= 0.25: return "very low - engineers considering leaving"
    else:               return "CRITICAL - team in crisis"

def _confidence_label(confidence: float) -> str:
    if confidence >= 0.75: return "high confidence"
    elif confidence >= 0.50: return "moderate confidence"
    elif confidence >= 0.30: return "low confidence"
    else:                    return "very low - founder may be in panic mode"

def _trust_label(trust: float) -> str:
    if trust >= 0.70:   return "trusts your judgment"
    elif trust >= 0.50: return "cautiously receptive"
    elif trust >= 0.30: return "skeptical - needs results to rebuild trust"
    else:               return "lost confidence in advisor"

def _burnout_label(burnout: float) -> str:
    if burnout < 0.30:   return "fresh - clear-headed"
    elif burnout < 0.55: return "moderate stress"
    elif burnout < 0.75: return "high stress - judgment degrading"
    else:               return "burned out - unreliable advice"

def _sentiment_label(sentiment: float) -> str:
    if sentiment >= 0.75:   return "bullish - likely to approve funding"
    elif sentiment >= 0.50: return "cautiously positive"
    elif sentiment >= 0.30: return "skeptical - missed milestones"
    else:                   return "cold - losing confidence"
