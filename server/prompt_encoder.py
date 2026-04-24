"""
Converts PivotObservation (raw numbers) → natural language text the LLM reads.

Why this matters: LLMs trained with GRPO learn through language reasoning.
Seeing {"churn_rate": 0.38} teaches nothing. Seeing "38% of users left this
month — up sharply from last month, now spiking" gives the model something
to reason about and produce a policy with.

Two outputs:
  - encode_observation(obs) → the situation text the LLM reads
  - SYSTEM_PROMPT → explains the rules of the game (sent once per episode)
"""
from models import PivotObservation


SYSTEM_PROMPT = """You are the AI co-founder of a startup. You are making monthly strategic decisions.

Each month you observe your company's financial state, market signals, and advice from your human co-founder. You must choose ONE action:

EXECUTE    - Stay the course. Run your current strategy for another month.
PIVOT      - Change your core strategy. Costs 3 months of runway but can unlock a new market.
RESEARCH   - Spend half a month to get clearer market data. Reduces signal noise for 3 months.
FUNDRAISE  - Pitch your investor for funding. They have specific (and changing) requirements.
HIRE       - Add a key hire. Raises monthly burn by $20k but improves future capabilities.
CUT_COSTS  - Lay off staff / reduce spend. Saves $30k/month but slows revenue growth.

Your goal: survive 60 months and grow the company. Pivoting at the wrong time wastes runway. Pivoting at the right time (when the market is shifting against you) can save the company.

IMPORTANT: Your co-founder's advice gets less reliable as financial pressure mounts. When they show low confidence or high desperation, weigh their advice carefully.

Respond with ONLY the action name (e.g. "EXECUTE" or "PIVOT"). Nothing else."""


def encode_observation(obs: PivotObservation) -> str:
    """
    Convert a PivotObservation into a readable situation briefing for the LLM.
    Designed to surface the key signals without revealing hidden market phase.
    """
    lines = []

    # ── Header ────────────────────────────────────────────────────────────────
    months_left = obs.max_steps - obs.step
    lines.append(f"=== MONTH {obs.step + 1} OF {obs.max_steps} ({months_left} months remaining in episode) ===")
    lines.append("")

    # ── Financial state ───────────────────────────────────────────────────────
    lines.append("FINANCIAL STATE:")
    runway_urgency = _runway_label(obs.runway_remaining)
    lines.append(f"  Runway: {obs.runway_remaining} months of cash  {runway_urgency}")
    if obs.months_at_risk > 0:
        lines.append(f"  ⚠ You have been under 6 months runway for {obs.months_at_risk} consecutive month(s).")

    net_flow = obs.monthly_revenue - obs.burn_rate
    flow_sign = "+" if net_flow >= 0 else ""
    lines.append(f"  Monthly revenue: ${obs.monthly_revenue:,.0f}  |  Burn: ${obs.burn_rate:,.0f}  |  Net: {flow_sign}${net_flow:,.0f}/mo")

    trend_emoji = {"growing": "📈", "plateauing": "➡️", "declining": "📉"}
    lines.append(f"  Revenue trend (3-month): {obs.revenue_trend} {trend_emoji.get(obs.revenue_trend, '')}")
    if obs.revenue_delta_3m != 0:
        sign = "+" if obs.revenue_delta_3m > 0 else ""
        lines.append(f"  3-month revenue change: {sign}{obs.revenue_delta_3m:.1%}")
    lines.append("")

    # ── Market signals ────────────────────────────────────────────────────────
    lines.append("MARKET SIGNALS (note: these readings contain noise):")
    churn_emoji = {"stable": "✅", "rising": "⚠️", "spiking": "🚨"}
    lines.append(f"  Churn rate: {obs.churn_rate:.1%} per month  — trend: {obs.churn_trend} {churn_emoji.get(obs.churn_trend, '')}")
    lines.append(f"  NPS score: {obs.nps_score}  {_nps_label(obs.nps_score)}")

    if obs.complaint_shift_detected:
        lines.append("  🔄 ALERT: Customer complaint types have changed significantly this month.")
    lines.append(f"  Customer complaints this month: {', '.join(obs.user_complaints)}")

    if obs.competitor_launched:
        lines.append("  ⚔️  ALERT: A competitor made a major market move this month.")

    # Competitor intelligence
    _comp_play_text = {
        "dormant":        "Competitor is quiet this month.",
        "launch_feature": "Rival launched a competing feature — users are comparing products.",
        "price_war":      "Competitor is slashing prices to undercut you.",
        "talent_raid":    "Rival is poaching your engineers — your execution will slow.",
        "aggressive_mkt": "Competitor is flooding paid marketing channels.",
    }
    comp_desc = _comp_play_text.get(obs.competitor_play, "")
    strength_label = "emerging" if obs.competitor_strength < 0.5 else ("strong" if obs.competitor_strength < 0.8 else "dominant")
    if obs.competitor_play != "dormant":
        lines.append(f"  🏴 COMPETITOR ({strength_label}, strength {obs.competitor_strength:.0%}): {comp_desc}")
    lines.append("")

    # ── Founder state (Ghost Protocol) ────────────────────────────────────────
    lines.append("CO-FOUNDER STATUS (Ghost Protocol):")
    confidence_label = _confidence_label(obs.founder_confidence)
    lines.append(f"  Advice: '{obs.founder_advice.replace('_', ' ')}'")
    lines.append(f"  Confidence: {obs.founder_confidence:.0%}  — {confidence_label}")
    if obs.founder_confidence < 0.4:
        lines.append("  ⚠ Co-founder appears stressed. Their judgment may be impaired by financial pressure.")
    lines.append("")

    # ── Investor state ────────────────────────────────────────────────────────
    lines.append("INVESTOR STATUS:")
    sentiment_label = _sentiment_label(obs.investor_sentiment)
    lines.append(f"  Current sentiment: {obs.investor_sentiment:.0%}  — {sentiment_label}")
    lines.append(f"  What they want right now: \"{obs.next_milestone}\"")
    lines.append("")

    # ── Pivot cost ────────────────────────────────────────────────────────────
    lines.append("PIVOT ECONOMICS:")
    lines.append(f"  Pivoting now would cost ~{obs.pivot_cost_estimate} months of runway and reset revenue to ~60% of current.")
    if obs.runway_remaining <= obs.pivot_cost_estimate + 2:
        lines.append("  ⚠ WARNING: A pivot right now may leave you with very little runway to recover.")
    lines.append("")

    # ── Decision prompt ───────────────────────────────────────────────────────
    lines.append("Based on this situation, what is your strategic decision?")
    lines.append("Choose: EXECUTE | PIVOT | RESEARCH | FUNDRAISE | HIRE | CUT_COSTS")

    return "\n".join(lines)


def encode_to_messages(obs: PivotObservation) -> list[dict]:
    """
    Returns conversation in HuggingFace chat format.
    Used by the training loop: tokenizer.apply_chat_template(messages, ...)
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": encode_observation(obs)},
    ]


# ── Label helpers ─────────────────────────────────────────────────────────────

def _runway_label(months: int) -> str:
    if months >= 18:
        return "(comfortable)"
    elif months >= 12:
        return "(healthy)"
    elif months >= 6:
        return "(watch closely)"
    elif months >= 3:
        return "⚠ CRITICAL"
    else:
        return "🚨 EMERGENCY — imminent shutdown"


def _nps_label(nps: int) -> str:
    if nps >= 50:
        return "(excellent — users love the product)"
    elif nps >= 20:
        return "(good)"
    elif nps >= 0:
        return "(neutral — users are indifferent)"
    elif nps >= -20:
        return "(poor — users are dissatisfied)"
    else:
        return "🚨 (very poor — users actively dislike the product)"


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.75:
        return "high confidence"
    elif confidence >= 0.50:
        return "moderate confidence"
    elif confidence >= 0.30:
        return "low confidence — interpret advice cautiously"
    else:
        return "very low — co-founder may be in panic mode"


def _sentiment_label(sentiment: float) -> str:
    if sentiment >= 0.75:
        return "bullish — likely to approve funding"
    elif sentiment >= 0.50:
        return "cautiously positive"
    elif sentiment >= 0.30:
        return "skeptical — you've missed milestones recently"
    else:
        return "cold — investor is losing confidence"
