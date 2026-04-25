"""
TeamManager — tracks organizational health for the CoFounder environment.

Manages:
  - Headcount split: Engineering, Sales, Support
  - Team morale (0-1): the single most impactful hidden variable in real startups
  - Monthly payroll: sum of headcount × role-specific salaries
  - Hiring ramp: new hires take 2-3 months to reach full productivity
  - RIF Hangover: layoffs cause 3-month morale crash in remaining team

Real-world data sources:
  - Carta 2024: startup annual attrition 25% vs 13% economy-wide
  - Pave Compensation 2024: seed-stage eng $140K base, sales $110K OTE, support $75K
  - McKinsey 2023: RIF survivors lose 35% productivity for 4-6 months
  - First Round Capital: team morale < 40% correlates with 2× product defect rate

Cross-manager interactions (resolved in cofounder_environment.py):
  - Morale < 0.4 → ProductManager velocity drops sharply
  - FIRE action → morale -0.20 (RIF hangover) + burn drops
  - Competitor TALENT_RAID → headcount -1 eng, morale -0.10
  - Revenue + runway improving → morale recovers +0.03/mo
"""
from __future__ import annotations
import random


# ── Salary constants (Pave 2024 seed-stage data) ─────────────────────────────
SALARY_ENG     = 11_667   # $140K / 12 = $11,667/mo
SALARY_SALES   =  9_167   # $110K / 12 = $9,167/mo
SALARY_SUPPORT =  6_250   # $75K  / 12 = $6,250/mo

# Hiring ramp: months until a new hire reaches full productivity
HIRING_RAMP_MONTHS = {"eng": 2, "sales": 4, "support": 1}


class TeamManager:
    """
    Tracks organizational structure, morale, and payroll.
    Called each step by CoFounderEnvironment.
    """

    def __init__(
        self,
        initial_eng:     int   = 4,
        initial_sales:   int   = 2,
        initial_support: int   = 2,
        initial_morale:  float = 0.75,
        rng_seed: int = 42,
    ):
        self._eng     = initial_eng
        self._sales   = initial_sales
        self._support = initial_support
        self._morale  = float(initial_morale)
        self._rng     = random.Random(rng_seed)

        # Ramp tracking: list of (role, months_remaining) for each new hire
        self._ramping: list[tuple[str, int]] = []
        # RIF hangover countdown (months remaining of morale penalty)
        self._rif_hangover = 0
        # Total fires this episode (used in reward)
        self._total_fires = 0
        # Total hires this episode
        self._total_hires = 0

    # ── Main step update ──────────────────────────────────────────────────────

    def tick(
        self,
        runway_remaining: int,
        monthly_revenue: float,
        burn_rate: float,
        phase_name: str,
    ) -> None:
        """
        Update morale and advance hire ramp timers each month.
        Called BEFORE action processing.
        """
        self._advance_ramp_timers()
        self._update_morale(runway_remaining, monthly_revenue, burn_rate, phase_name)

    # ── Action handlers ───────────────────────────────────────────────────────

    def handle_hire(self, role: str = "eng") -> dict:
        """
        HIRE action: add headcount to a specific role.
        New hire costs salary immediately but isn't fully productive for 2-4 months.
        Returns burn increase and effects dict.
        """
        role = role.lower() if role.lower() in ("eng", "sales", "support") else "eng"
        ramp_months = HIRING_RAMP_MONTHS[role]

        if role == "eng":
            self._eng += 1
        elif role == "sales":
            self._sales += 1
        else:
            self._support += 1

        self._ramping.append((role, ramp_months))
        self._total_hires += 1

        # Small morale boost from growth signal
        self._morale = min(1.0, self._morale + 0.03)

        salary = {"eng": SALARY_ENG, "sales": SALARY_SALES, "support": SALARY_SUPPORT}[role]
        return {
            "role": role,
            "burn_increase": salary,
            "ramp_months": ramp_months,
            "message": f"Hired {role} engineer. Burn +${salary:,}/mo. Full productivity in {ramp_months} months.",
        }

    def handle_fire(self, role: str = "eng") -> dict:
        """
        FIRE action: lay off one person.
        Immediately reduces payroll but triggers RIF Hangover (morale -0.20 for 3 months).
        Per McKinsey 2023: survivors lose 35% productivity and confidence for 4-6 months.
        """
        role = role.lower() if role.lower() in ("eng", "sales", "support") else "eng"

        # Pick role with most headcount if requested role has only 1 person
        if role == "eng"     and self._eng     <= 1: role = "sales"
        if role == "sales"   and self._sales   <= 1: role = "support"
        if role == "support" and self._support <= 0: return {"error": "No one to fire", "burn_decrease": 0}

        if role == "eng":
            self._eng -= 1
        elif role == "sales":
            self._sales -= 1
        else:
            self._support -= 1

        # RIF Hangover
        self._morale = max(0.05, self._morale - 0.20)
        self._rif_hangover = max(self._rif_hangover, 3)
        self._total_fires += 1

        salary = {"eng": SALARY_ENG, "sales": SALARY_SALES, "support": SALARY_SUPPORT}[role]
        return {
            "role": role,
            "burn_decrease": salary,
            "rif_hangover_months": 3,
            "morale_hit": -0.20,
            "message": f"Laid off {role}. Burn -${salary:,}/mo. Team morale drops sharply for 3 months.",
        }

    def handle_cut_costs(self) -> None:
        """CUT_COSTS affects morale (uncertainty signal)."""
        self._morale = max(0.10, self._morale - 0.08)

    def handle_fundraise_success(self) -> None:
        """Successful fundraise boosts morale."""
        self._morale = min(1.0, self._morale + 0.12)

    def handle_competitor_talent_raid(self) -> dict:
        """
        Competitor poaches one engineer (TALENT_RAID play).
        Real: Carta 2024 shows 25% annual attrition rises to 40% under active poaching.
        """
        if self._eng <= 1:
            return {"poached": False, "message": "Competitor talent raid deflected (team too small to target)."}

        self._eng -= 1
        self._morale = max(0.05, self._morale - 0.10)
        return {
            "poached": True,
            "burn_decrease": SALARY_ENG,   # salary freed (but recruiting cost offsets)
            "message": "Competitor poached a senior engineer. Morale drops. Budget for recruiting replacement.",
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _advance_ramp_timers(self) -> None:
        """Tick down ramp timers; fully ramped hires contribute normally."""
        self._ramping = [(role, months - 1) for role, months in self._ramping if months > 1]
        if self._rif_hangover > 0:
            self._rif_hangover -= 1

    def _update_morale(
        self,
        runway_remaining: int,
        monthly_revenue: float,
        burn_rate: float,
        phase_name: str,
    ) -> None:
        """
        Morale update logic — the most psychologically nuanced mechanic.
        Morale is a lagging indicator: it takes months to improve and can crash fast.
        """
        # RIF hangover suppresses recovery
        if self._rif_hangover > 0:
            self._morale = max(0.05, self._morale - 0.02)   # continued decay during hangover
            return

        # Financial pressure (runway < 6 = anxiety)
        if runway_remaining < 3:
            self._morale = max(0.05, self._morale - 0.05)
        elif runway_remaining < 6:
            self._morale = max(0.10, self._morale - 0.02)
        elif monthly_revenue > burn_rate * 0.8:
            # Revenue approaching break-even → morale boost
            self._morale = min(1.0, self._morale + 0.015)

        # Phase mood: decline is demoralizing regardless
        if phase_name == "DECLINE":
            self._morale = max(0.10, self._morale - 0.012)

        # Natural recovery toward 0.65 base (teams are generally resilient)
        target_base = 0.65
        if self._morale < target_base:
            self._morale += 0.008   # slow recovery
        elif self._morale > target_base + 0.1:
            self._morale -= 0.005   # mild decay from high euphoria

        self._morale = max(0.05, min(1.0, self._morale))

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def team_size(self) -> int:
        return self._eng + self._sales + self._support

    @property
    def eng_headcount(self) -> int:
        return self._eng

    @property
    def sales_headcount(self) -> int:
        return self._sales

    @property
    def support_headcount(self) -> int:
        return self._support

    @property
    def morale(self) -> float:
        return round(self._morale, 3)

    @property
    def monthly_payroll(self) -> float:
        """Total monthly payroll cost."""
        return (self._eng * SALARY_ENG +
                self._sales * SALARY_SALES +
                self._support * SALARY_SUPPORT)

    @property
    def ramping_count(self) -> int:
        """Number of employees still in ramp-up period."""
        return len(self._ramping)

    @property
    def total_fires(self) -> int:
        return self._total_fires

    @property
    def total_hires(self) -> int:
        return self._total_hires

    def snapshot(self) -> dict:
        return {
            "team_size":         self.team_size,
            "eng_headcount":     self.eng_headcount,
            "sales_headcount":   self.sales_headcount,
            "support_headcount": self.support_headcount,
            "team_morale":       self.morale,
            "monthly_payroll":   self.monthly_payroll,
            "ramping_count":     self.ramping_count,
            "rif_hangover":      self._rif_hangover,
        }
