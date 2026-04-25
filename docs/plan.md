# Plan: Strategist Co-Founder Advisor Environment Transformation

This document outlines the roadmap for transforming the current environment from a single-objective pivot simulation into a comprehensive, multi-dimensional startup strategist co-founder simulator. The upgraded environment will train the agent to act as an expert advisor to a startup founder, managing and giving strategic counsel on Product, Team, and Marketing simultaneously alongside existing Financial, Market, and Competitor dynamics.

## 1. Training Evolution: Before vs. After

### What the Agent Learns
* **Before (Pivot-Only)**: The agent performed single-track optimization. It solely learned to read the exact timing of market decline or saturation and press the `PIVOT` button at the mathematically optimal month.
* **After (Strategist Advisor)**: The agent must act as a strategic co-founder advising the primary founder. It will learn to guide holistic business trade-offs and communicate long-term consequences. This includes advising when to pay down technical debt vs. launching new revenue-generating features, or guiding the founder on whether to increase marketing spend vs. maintaining a healthy Customer Acquisition Cost (CAC), while managing the founder's confidence and trust.

### Observation Space (State)
* **Before**: The observation space contained ~15 fields heavily focused on runway, burn rate, and noisy complaint strings.
* **After**: The state matrix will encompass 30+ fields covering:
  * **Product Health**: Product Market Fit (PMF) scores, technical debt severity, and individual feature success metrics.
  * **Team Dynamics**: Headcount split by roles (Engineering, Sales, Support) and team morale percentages.
  * **Marketing Metrics**: Customer Acquisition Cost (CAC), Lifetime Value (LTV), brand awareness, and pipeline generation.

### Reward Shape
* **Before**: Rewards were sparse and event-driven. A massive point swing (+50 or -50) occurred based almost entirely on hitting the correct "Pivot Window".
* **After**: Rewards will be dense, multi-objective, and dynamically weighted based on the startup's lifecycle phase. The agent will receive continuous micro-rewards and penalties for giving advice that maintains morale above 40%, drives down CAC responsibly, and steers the company clear of crippling tech debt. A substantial chunk of the reward will reflect **Founder Trust & Alignment**, measuring how well the agent’s strategic advice is received and successfully adopted by the simulated primary founder. Survival remains paramount, but optimizing the advisory dynamic is multifaceted. For example, during the "Pre-Seed" phase, the reward function heavily weights product-market-fit signals and ignores technical debt penalties. However, once the startup reaches "Series A" revenue thresholds, the reward function shifts its weights to heavily penalize high churn, poor unit economics, and unmanaged tech debt, accurately reflecting how real-world board expectations change.

### Training Difficulty
* **Before**: The environment was essentially a basic temporal classification problem (when to pivot) with a tiny 7-item action space.
* **After**: The environment requires long-term strategic planning with compounding, delayed consequences (known in reinforcement learning as the "Credit Assignment Problem"). For example, hiring a specialized marketing team in month 4 might only pay off in month 8, and only if the product team actively cleared tech debt in month 5, AND only if the founder trust is high enough to execute the marketing budget. The state space explodes from a few integers to a vast multidimensional matrix tracking HR, engineering, finance, and macro conditions simultaneously. The agent must balance aggressive exploration of new features with the defensive exploitation of existing cash flows. This will require significantly more complex training infrastructure, shifting from simple Q-learning tables to advanced LLM-based policy optimization (e.g., Proximal Policy Optimization combined with Chain-of-Thought prompting), enabling the model to learn deep contextual nuance and plan strategies spanning years of simulated startup time.

---

## 2. Implementation Steps

### Step 1: Rename and Re-scaffold Core Files
* **Rename `server/pivot_environment.py`** to `server/cofounder_environment.py` to correctly reflect the broader scope of the simulation.
* **Rename Base Classes**: Change `ThePivotEnvironment` to `CoFounderEnvironment` to match the new file structure.
* **Update Imports**: Update all intra-project import references in `server/__init__.py`, `server/app.py`, and `client.py` to prevent broken references.

### Step 2: Expand Data Models (`models.py`)
* **Expand `ActionType` Enum**: Explicitly add `LAUNCH_FEATURE`, `MARKETING_CAMPAIGN`, `SET_PRICING`, `FIRE`, and `PARTNERSHIP` to the action space.
* **Rename Schemas**: Rename `PivotObservation` to `CoFounderObservation` and `PivotAction` to `CoFounderAction`.
* **Add New Fields**: Inject detailed observation fields for Product (PMF, tech debt), Team (headcount split, morale), and Marketing (CAC, LTV, brand awareness).

### Step 3: Build Core Subsystem Managers (New Files)
To modularize the complexity, we will break out different decision verticals into their own sub-managers:
* **`server/product_manager.py`**: Will track roadmap progress, calculate technical debt accumulation, and compute product-market fit (PMF) based on user complaints and feature launches.
* **`server/team_manager.py`**: Will track organizational structure, calculate payroll dependencies, handle role-specific hiring/firing, and model team morale decay and boosts.
* **`server/marketing_manager.py`**: Will model customer acquisition cost (CAC) elasticity, track lifetime value (LTV) dynamics based on market state, and simulate marketing campaign Return on Investment (ROI).

### Step 4: Overhaul Existing NPC Agents (`server/`)
Our existing Non-Player Character (NPC) entities must be radically upgraded to react dynamically to the Strategist's advice:
* **`server/founder.py` (The CEO)**: Transition from a static "ghost protocol" advice generator to a fully realized psychological entity. The Founder NPC will now have hidden states for **Burnout**, **Stubbornness**, and **Trust in the Advisor**. If the Strategist communicates poorly, the Founder's trust drops, and they may override or ignore optimal advice. The Strategist must learn "Founder management" alongside business management.
* **`server/investor.py` (The Board/VCs)**: Upgrade from a simple milestone checker (Seed, Series A, Series B) to a complex Board simulator. Investors will now dynamically shift their demands from "Hyper-Growth" to "Profitability" based on the macro market phase. They will issue term sheets with toxic liquidation preferences during distressed times, forcing the Strategist to negotiate dilution versus survival.
* **`server/competitor.py` (The Rival)**: Upgrade from 5 randomized plays to an active reactive agent. If the Strategist advises launching a disruptive feature, the competitor will attempt to fast-follow. If the Strategist advises a price hike, the competitor will trigger a "Price War" play to steal churned users. They will also randomly attempt to poach key engineering hires, directly damaging the `team_manager` metrics.
* **`server/market.py` (Macro Environment)**: Integrate realistic macro shocks (e.g., SVB collapse scenario, sudden interest rate spikes causing SaaS multiples to crash from 20x to 5x) forcing the Strategist to suddenly pivot from growth to immense defensive cost-cutting.

### Step 5: Wire Managers into Main Environment (`server/cofounder_environment.py`)
* **Initialization**: The `reset()` method must deeply instantiate all new subsystem managers (Product, Team, Marketing, Founder, Board) alongside injecting the macro-economy configuration derived from the chosen scenario (e.g., `consumer_app.json`).
* **Routing**: The `step()` cycle now becomes a complex orchestration engine rather than simple logic conditionals. Incoming actions like `LAUNCH_FEATURE` are parsed and routed precisely: The Product Manager defines the complexity of the feature, the Team Manager checks if there are enough unassigned engineer hours to build it this month (while accounting for morale efficiency), and the Founder Manager updates trust levels based on how successfully the feature launches.
* **Cross-Manager Physics (The End-of-Step Propagation)**: Before returning the new state to the agent, the environment resolves how the managers interact with each other. If the Team Manager reports an engineer quit, the Product Manager's delivery timeline instantly extends. If the Marketing Manager runs a campaign driving 1,000 new users but the Product Manager reports high Technical Debt, the Server environment calculates a spike in server instances, inflating the Burn Rate dynamically, which alerts the Board Manager. The interconnectivity simulates the chaotic butterfly-effects inherent in a growing startup.

### Step 6: Overhaul Rewards & Prompts
* **Refactor `server/reward.py`**: Introduce new scoring rubrics for `ProductHealth`, `TeamMorale`, and `LTV_CAC_Ratio`. A substantial chunk of the reward will reflect **Founder Trust & Alignment**, measuring how well the agent’s strategic advice is received and successfully adopted by the simulated primary founder. Survival remains paramount, but optimizing the advisory dynamic is multifaceted.
* **Refactor `server/prompt_encoder.py`**: Inject the expanded state matrix into the LLM's prompt context. Ensure the prompt formatting clearly separates short-term tactical alerts (like low runway or competitor poaching attempts) from long-term strategic metrics (like degrading PMF or a burned-out CEO).

### Step 7: Update Scenarios and Training Curriculum (`scenarios/*.json` & `training/`)
* **Update Configs**: Interject the required base metrics (`team_size`, `pmf_score`, `base_cac`) into the `initial_state` block across all JSON configuration files (e.g., `b2c_saas.json`, `consumer_app.json`).
* **Update `training/evaluate.py`**: Track new metrics over evaluation episodes, shifting the evaluation paradigm from a binary pivot success rate to a comprehensive balanced scorecard approach.

### Step 8: Observability and Baseline Benchmarks
* **Deep Telemetry Mapping**: Expand the `server/wandb_logger.py` to seamlessly graph the 30+ dimensional state space into interactive, multi-layer dashboards tracking Product-Market-Fit scores mapped against CAC decay, and Headcount matched against feature velocity per month. Accurately plotting the exact point where a team's technical debt overwhelmed their ability to ship features is critical for tracking why a trained RL agent ultimately failed to recognize a crisis developing 10 steps prior.
* **Creating Baseline Heuristics (`training/baseline_agent.py`)**: Introduce a pre-programmed, rule-based "baseline CEO" that executes standard startup playbooks perfectly (e.g., aggressively hiring engineering if runway exceeds 14 months, prioritizing bug-fixes if the complaint string 'slow performance' appears for 2 straight months). This heuristic gives the new LLM-based Strategist agent a robust, mathematically competent benchmark to out-perform. The RL agent's strategic advice MUST outlive and out-earn this simple baseline on average to prove it is learning advanced, multi-factor nuance rather than just simple scripts.

---

## 3. Relevant Files Impacted
* `server/pivot_environment.py` → `server/cofounder_environment.py` (Core rewrite)
* `models.py` (ActionType & Pydantic schema expansion)
* `server/reward.py` (Multi-objective reward rubrics)
* `server/prompt_encoder.py` (Stringifying new observations for the LLM)
* `server/investor.py` & `server/competitor.py` (Complexity upgrades)
* `server/wandb_logger.py` (Metrics telemetry expansion)
* `server/app.py` & `client.py` (Scaffolding and API updates)
* `scenarios/*.json` (Configuration schema updates)
* `training/evaluate.py` (New evaluation metrics)
* `training/baseline_agent.py` (New baseline heuristics)

---

## 4. Verification and QA
1. **Extensive Schema & Contract Validation**: Use `pytest` alongside the strict typing system of Pydantic to ensure all inputs and outputs of the massive new observation space strictly abide by schemas. If the API expects `morale` as a floating point between 0.0 and 1.0 but receives an integer index of `40`, the engine will hard-crash during massive RL batch training. Validating API schema contracts is mission-critical when the state space jumps from ~15 integer/string fields to 30+ mixed-type arrays and matrices.
2. **Deep Manual Smoke Testing**: Execute manual episodes via the `client.py` terminal directly acting as the Strategist agent. Track intricate, sequential dependencies (e.g., advising the CEO to fire an engineer, and then carefully inspecting if morale plummets 1 step later, causing the Product Manager's feature velocity metric to crater 2 steps later, leading to increasing customer complaint frequency 3 steps later). If this multi-step cascade of interrelated actions executes flawlessly, the underlying physics engine is healthy.
3. **Agent Loop Stress Validation**: Stress-test the full Reinforcement Learning/LLM agent loop to ensure `evaluate.py` cleanly captures the multi-dimensional reward scoring across 1,000+ fast-forwarded episodes without triggering a memory leak or hanging on API timeouts when requesting continuous LLM reasoning context blocks up to 10,000 tokens long per action cycle.

---

## 5. Strategic Architectural Decisions
* **Manager Pattern**: We chose to incorporate Product, Team, and Marketing as specialized internal managers to modularize the systemic complexity while keeping `step()` readable and maintainable.
* **Reward Shaping**: We are shifting the reward function from a simple, sparse event-driven model to a dense, continuous management scorecard to provide better gradient signals during training and discourage degenerate short-term strategies.

---

## 6. Real Company Data Requirements for Advanced Training

To train this Strategist Co-Founder agent effectively so it generalizes to real-world scenarios, we must transition from synthetic, hand-coded scenarios (e.g., `b2c_saas.json`) to environments driven by massive, high-fidelity real startup datasets. The following external data categories must be ingested to build deeply realistic, dynamic scenario generators:

### A. Macro-Economic & Vertical-Specific Context (2024-2026)
* **Venture Capital Deployment Rates**: Real-time data on active funds, time between rounds (currently stretching from 18 to 24+ months), and graduation rates from Seed to Series A across specific verticals (AI, FinTech, Consumer Social, DeepTech).
* **Valuation Multiples**: Up-to-date ARR (Annual Recurring Revenue) multiples from public market analogs (e.g., BVP Nasdaq Emerging Cloud Index) and private market down-round / cram-down structures.
* **Capital Cost baselines**: Interest rate yield curves affecting debt financing vs. equity financing tradeoffs.

### B. Granular Financial & Cap Table Data
* **Detailed P&L Structures**: Anonymized income statements mapping exact ratios of R&D vs. S&M (Sales & Marketing) vs. G&A (General & Administrative) spend over time (sourced from Carta, PitchBook, or Kruze Consulting).
* **Cap Table Dilution Mechanics**: Standard employee option pool sizes (10-15%), lead investor target ownership (15-25%), pro-rata rights exercises, and the mathematical reality of liquidation preferences during distressed M&A events.
* **Burn Rate & Infrastructure Costs**: True AWS/GCP cloud spend as a percentage of revenue, OpenAI/Anthropic API token costs for AI wrappers, and real estate vs. remote work cost disparities.

### C. Deep Unit Economics (GTM & Growth)
* **Channel-Specific CAC**: Differential Customer Acquisition Cost mapping (e.g., Outbound Enterprise Sales vs. Inbound Content/SEO vs. Paid Social vs. PLG virality).
* **LTV Degradation & Churn Curves**: Logo Churn vs. Net Revenue Retention (NRR). Understanding the difference between a leaky bucket (High Logo Churn) and expansion revenue (NRR > 110%).
* **Payback Periods**: Historical norms extending from 8-12 months (ZIRP era) to 18-24 months (Current era).

### D. Product, Engineering & Tech Debt Telemetry
* **DORA Metrics**: Lead time for changes, deployment frequency, time to restore service, and change failure rate as predictors of engineering health.
* **Tech Debt Compounding Formulas**: Quantitative data proving how a 1-month rushed feature build costs 3 months of refactoring and bug-fixing later into the lifecycle.
* **Product-Market Fit (PMF) Signals**: Real survey distributions (e.g., the Superhuman PMF engine—tracking the exact percentage of users who would be "very disappointed" if the product disappeared).

### E. Organizational & HR Psychology Data
* **Compensation & Equity Bands**: Real-time market salaries from platforms like Pave or Option Impact, cross-referenced by geography and funding stage (e.g., balancing lower base salary with higher equity correctly).
* **Attrition & Productivity Drops**: True turnover rates by job function (Sales vs. Engineering). Crucially, capturing the "Reduction in Force (RIF) Hangover"—quantifying the 3-6 month productivity plunge and morale crash occurring in the surviving team after layoffs.
* **Hiring Ramp Constraints**: The reality that a newly hired Enterprise Account Executive takes 4-6 months just to ramp up and close their first deal, draining cash before returning revenue.

### F. Qualitative Strategic Data (Crisis & Communication)
* **Post-Mortems**: NLP ingestion of Y Combinator startup failure autopsies to learn exact failure patterns (e.g., co-founder disputes, running out of cash exactly 1 week before a term sheet signs).
* **Board Decks & Investor Updates**: Text data on successful vs. failing investor update cadence, tone, and transparency.
* **Crisis PR**: Historic templates on handling data breaches, massive AWS outages, and pivoting narratives.

Building this environment with the absolute **latest post-ZIRP data (2024-2026)** ensures the model isn't trained on outdated zero-interest-rate assumptions where money was free and growth was prized over unit economics. The resulting advice is grounded, incredibly pragmatic, and financially defensive.

---

## 7. The Final Outcome: The Trained Strategist Model

A model trained inside this comprehensive, data-rich environment will behave fundamentally differently from a standard chat assistant (like ChatGPT) giving generic startup advice. It ceases to be a text predictor and becomes a hyper-rational, mathematically grounded **Intellectual Sparring Partner and Board Member** that has "lived" through 100,000 simulated startup lifecycles.

### A. Hyper-Contextual, Quantitative Advice
Instead of generating generic platitudes like *"you should focus on growth,"* it will actively run the company's P&L and metrics in its context window and diagnose fatal flaws:
* **The Output**: *"Your CAC on LinkedIn ads is $450, but your LTV is capped at $900 due to your 6% monthly logo churn. In the current B2B SaaS environment, an LTV:CAC ratio of exactly 2:0 will kill your 14-month runway before you can raise a Series A, because your payback period is exceeding 18 months. Stop paid marketing spend immediately, reassign 2 engineers from the 'New Dashboard' epic to fixing the core onboarding pipeline to stem the churn, and raise your tier-1 pricing by 20% to test price elasticity."*

### B. Proactive Multi-Dimensional Trade-off Analysis
When the human founder suggests an action, the agent will anticipate second- and third-order effects that human bias often overlooks.
* **The Output (Layoff Scenario)**: If a founder suggests firing 20% of the team to extend runway by 6 months, the trained model will calculate the hidden costs: *"Cutting 4 engineers extends our runway to February 2026. However, based on the RIF-Hangover model, remaining engineering morale will drop by 35%. This will crush our feature velocity for the next 4 months, pushing the highly anticipated Enterprise SSO feature past the Q3 buying cycle. We will save $600k in payroll but risk losing $850k in Q3 pipeline. I advise against a layoff; instead, let's explore venture debt or cut our AWS/Datadog infrastructure spend first."*

### C. Executive Stakeholder & Emotional Intelligence
Having been repeatedly penalized during its training phase for losing developer morale or eroding investor confidence precisely when funding was needed, the agent learns the delicate art of managing optics.
* **The Output**: It won't just tell the founder the metrics are bad; it tells them *how* to deliver the message. *"Our NRR dropped below 95% this quarter. Do not hide this in the upcoming board meeting. Investors in 2025 severely punish late surprises. Draft an update that highlights the churn problem on slide 2, paired immediately with our 3-point remediation plan involving the new Customer Success hires. Transparency now preserves the trust we need for the bridge round in 9 months."*

### D. Dynamic Strategy Phase-Shifting
The model will continuously adapt its fundamental personality and risk appetite based on the company's lifecycle stage, exactly as an elite serial entrepreneur would:
* **Pre-Seed / PMF Search**: It advises scrappiness, rapid iteration, and "doing things that don't scale." It tells the founder to ignore tech debt entirely because survival requires validation.
* **Series A / Scaling**: It flips the script. It harshly demands the founder stop writing code, begin hiring functional VPs, standardize the sales playbook, and forcefully start paying down the tech debt accrued in the Seed phase before the system collapses under user load.
* **Distressed / Pivot Mode**: It becomes ruthlessly objective, calculating liquidation scenarios, acqui-hire probabilities, and executing aggressive pivot testing if the core market is proven dead.

**Ultimately**, this environment creates an artificial co-founder that brings the battle-tested, mathematical rigor of a top-tier VC partner, combined with the operational, deep-in-the-weeds grit of a second-time technical founder.
