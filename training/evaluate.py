"""
Evaluate a trained model against baseline agents across all 5 scenarios.

Updated for CoFounderEnvironment (plan.md Step 7):
  - Tracks 30+ dimensional balanced scorecard (not just pivot rate)
  - Evaluates PMF health, team morale, unit economics, founder trust
  - Uses CoFounderObservation / CoFounderAction class names
  - Backward compatible: also accepts old PivotObservation/PivotAction names

Usage:
  python training/evaluate.py --model_path ./trained_model --n_episodes 100
  python training/evaluate.py --baselines_only --n_episodes 50
"""
import argparse
import json
import os
import sys
import pathlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.baseline_agent import (
    RandomAgent, StubbornAgent, PanicAgent,
    StrategistAgent, run_episodes,
)

SCENARIOS_DIR = pathlib.Path(__file__).parent.parent / "scenarios"
PLOTS_DIR     = pathlib.Path(__file__).parent.parent / "plots"


def load_scenarios() -> list[dict]:
    scenarios = []
    for f in sorted(SCENARIOS_DIR.glob("*.json")):
        with open(f) as fh:
            scenarios.append(json.load(fh))
    return scenarios


def evaluate_trained_model(model_path: str, scenarios: list[dict], n_episodes: int) -> list[dict]:
    """Load a trained HuggingFace model and evaluate it on all scenarios."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError:
        print("transformers not installed — skipping trained model evaluation")
        return []

    from models import CoFounderAction, ActionType, CoFounderObservation
    from server.cofounder_environment import CoFounderEnvironment
    from server.prompt_encoder import encode_to_messages
    from training.market_data import infer_sector_from_scenario

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model.eval()

    ACTION_MAP = {
        "execute":            ActionType.EXECUTE,
        "pivot":              ActionType.PIVOT,
        "research":           ActionType.RESEARCH,
        "fundraise":          ActionType.FUNDRAISE,
        "hire":               ActionType.HIRE,
        "cut_costs":          ActionType.CUT_COSTS,
        "cut costs":          ActionType.CUT_COSTS,
        "sell":               ActionType.SELL,
        "launch_feature":     ActionType.LAUNCH_FEATURE,
        "launch feature":     ActionType.LAUNCH_FEATURE,
        "marketing_campaign": ActionType.MARKETING_CAMPAIGN,
        "marketing campaign": ActionType.MARKETING_CAMPAIGN,
        "set_pricing":        ActionType.SET_PRICING,
        "fire":               ActionType.FIRE,
        "partnership":        ActionType.PARTNERSHIP,
    }

    def llm_act(obs: CoFounderObservation, sector: str) -> CoFounderAction:
        messages = encode_to_messages(obs, sector=sector)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        decoded = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        # Parse DECISION: prefix first
        action_str = "execute"
        for line in decoded.split("\n"):
            line_stripped = line.strip()
            if line_stripped.upper().startswith("DECISION:"):
                action_str = line_stripped.split(":", 1)[1].strip().lower()
                break
        if action_str == "execute":
            action_str = decoded.strip().lower().split()[0] if decoded.strip() else "execute"

        action_type = ACTION_MAP.get(action_str, ActionType.EXECUTE)
        return CoFounderAction(action_type=action_type)

    class TrainedAgent:
        name = "trained_llm"
        def act(self, obs, sector="b2b_enterprise"):
            return llm_act(obs, sector)

    results = []
    agent   = TrainedAgent()
    for scenario in scenarios:
        sector = infer_sector_from_scenario(scenario.get("name", ""))
        r = run_episodes(agent, scenario, n_episodes, seed=999)
        results.append(r)
        print(f"  trained_llm  {scenario['name']:20s}  "
              f"reward={r['mean_reward']:7.1f}  "
              f"survival={r['survival_rate']:.0%}  "
              f"pmf={r.get('mean_final_pmf', 0):.2f}  "
              f"morale={r.get('mean_final_morale', 0):.2f}  "
              f"ltv_cac={r.get('mean_final_ltv_cac', 0):.1f}")
    return results


def make_comparison_plots(all_results: list[dict], save_dir: pathlib.Path):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("matplotlib not installed — skipping plots")
        return

    save_dir.mkdir(exist_ok=True)
    scenarios = sorted(set(r["scenario"] for r in all_results))
    agents    = sorted(set(r["agent"] for r in all_results))
    colors    = {
        "random":      "#888888",
        "stubborn":    "#ff6b6b",
        "panic":       "#ffaa00",
        "strategist":  "#44aaff",
        "trained_llm": "#00ff88",
    }

    # ── Chart 1: Mean reward by agent ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="#0d0d0d")
    fig.patch.set_facecolor("#0d0d0d")
    for ax in axes.flat:
        ax.set_facecolor("#0d0d0d")
        ax.tick_params(colors="white")
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    metrics = [
        ("mean_reward",        "Mean Reward"),
        ("survival_rate",      "Survival Rate"),
        ("mean_final_pmf",     "Final PMF Score"),
        ("mean_final_morale",  "Final Team Morale"),
        ("mean_final_ltv_cac", "Final LTV:CAC"),
        ("mean_final_trust",   "Final Founder Trust"),
    ]

    for ax, (metric, title) in zip(axes.flat, metrics):
        agent_vals = {}
        for agent in agents:
            vals = [r.get(metric, 0) for r in all_results if r["agent"] == agent]
            agent_vals[agent] = sum(vals) / len(vals) if vals else 0

        scale = 100 if metric == "survival_rate" else 1
        bars  = ax.bar(
            list(agent_vals.keys()),
            [v * scale for v in agent_vals.values()],
            color=[colors.get(a, "#aaaaaa") for a in agent_vals],
        )
        ax.set_title(title, color="white", pad=8, fontsize=10)
        ax.set_ylabel(f"{title}", color="white", fontsize=8)
        for bar, val in zip(bars, agent_vals.values()):
            label = f"{val*scale:.1f}{'%' if metric == 'survival_rate' else ''}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    label, ha="center", color="white", fontsize=8)

    plt.suptitle("CoFounder Strategist — Balanced Scorecard Comparison", color="white", fontsize=14, y=1.01)
    plt.tight_layout()
    path = save_dir / "agent_comparison_balanced.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"Saved {path}")


def log_to_wandb(all_results: list[dict]):
    try:
        from dotenv import load_dotenv
        load_dotenv()
        import wandb
        wandb.init(project=os.getenv("WANDB_PROJECT", "models-nexica-ai"),
                   name="evaluation_v2", reinit=True)
        for r in all_results:
            base = f"eval/{r['agent']}/{r['scenario']}"
            wandb.log({
                f"{base}/mean_reward":        r["mean_reward"],
                f"{base}/survival_rate":      r["survival_rate"],
                f"{base}/pivot_rate":         r["pivot_rate"],
                f"{base}/mean_final_pmf":     r.get("mean_final_pmf", 0),
                f"{base}/mean_final_morale":  r.get("mean_final_morale", 0),
                f"{base}/mean_final_ltv_cac": r.get("mean_final_ltv_cac", 0),
                f"{base}/mean_final_trust":   r.get("mean_final_trust", 0),
                f"{base}/balanced_score":     r.get("mean_balanced_score", 0),
            })
        wandb.finish()
        print("Results logged to W&B")
    except Exception as e:
        print(f"W&B logging skipped: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",    type=str, default=None)
    parser.add_argument("--n_episodes",    type=int, default=50)
    parser.add_argument("--baselines_only",action="store_true")
    parser.add_argument("--no_wandb",      action="store_true")
    args = parser.parse_args()

    scenarios = load_scenarios()
    print(f"Loaded {len(scenarios)} scenarios")

    all_results    = []
    baseline_agents = [RandomAgent(), StubbornAgent(), PanicAgent(), StrategistAgent()]

    print("\n=== BASELINE AGENTS ===")
    for scenario in scenarios:
        print(f"\nScenario: {scenario['display_name']}")
        for agent in baseline_agents:
            r = run_episodes(agent, scenario, args.n_episodes)
            all_results.append(r)
            print(f"  {agent.name:12s}  reward={r['mean_reward']:7.1f}  "
                  f"survival={r['survival_rate']:.0%}  "
                  f"pmf={r.get('mean_final_pmf', 0):.2f}  "
                  f"morale={r.get('mean_final_morale', 0):.2f}")

    if not args.baselines_only and args.model_path:
        print("\n=== TRAINED MODEL ===")
        trained = evaluate_trained_model(args.model_path, scenarios, args.n_episodes)
        all_results.extend(trained)

    make_comparison_plots(all_results, PLOTS_DIR)

    if not args.no_wandb:
        log_to_wandb(all_results)

    out_path = pathlib.Path(__file__).parent / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
