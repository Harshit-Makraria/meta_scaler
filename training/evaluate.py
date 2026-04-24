"""
Evaluate a trained model against baseline agents across all 5 scenarios.
Logs results to W&B and saves comparison plots.

Usage:
  # After training, point to your saved model:
  python training/evaluate.py --model_path ./trained_model --n_episodes 100

  # Or evaluate baselines only (no trained model):
  python training/evaluate.py --baselines_only --n_episodes 50
"""
import argparse
import json
import os
import sys
import pathlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.baseline_agent import RandomAgent, StubbornAgent, PanicAgent, run_episodes

SCENARIOS_DIR = pathlib.Path(__file__).parent.parent / "scenarios"
PLOTS_DIR = pathlib.Path(__file__).parent.parent / "plots"


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

    from models import PivotAction, ActionType, PivotObservation
    from server.pivot_environment import ThePivotEnvironment
    from server.prompt_encoder import encode_to_messages

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model.eval()

    ACTION_MAP = {
        "execute": ActionType.EXECUTE,
        "pivot": ActionType.PIVOT,
        "research": ActionType.RESEARCH,
        "fundraise": ActionType.FUNDRAISE,
        "hire": ActionType.HIRE,
        "cut_costs": ActionType.CUT_COSTS,
        "cut costs": ActionType.CUT_COSTS,
    }

    def llm_act(obs: PivotObservation) -> PivotAction:
        messages = encode_to_messages(obs)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        action_str = decoded.strip().lower().split()[0] if decoded.strip() else "execute"
        action_type = ACTION_MAP.get(action_str, ActionType.EXECUTE)
        return PivotAction(action_type=action_type)

    class TrainedAgent:
        name = "trained_llm"
        def act(self, obs): return llm_act(obs)

    results = []
    agent = TrainedAgent()
    for scenario in scenarios:
        r = run_episodes(agent, scenario, n_episodes, seed=999)
        results.append(r)
        print(f"  trained_llm  {scenario['name']:20s}  reward={r['mean_reward']:7.1f}  survival={r['survival_rate']:.0%}")
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
    agents = sorted(set(r["agent"] for r in all_results))
    colors = {"random": "#888888", "stubborn": "#ff6b6b", "panic": "#ffaa00", "trained_llm": "#00ff88"}

    # Reward comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d0d0d")
    fig.patch.set_facecolor("#0d0d0d")

    for ax in axes:
        ax.set_facecolor("#0d0d0d")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Mean reward per agent (averaged across scenarios)
    ax1 = axes[0]
    agent_rewards = {}
    for agent in agents:
        vals = [r["mean_reward"] for r in all_results if r["agent"] == agent]
        agent_rewards[agent] = sum(vals) / len(vals) if vals else 0
    bars = ax1.bar(list(agent_rewards.keys()), list(agent_rewards.values()),
                   color=[colors.get(a, "#aaaaaa") for a in agent_rewards])
    ax1.set_title("Mean Reward by Agent (all scenarios)", color="white", pad=12)
    ax1.set_ylabel("Mean Episode Reward", color="white")
    ax1.set_xlabel("Agent", color="white")
    for bar, val in zip(bars, agent_rewards.values()):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.0f}", ha="center", color="white", fontsize=9)

    # Survival rate per agent
    ax2 = axes[1]
    agent_survival = {}
    for agent in agents:
        vals = [r["survival_rate"] for r in all_results if r["agent"] == agent]
        agent_survival[agent] = sum(vals) / len(vals) if vals else 0
    bars = ax2.bar(list(agent_survival.keys()), [v * 100 for v in agent_survival.values()],
                   color=[colors.get(a, "#aaaaaa") for a in agent_survival])
    ax2.set_title("Survival Rate by Agent (all scenarios)", color="white", pad=12)
    ax2.set_ylabel("Survival Rate %", color="white")
    ax2.set_xlabel("Agent", color="white")
    ax2.set_ylim(0, 110)
    for bar, val in zip(bars, agent_survival.values()):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.0%}", ha="center", color="white", fontsize=9)

    plt.tight_layout()
    path = save_dir / "agent_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"Saved {path}")


def log_to_wandb(all_results: list[dict]):
    try:
        from dotenv import load_dotenv
        load_dotenv()
        import wandb
        wandb.init(project=os.getenv("WANDB_PROJECT", "models-nexica-ai"),
                   name="evaluation", reinit=True)
        for r in all_results:
            wandb.log({
                f"eval/{r['agent']}/{r['scenario']}/mean_reward": r["mean_reward"],
                f"eval/{r['agent']}/{r['scenario']}/survival_rate": r["survival_rate"],
                f"eval/{r['agent']}/{r['scenario']}/pivot_rate": r["pivot_rate"],
            })
        wandb.finish()
        print("Results logged to W&B")
    except Exception as e:
        print(f"W&B logging skipped: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model directory")
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--baselines_only", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    scenarios = load_scenarios()
    print(f"Loaded {len(scenarios)} scenarios")

    all_results = []
    baseline_agents = [RandomAgent(), StubbornAgent(), PanicAgent()]

    print("\n=== BASELINE AGENTS ===")
    for scenario in scenarios:
        print(f"\nScenario: {scenario['display_name']}")
        for agent in baseline_agents:
            r = run_episodes(agent, scenario, args.n_episodes)
            all_results.append(r)
            print(f"  {agent.name:10s}  reward={r['mean_reward']:7.1f}  "
                  f"survival={r['survival_rate']:.0%}  pivot_rate={r['pivot_rate']:.0%}")

    if not args.baselines_only and args.model_path:
        print("\n=== TRAINED MODEL ===")
        trained_results = evaluate_trained_model(args.model_path, scenarios, args.n_episodes)
        all_results.extend(trained_results)

    make_comparison_plots(all_results, PLOTS_DIR)

    if not args.no_wandb:
        log_to_wandb(all_results)

    out_path = pathlib.Path(__file__).parent / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
