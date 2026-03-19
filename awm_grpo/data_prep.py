"""
Data preparation: convert AWM tasks to slime-compatible training JSONL.

Each output row contains:
  - text: list of chat messages [system_prompt, user_task]
  - metadata: {scenario, task, task_idx}

Usage:
  python data_prep.py \
    --tasks_path ../agent-world-model/outputs/gen_tasks.jsonl \
    --envs_path ../agent-world-model/outputs/gen_envs.jsonl \
    --verifier_path ../agent-world-model/outputs/gen_verifier.pure_code.jsonl \
    --output_dir ./data \
    --train_ratio 0.8 \
    --val_ratio 0.1
"""
import argparse
import json
import os
import random
import sys

AWM_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent-world-model")
if AWM_ROOT not in sys.path:
    sys.path.insert(0, AWM_ROOT)

from awm.tools import tools_jsonl_load, normalize_scenario_name
from awm.core.agent import get_system_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare AWM data for slime GRPO training")
    parser.add_argument("--tasks_path", type=str,
                        default="../agent-world-model/outputs/gen_tasks.jsonl")
    parser.add_argument("--envs_path", type=str,
                        default="../agent-world-model/outputs/gen_envs.jsonl")
    parser.add_argument("--verifier_path", type=str,
                        default="../agent-world-model/outputs/gen_verifier.pure_code.jsonl")
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--train_ratio", type=float, default=0.76,
                        help="Fraction of scenarios for training (default: 400/526 ≈ 0.76)")
    parser.add_argument("--val_ratio", type=float, default=0.12,
                        help="Fraction of scenarios for validation (default: 60/526 ≈ 0.12)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_scenarios", type=int, default=0,
                        help="Limit number of scenarios (0 = all)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load available scenarios (only those with generated environments)
    envs_data = tools_jsonl_load(args.envs_path)
    available_scenarios = set(normalize_scenario_name(e["scenario"]) for e in envs_data)
    print(f"Available scenarios with environments: {len(available_scenarios)}")

    # Load verifiers to know which tasks have verification
    verifier_keys = set()
    for item in tools_jsonl_load(args.verifier_path):
        scenario = normalize_scenario_name(item["scenario"])
        task_idx = item["task_idx"]
        verifier_keys.add(f"{scenario}::{task_idx}")
    print(f"Available verifiers: {len(verifier_keys)}")

    # Load tasks
    tasks_data = tools_jsonl_load(args.tasks_path)
    tasks_by_scenario: dict[str, list[str]] = {}
    for item in tasks_data:
        scenario = normalize_scenario_name(item["scenario"])
        if scenario in available_scenarios:
            tasks_by_scenario[scenario] = item["tasks"]

    print(f"Scenarios with tasks and environments: {len(tasks_by_scenario)}")

    # Filter to scenarios that have at least one verified task
    valid_scenarios = []
    for scenario in tasks_by_scenario:
        tasks = tasks_by_scenario[scenario]
        has_verifier = any(
            f"{scenario}::{i}" in verifier_keys
            for i in range(len(tasks))
        )
        if has_verifier:
            valid_scenarios.append(scenario)

    print(f"Scenarios with verified tasks: {len(valid_scenarios)}")

    if args.max_scenarios > 0:
        valid_scenarios = valid_scenarios[:args.max_scenarios]

    # Split by scenario (not by task, to avoid train/test leakage)
    random.seed(args.seed)
    random.shuffle(valid_scenarios)

    n_total = len(valid_scenarios)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)

    train_scenarios = valid_scenarios[:n_train]
    val_scenarios = valid_scenarios[n_train:n_train + n_val]
    test_scenarios = valid_scenarios[n_train + n_val:]

    print(f"\nSplit: train={len(train_scenarios)}, val={len(val_scenarios)}, test={len(test_scenarios)}")

    # Get system prompt
    system_prompt = get_system_prompt()

    # Generate JSONL files
    splits = {
        "train": train_scenarios,
        "val": val_scenarios,
        "test": test_scenarios,
    }

    stats = {}
    for split_name, scenarios in splits.items():
        output_path = os.path.join(args.output_dir, f"{split_name}.jsonl")
        count = 0
        verified_count = 0

        with open(output_path, "w", encoding="utf-8") as f:
            for scenario in scenarios:
                tasks = tasks_by_scenario.get(scenario, [])
                for task_idx, task in enumerate(tasks):
                    verifier_key = f"{scenario}::{task_idx}"
                    has_verifier = verifier_key in verifier_keys

                    if not has_verifier:
                        continue  # Skip tasks without verification code

                    # Build chat messages
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": task},
                    ]

                    row = {
                        "text": messages,
                        "metadata": {
                            "scenario": scenario,
                            "task": task,
                            "task_idx": task_idx,
                        },
                    }

                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
                    if has_verifier:
                        verified_count += 1

        stats[split_name] = {
            "scenarios": len(scenarios),
            "tasks": count,
            "verified_tasks": verified_count,
        }
        print(f"  {split_name}: {count} tasks from {len(scenarios)} scenarios -> {output_path}")

    # Save split info
    split_info = {
        "seed": args.seed,
        "splits": {
            name: {
                "scenarios": scenarios,
                **stats[name],
            }
            for name, scenarios in splits.items()
        },
    }
    info_path = os.path.join(args.output_dir, "split_info.json")
    with open(info_path, "w") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    print(f"\nSplit info saved to {info_path}")


if __name__ == "__main__":
    main()
