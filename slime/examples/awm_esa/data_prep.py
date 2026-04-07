"""
Prepare ESA (Execution-Status Awareness) training data.

Same as AWM data_prep but uses the ESA system prompt with
the structured control block (CONTINUE/VERIFY/REPLAN/STOP).

Usage:
  cd slime/
  python examples/awm_esa/data_prep.py \
    --awm_root ../agent-world-model \
    --output_dir examples/awm_esa/data
"""
import argparse
import json
import os
import random
import sys

# Import ESA system prompt
ESA_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ESA_DIR)
from generate_with_esa import get_esa_system_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--awm_root", default="../agent-world-model")
    parser.add_argument("--output_dir", default="examples/awm_esa/data")
    parser.add_argument("--train_ratio", type=float, default=0.76)
    parser.add_argument("--val_ratio", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.awm_root not in sys.path:
        sys.path.insert(0, args.awm_root)
    from awm.tools import tools_jsonl_load, normalize_scenario_name

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    envs = {normalize_scenario_name(e["scenario"])
            for e in tools_jsonl_load(os.path.join(args.awm_root, "outputs/gen_envs.jsonl"))}

    verifier_keys = set()
    for item in tools_jsonl_load(os.path.join(args.awm_root, "outputs/gen_verifier.pure_code.jsonl")):
        s = normalize_scenario_name(item["scenario"])
        verifier_keys.add(f"{s}::{item['task_idx']}")

    tasks_by_scenario = {}
    for item in tools_jsonl_load(os.path.join(args.awm_root, "outputs/gen_tasks.jsonl")):
        s = normalize_scenario_name(item["scenario"])
        if s in envs:
            tasks_by_scenario[s] = item["tasks"]

    valid = [s for s in tasks_by_scenario
             if any(f"{s}::{i}" in verifier_keys for i in range(len(tasks_by_scenario[s])))]

    print(f"Scenarios: {len(envs)} with envs, {len(valid)} with verified tasks")

    # Split by scenario
    random.seed(args.seed)
    random.shuffle(valid)
    n_train = int(len(valid) * args.train_ratio)
    n_val = int(len(valid) * args.val_ratio)

    splits = {
        "train": valid[:n_train],
        "val": valid[n_train:n_train + n_val],
        "test": valid[n_train + n_val:],
    }

    for name, scenarios in splits.items():
        path = os.path.join(args.output_dir, f"{name}.jsonl")
        count = 0
        with open(path, "w") as f:
            for scenario in scenarios:
                for idx, task in enumerate(tasks_by_scenario.get(scenario, [])):
                    if f"{scenario}::{idx}" not in verifier_keys:
                        continue
                    row = {
                        "text": [
                            {"role": "system", "content": get_esa_system_prompt()},
                            {"role": "user", "content": task},
                        ],
                        "metadata": {
                            "scenario": scenario,
                            "task": task,
                            "task_idx": idx,
                        },
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
        print(f"  {name}: {count} tasks from {len(scenarios)} scenarios -> {path}")


if __name__ == "__main__":
    main()
