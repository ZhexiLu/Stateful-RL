"""
Check which tasks have initial DB state that already satisfies the verifier.
These are "false positive" tasks where the verifier returns COMPLETE even
without any agent actions.
"""
import json
import os
import sys
import sqlite3
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from awm.tools import tools_jsonl_load, normalize_scenario_name
from awm.core.db import create_sqlite_database
from awm.core.sample import execute_sample_data
from awm.core.verifier import execute_verification_code, VerificationMode

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")


def check_single_scenario(args):
    """Check all tasks for a single scenario. Runs in subprocess."""
    scenario, db_schema, sample_data, verifier_items = args

    scenario_norm = normalize_scenario_name(scenario)
    # Each worker uses a unique temp dir to avoid filename conflicts
    pid = os.getpid()
    db_dir = f"/tmp/fp_check_dbs/{scenario_norm}_{pid}"
    os.makedirs(db_dir, exist_ok=True)

    db_path = os.path.join(db_dir, f"{scenario_norm}.db")

    results = []

    try:
        # Create pristine DB
        create_sqlite_database(scenario, db_schema, db_dir)
        execute_sample_data(db_path, sample_data, scenario)

        for v_item in verifier_items:
            task_idx = v_item["task_idx"]
            task = v_item["task"]
            code = v_item.get("verification", {}).get("code", "")

            if not code:
                results.append({
                    "scenario": scenario, "task_idx": task_idx,
                    "task": task, "status": "no_verifier",
                    "initial_passes": False
                })
                continue

            # Find function name
            func_name = "verify_task_completion"
            for line in code.split("\n"):
                line = line.strip()
                if line.startswith("def verify_") and "(" in line:
                    func_name = line.split("(")[0].replace("def ", "").strip()
                    break

            mode = VerificationMode.code if "final_answer" in code else VerificationMode.sql

            try:
                result = execute_verification_code(
                    python_code=code,
                    function_name=func_name,
                    initial_db_path=db_path,
                    mode=mode,
                    final_db_path=db_path,  # same as initial!
                    final_answer="test placeholder",
                )

                exec_result = result.get("result", {})
                passes = (
                    isinstance(exec_result, dict)
                    and str(exec_result.get("result", "")).lower() == "complete"
                )

                results.append({
                    "scenario": scenario, "task_idx": task_idx,
                    "task": task, "status": "checked",
                    "initial_passes": passes,
                })
            except Exception as e:
                results.append({
                    "scenario": scenario, "task_idx": task_idx,
                    "task": task, "status": f"error: {str(e)[:100]}",
                    "initial_passes": False,
                })
    except Exception as e:
        for v_item in verifier_items:
            results.append({
                "scenario": scenario, "task_idx": v_item["task_idx"],
                "task": v_item["task"], "status": f"db_error: {str(e)[:100]}",
                "initial_passes": False,
            })
    finally:
        import shutil
        try:
            shutil.rmtree(db_dir, ignore_errors=True)
        except:
            pass

    return results


def main():
    print("Loading data...")
    verifiers = tools_jsonl_load("outputs/gen_verifier.pure_code.jsonl")
    db_schemas_raw = tools_jsonl_load("outputs/gen_db.jsonl")
    samples_raw = tools_jsonl_load("outputs/gen_sample.jsonl")

    # Index by normalized scenario name
    db_schemas = {normalize_scenario_name(d["scenario"]): d["db_schema"] for d in db_schemas_raw}
    samples = {normalize_scenario_name(s["scenario"]): s.get("sample_data", {}) for s in samples_raw}

    # Group verifiers by scenario
    scenario_verifiers = defaultdict(list)
    for v in verifiers:
        s = normalize_scenario_name(v["scenario"])
        scenario_verifiers[s].append(v)

    print(f"Scenarios: {len(scenario_verifiers)}, Total tasks: {len(verifiers)}")

    # Build worker args
    worker_args = []
    for scenario_norm, v_items in scenario_verifiers.items():
        if scenario_norm not in db_schemas:
            print(f"  SKIP {scenario_norm}: no DB schema")
            continue
        worker_args.append((
            v_items[0]["scenario"],  # original name for DB creation
            db_schemas[scenario_norm],
            samples.get(scenario_norm, {}),
            v_items,
        ))

    print(f"Processing {len(worker_args)} scenarios with multiprocessing...")

    all_results = []
    num_workers = min(os.cpu_count() or 4, 32)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(check_single_scenario, args): args[0]
                   for args in worker_args}

        done = 0
        for future in as_completed(futures):
            done += 1
            results = future.result()
            all_results.extend(results)
            if done % 100 == 0:
                print(f"  {done}/{len(worker_args)} scenarios done...")

    # Analyze results
    total = len(all_results)
    false_positives = [r for r in all_results if r["initial_passes"]]
    checked = [r for r in all_results if r["status"] == "checked"]
    errors = [r for r in all_results if r["status"].startswith("error")]
    no_verifier = [r for r in all_results if r["status"] == "no_verifier"]

    print(f"\n{'='*60}")
    print(f"RESULTS: False Positive Detection")
    print(f"{'='*60}")
    print(f"Total tasks: {total}")
    print(f"Successfully checked: {len(checked)}")
    print(f"Errors: {len(errors)}")
    print(f"No verifier: {len(no_verifier)}")
    print(f"\nFalse positives (initial DB passes verifier): {len(false_positives)}")
    print(f"False positive rate: {len(false_positives)}/{len(checked)} = "
          f"{len(false_positives)/max(len(checked),1)*100:.1f}%")

    # Per-scenario breakdown
    scenario_fp = defaultdict(lambda: [0, 0])  # [fp_count, total_count]
    for r in all_results:
        s = normalize_scenario_name(r["scenario"])
        if r["status"] == "checked":
            scenario_fp[s][1] += 1
            if r["initial_passes"]:
                scenario_fp[s][0] += 1

    # Scenarios sorted by FP rate
    fp_scenarios = {s: (fp, tot) for s, (fp, tot) in scenario_fp.items() if fp > 0}
    clean_scenarios = {s: (fp, tot) for s, (fp, tot) in scenario_fp.items() if fp == 0}

    print(f"\n--- Scenarios with false positives: {len(fp_scenarios)} ---")
    for s, (fp, tot) in sorted(fp_scenarios.items(), key=lambda x: x[1][0]/x[1][1], reverse=True):
        print(f"  {s}: {fp}/{tot} tasks are false positive ({fp/tot*100:.0f}%)")

    print(f"\n--- Clean scenarios (no false positives): {len(clean_scenarios)} ---")

    # Save full results
    output_path = "eval_results/false_positive_analysis.json"
    os.makedirs("eval_results", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "total_tasks": total,
                "checked": len(checked),
                "false_positives": len(false_positives),
                "fp_rate": len(false_positives) / max(len(checked), 1),
                "scenarios_with_fp": len(fp_scenarios),
                "clean_scenarios": len(clean_scenarios),
            },
            "false_positive_tasks": false_positives,
            "per_scenario": {
                s: {"fp": fp, "total": tot}
                for s, (fp, tot) in sorted(scenario_fp.items())
            },
        }, f, indent=2, ensure_ascii=False)

    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
