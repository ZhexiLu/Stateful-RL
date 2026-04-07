"""
Test predicate evaluation against actual AWM databases.

For each task, evaluates all predicates on the initial DB and verifies:
  1. Non-tautological predicates return 0 on initial DB (task not yet done)
  2. Phi(s_0) reflects the correct initial progress
  3. Predicates are executable without errors

Usage:
  cd slime/
  PYTHONPATH=../agent-world-model:. python examples/awm_esa/test_predicates.py
"""
import json
import os
import shutil
import sqlite3
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SLIME_DIR = os.path.join(SCRIPT_DIR, "..", "..")
AWM_ROOT = os.path.join(SLIME_DIR, "..", "agent-world-model")

if AWM_ROOT not in sys.path:
    sys.path.insert(0, AWM_ROOT)

from awm.tools import tools_jsonl_load, normalize_scenario_name
from awm.core.db import create_sqlite_database
from awm.core.sample import execute_sample_data

sys.path.insert(0, SCRIPT_DIR)
from generate_with_esa import _evaluate_predicates


def main():
    # Load predicates
    pred_path = os.path.join(SCRIPT_DIR, "data", "predicates.jsonl")
    all_preds = {}
    for line in open(pred_path):
        r = json.loads(line)
        if r.get("status") == "success" and r.get("predicates"):
            key = f"{r['scenario']}::{r['task_idx']}"
            all_preds[key] = r

    # Load DB schemas and sample data
    schemas = {}
    for item in tools_jsonl_load(os.path.join(AWM_ROOT, "outputs/gen_db.jsonl")):
        schemas[normalize_scenario_name(item["scenario"])] = item

    sample_data = {}
    for item in tools_jsonl_load(os.path.join(AWM_ROOT, "outputs/gen_sample.jsonl")):
        sample_data[normalize_scenario_name(item["scenario"])] = item

    # Test on a sample
    import random
    random.seed(42)
    keys = list(all_preds.keys())
    random.shuffle(keys)
    test_keys = keys[:50]

    template_dir = "/dev/shm/awm_test_predicates"
    os.makedirs(template_dir, exist_ok=True)

    stats = {
        "total_tasks": 0,
        "total_predicates": 0,
        "non_tautological": 0,
        "executable_ok": 0,
        "execute_errors": 0,
        "phi_zero_on_initial": 0,  # tasks where Phi(s_0) = 0 (all non-taut preds unsatisfied)
        "phi_nonzero_on_initial": 0,
        "phi_values": [],
    }

    print(f"Testing {len(test_keys)} tasks...\n")

    for key in test_keys:
        r = all_preds[key]
        scenario = r["scenario"]
        task_idx = r["task_idx"]
        predicates = r["predicates"]

        # Ensure template DB
        db_path = os.path.join(template_dir, f"{scenario}.db")
        if not os.path.exists(db_path):
            schema = schemas.get(scenario)
            if not schema:
                continue
            tmp_path, _, _, _ = create_sqlite_database(scenario, schema["db_schema"], template_dir)
            sd = sample_data.get(scenario)
            if sd:
                execute_sample_data(tmp_path, sd["sample_data"], scenario)
            if tmp_path != db_path:
                os.rename(tmp_path, db_path)

        # Evaluate predicates on initial DB
        phi = _evaluate_predicates(db_path, predicates)

        # Count stats
        useful = [p for p in predicates
                  if p.get("executable", True)
                  and p.get("initial_value") != p.get("expected")]

        stats["total_tasks"] += 1
        stats["total_predicates"] += len(predicates)
        stats["non_tautological"] += len(useful)

        # Test each predicate individually
        exec_ok = 0
        exec_err = 0
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            for p in predicates:
                try:
                    cursor.execute(p["sql"])
                    _ = cursor.fetchone()
                    exec_ok += 1
                except Exception as e:
                    exec_err += 1
            conn.close()
        except Exception:
            exec_err = len(predicates)

        stats["executable_ok"] += exec_ok
        stats["execute_errors"] += exec_err

        if phi == 0.0:
            stats["phi_zero_on_initial"] += 1
        else:
            stats["phi_nonzero_on_initial"] += 1
        stats["phi_values"].append(phi)

        # Print details for first 10
        if stats["total_tasks"] <= 10:
            print(f"  {key}: {len(predicates)} preds, {len(useful)} useful, "
                  f"Phi(s_0)={phi:.3f}, exec_ok={exec_ok}, exec_err={exec_err}")
            if useful:
                for p in useful[:3]:
                    print(f"    [{p['check_type']}] initial={p.get('initial_value')} "
                          f"expected={p.get('expected')} sql={p['sql'][:80]}...")

    # Summary
    print(f"\n{'='*60}")
    print(f"PREDICATE EVALUATION TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tasks tested: {stats['total_tasks']}")
    print(f"Total predicates: {stats['total_predicates']}")
    print(f"Non-tautological: {stats['non_tautological']} "
          f"({100*stats['non_tautological']/max(1,stats['total_predicates']):.1f}%)")
    print(f"Executable OK: {stats['executable_ok']} "
          f"({100*stats['executable_ok']/max(1,stats['total_predicates']):.1f}%)")
    print(f"Execute errors: {stats['execute_errors']}")
    print()
    print(f"Phi(s_0) = 0 (correct — task not yet done): {stats['phi_zero_on_initial']} "
          f"({100*stats['phi_zero_on_initial']/max(1,stats['total_tasks']):.1f}%)")
    print(f"Phi(s_0) > 0 (some preds already satisfied): {stats['phi_nonzero_on_initial']} "
          f"({100*stats['phi_nonzero_on_initial']/max(1,stats['total_tasks']):.1f}%)")

    phis = stats["phi_values"]
    if phis:
        print(f"Phi(s_0) mean: {sum(phis)/len(phis):.3f}, "
              f"median: {sorted(phis)[len(phis)//2]:.3f}, "
              f"max: {max(phis):.3f}")

    # Cleanup
    shutil.rmtree(template_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
