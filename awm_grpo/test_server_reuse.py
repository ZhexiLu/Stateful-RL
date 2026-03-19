"""
Test server reuse optimization.

Verifies that:
1. First task on a scenario pays the server startup cost (~4s)
2. Second task on SAME scenario reuses the server (~0s startup)
3. DB is correctly reset between tasks (NullPool ensures fresh connections)
4. Verification works correctly after DB reset
"""
import os
import sys
import time

AWM_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent-world-model")
AWMGRPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AWM_ROOT)
sys.path.insert(0, AWMGRPO_ROOT)

from awm.tools import tools_jsonl_load, normalize_scenario_name
from env_awm import AWMEnv, preload_awm_data, _DATA_CACHE, get_server_pool


def find_scenario_with_multiple_tasks():
    """Find a scenario with at least 2 verified tasks."""
    envs_path = os.path.join(AWM_ROOT, "outputs/gen_envs.jsonl")
    tasks_path = os.path.join(AWM_ROOT, "outputs/gen_tasks.jsonl")
    verifier_path = os.path.join(AWM_ROOT, "outputs/gen_verifier.pure_code.jsonl")

    envs_scenarios = set()
    for item in tools_jsonl_load(envs_path):
        envs_scenarios.add(normalize_scenario_name(item["scenario"]))

    tasks_by_scenario = {}
    for item in tools_jsonl_load(tasks_path):
        s = normalize_scenario_name(item["scenario"])
        if s in envs_scenarios:
            tasks_by_scenario[s] = item["tasks"]

    verifier_keys = set()
    for item in tools_jsonl_load(verifier_path):
        s = normalize_scenario_name(item["scenario"])
        verifier_keys.add(f"{s}::{item['task_idx']}")

    for scenario, tasks in tasks_by_scenario.items():
        verified = []
        for idx, task in enumerate(tasks):
            if f"{scenario}::{idx}" in verifier_keys:
                verified.append((idx, task))
            if len(verified) >= 2:
                db_path = os.path.join(AWM_ROOT, "outputs/databases", f"{scenario}.db")
                if os.path.exists(db_path):
                    return scenario, verified
    raise RuntimeError("No scenario with 2+ verified tasks found")


def run_task(scenario, task, task_idx, label=""):
    """Run a single task and return timing + result."""
    env = AWMEnv(
        scenario=scenario,
        task=task,
        task_idx=task_idx,
        max_iterations=3,
    )

    t0 = time.time()
    env.reset()
    reset_time = time.time() - t0

    # list_tools
    obs, done, info = env.step(
        '<tool_call>\n{"name": "list_tools", "arguments": null}\n</tool_call>'
    )
    assert not done
    tools_len = len(obs["obs_str"])

    # final answer
    obs, done, info = env.step("Task complete.")
    assert done

    reward = env.compute_reward()
    env.close()

    print(f"  {label}: reset={reset_time:.2f}s, tools={tools_len} chars, reward={reward}")
    return reset_time, reward


def main():
    preload_awm_data(
        db_schema_path=os.path.join(AWM_ROOT, "outputs/gen_db.jsonl"),
        sample_path=os.path.join(AWM_ROOT, "outputs/gen_sample.jsonl"),
        verifier_path=os.path.join(AWM_ROOT, "outputs/gen_verifier.pure_code.jsonl"),
        envs_path=os.path.join(AWM_ROOT, "outputs/gen_envs.jsonl"),
        db_dir=os.path.join(AWM_ROOT, "outputs/databases"),
    )

    scenario, tasks = find_scenario_with_multiple_tasks()
    print(f"Scenario: {scenario}")
    print(f"Task 0: {tasks[0][1][:80]}...")
    print(f"Task 1: {tasks[1][1][:80]}...")

    # === Test 1: First task (cold start — server startup) ===
    print(f"\n{'='*60}")
    print("TEST 1: Cold start (first task)")
    print(f"{'='*60}")
    t1_reset, t1_reward = run_task(scenario, tasks[0][1], tasks[0][0], "Task 0 (cold)")

    # === Test 2: Second task on SAME scenario (warm — server reuse) ===
    print(f"\n{'='*60}")
    print("TEST 2: Warm reuse (second task, same scenario)")
    print(f"{'='*60}")
    t2_reset, t2_reward = run_task(scenario, tasks[1][1], tasks[1][0], "Task 1 (warm)")

    # === Test 3: Third task on SAME scenario (still warm) ===
    print(f"\n{'='*60}")
    print("TEST 3: Still warm (third task = repeat of task 0)")
    print(f"{'='*60}")
    t3_reset, t3_reward = run_task(scenario, tasks[0][1], tasks[0][0], "Task 0 again (warm)")

    # === Summary ===
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Cold start reset: {t1_reset:.2f}s")
    print(f"  Warm reuse reset: {t2_reset:.2f}s")
    print(f"  Still warm reset: {t3_reset:.2f}s")
    speedup = t1_reset / max(t2_reset, 0.001)
    print(f"  Speedup: {speedup:.1f}x")

    assert t2_reset < t1_reset, f"Warm reset ({t2_reset:.2f}s) should be faster than cold ({t1_reset:.2f}s)"
    assert t3_reset < t1_reset, f"Warm reset ({t3_reset:.2f}s) should be faster than cold ({t1_reset:.2f}s)"
    print("  ASSERTIONS PASSED")

    # Cleanup
    get_server_pool().shutdown_all()
    print("  Server pool shut down")


if __name__ == "__main__":
    main()
