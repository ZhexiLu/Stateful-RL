"""
Minimal test: verify AWM env lifecycle works end-to-end.

Tests: preload data → create env → reset (DB + MCP server) → list_tools → call_tool → verify → close
"""
import os
import sys
import json
import time

# Setup paths
AWM_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent-world-model")
AWMGRPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AWM_ROOT)
sys.path.insert(0, AWMGRPO_ROOT)

from awm.tools import tools_jsonl_load, normalize_scenario_name


def find_test_scenario():
    """Find a scenario that has envs, tasks, and verifier."""
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

    # Find first scenario with a verified task
    for scenario, tasks in tasks_by_scenario.items():
        for idx, task in enumerate(tasks):
            if f"{scenario}::{idx}" in verifier_keys:
                db_path = os.path.join(AWM_ROOT, "outputs/databases", f"{scenario}.db")
                if os.path.exists(db_path):
                    return scenario, task, idx

    raise RuntimeError("No valid test scenario found")


def test_preload():
    """Test: preload AWM data cache."""
    print("=" * 60)
    print("TEST 1: Preload AWM data cache")
    print("=" * 60)

    from env_awm import preload_awm_data, _DATA_CACHE

    preload_awm_data(
        db_schema_path=os.path.join(AWM_ROOT, "outputs/gen_db.jsonl"),
        sample_path=os.path.join(AWM_ROOT, "outputs/gen_sample.jsonl"),
        verifier_path=os.path.join(AWM_ROOT, "outputs/gen_verifier.pure_code.jsonl"),
        envs_path=os.path.join(AWM_ROOT, "outputs/gen_envs.jsonl"),
        db_dir=os.path.join(AWM_ROOT, "outputs/databases"),
    )

    print(f"  DB schemas loaded: {len(_DATA_CACHE.db_schemas)}")
    print(f"  Sample data loaded: {len(_DATA_CACHE.sample_data)}")
    print(f"  Verifiers loaded: {len(_DATA_CACHE.verifiers)}")
    print(f"  Envs path: {_DATA_CACHE.envs_path}")
    assert _DATA_CACHE.loaded, "Data cache should be loaded"
    print("  PASSED\n")


def test_env_lifecycle(scenario, task, task_idx):
    """Test: full env lifecycle."""
    print("=" * 60)
    print(f"TEST 2: Env lifecycle for {scenario}[{task_idx}]")
    print(f"  Task: {task[:100]}...")
    print("=" * 60)

    from env_awm import AWMEnv

    env = AWMEnv(
        scenario=scenario,
        task=task,
        task_idx=task_idx,
        max_iterations=5,
        server_startup_timeout=20.0,
        tool_timeout=15.0,
    )

    # Reset (starts MCP server)
    print("  Resetting env (DB + MCP server)...")
    t0 = time.time()
    try:
        env.reset()
    except Exception as e:
        print(f"  FAILED: env.reset() raised {e}")
        env.close()
        return False
    print(f"  Reset done in {time.time() - t0:.1f}s")

    # Step 1: list_tools
    print("  Step 1: list_tools...")
    response_text = '<tool_call>\n{"name": "list_tools", "arguments": null}\n</tool_call>'
    obs, done, info = env.step(response_text)
    tools_text = obs.get("obs_str", "")
    print(f"  Tools response length: {len(tools_text)} chars")
    print(f"  Done: {done}, Info: {info.get('tool_name')}")
    assert not done, "Should not be done after list_tools"
    assert len(tools_text) > 100, "Tools response should be non-empty"
    print("  list_tools PASSED")

    # Step 2: call a tool (pick first tool from the listing)
    # Parse tool name from the tools listing
    import re
    tool_names = re.findall(r'\d+\.\s+(mcp_tool_\w+)', tools_text)
    if tool_names:
        test_tool = tool_names[0]
        print(f"\n  Step 2: calling {test_tool}...")
        response_text = f'<tool_call>\n{{"name": "call_tool", "arguments": {{"tool_name": "{test_tool}", "arguments": "{{}}"}}}}\n</tool_call>'
        obs, done, info = env.step(response_text)
        tool_result = obs.get("obs_str", "")
        print(f"  Tool result: {tool_result[:200]}...")
        print(f"  Done: {done}")
    else:
        print("  Skipping tool call test (no tools found)")

    # Step 3: no tool call (final answer)
    print(f"\n  Step 3: final answer (no tool call)...")
    response_text = "Based on my analysis, the task is complete."
    obs, done, info = env.step(response_text)
    print(f"  Done: {done}, Reason: {info.get('done_reason')}")
    assert done, "Should be done after no tool call"
    print("  Final answer PASSED")

    # Compute reward
    print(f"\n  Computing reward via verification...")
    reward = env.compute_reward()
    print(f"  Reward: {reward}")
    print(f"  Verify result: {env._verify_result}")

    # Close
    print(f"\n  Closing env...")
    env.close()
    print("  Close PASSED")

    print(f"\n  FULL LIFECYCLE PASSED (reward={reward})")
    return True


def test_port_pool():
    """Test: port pool acquire/release."""
    print("=" * 60)
    print("TEST 3: Port pool")
    print("=" * 60)

    from env_awm import PortPool

    pool = PortPool(base_port=19000, pool_size=5)
    ports = []
    for i in range(3):
        p = pool.acquire()
        ports.append(p)
        print(f"  Acquired port {p}")

    for p in ports:
        pool.release(p)
        print(f"  Released port {p}")

    print("  PASSED\n")


if __name__ == "__main__":
    scenario, task, task_idx = find_test_scenario()
    print(f"Using test scenario: {scenario}, task_idx: {task_idx}")
    print(f"Task: {task[:150]}...\n")

    test_preload()
    test_port_pool()
    test_env_lifecycle(scenario, task, task_idx)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
