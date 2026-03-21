"""
End-to-end rollout test with mock LLM.

Tests the full rollout logic WITHOUT a real LLM server:
  - AWM env lifecycle (DB reset, MCP server, tool calls, verification)
  - Token management (prompt tokens, response tokens, observation tokens)
  - Loss mask correctness (1 for model tokens, 0 for env tokens)
  - Reward computation via AWM verification

Uses a scripted "agent" that:
  1. Calls list_tools
  2. Calls the first available read-like tool
  3. Gives a final answer
"""
import os
import sys
import json
import re
import time
import asyncio

AWM_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent-world-model")
AWMGRPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AWM_ROOT)
sys.path.insert(0, AWMGRPO_ROOT)

from awm.tools import tools_jsonl_load, normalize_scenario_name
from env_awm import AWMEnv, preload_awm_data, _DATA_CACHE


def find_test_scenario():
    """Find a scenario with envs, tasks, and verifier."""
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
        for idx, task in enumerate(tasks):
            if f"{scenario}::{idx}" in verifier_keys:
                db_path = os.path.join(AWM_ROOT, "outputs/databases", f"{scenario}.db")
                if os.path.exists(db_path):
                    return scenario, task, idx
    raise RuntimeError("No valid test scenario found")


async def simulate_rollout(scenario: str, task: str, task_idx: int):
    """
    Simulate a full multi-turn rollout with scripted agent responses.

    This mimics what rollout.py does but without SGLang.
    """
    from transformers import AutoTokenizer

    print(f"\n{'='*60}")
    print(f"SIMULATED ROLLOUT: {scenario}[{task_idx}]")
    print(f"Task: {task[:120]}...")
    print(f"{'='*60}")

    # Load tokenizer (Qwen3-4B or fallback)
    model_name = "Qwen/Qwen3-4B"
    print(f"\nLoading tokenizer: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        print("  Qwen3-4B not available, trying Qwen2.5-0.5B...")
        model_name = "Qwen/Qwen2.5-0.5B"
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            print(f"  No tokenizer available ({e}), using mock tokenizer")
            tokenizer = None

    # Build prompt (same as data_prep.py)
    from awm.core.agent import get_system_prompt
    system_prompt = get_system_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    # Tokenize prompt
    if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    else:
        prompt_str = f"System: {system_prompt}\nUser: {task}\nAssistant:"
        prompt_ids = list(range(len(prompt_str.split())))  # mock token ids

    print(f"Prompt tokens: {len(prompt_ids)}")

    # Initialize tracking (mimics rollout.py)
    all_tokens = list(prompt_ids)
    response_tokens = []
    loss_mask = []
    log_probs = []

    # Create and start env
    env = AWMEnv(
        scenario=scenario,
        task=task,
        task_idx=task_idx,
        max_iterations=5,
        server_startup_timeout=20.0,
    )

    t0 = time.time()
    await env.reset()
    print(f"Env reset: {time.time()-t0:.1f}s")

    # === Turn 1: list_tools ===
    print(f"\n--- Turn 1: list_tools ---")
    agent_response_1 = (
        'I need to first see what tools are available.\n'
        '<tool_call>\n{"name": "list_tools", "arguments": null}\n</tool_call>'
    )
    # Tokenize agent response
    if tokenizer:
        resp_ids = tokenizer.encode(agent_response_1, add_special_tokens=False)
    else:
        resp_ids = list(range(len(agent_response_1.split())))

    # Append model tokens (loss_mask=1)
    all_tokens.extend(resp_ids)
    response_tokens.extend(resp_ids)
    loss_mask.extend([1] * len(resp_ids))
    log_probs.extend([-0.5] * len(resp_ids))  # mock log probs
    print(f"  Model tokens: {len(resp_ids)} (loss_mask=1)")

    obs, done, info = await env.step(agent_response_1)
    assert not done
    tools_text = obs["obs_str"]
    print(f"  Env response: {len(tools_text)} chars")

    # Tokenize observation
    if tokenizer:
        obs_message = {"role": "tool", "content": tools_text}
        obs_str = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": agent_response_1}, obs_message],
            tokenize=False, add_generation_prompt=True,
        )
        prev_str = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": agent_response_1}],
            tokenize=False, add_generation_prompt=False,
        )
        obs_ids = tokenizer.encode(obs_str, add_special_tokens=False)[
            len(tokenizer.encode(prev_str, add_special_tokens=False)):
        ]
    else:
        obs_ids = list(range(50))  # mock

    # Append observation tokens (loss_mask=0)
    all_tokens.extend(obs_ids)
    response_tokens.extend(obs_ids)
    loss_mask.extend([0] * len(obs_ids))
    log_probs.extend([0.0] * len(obs_ids))
    print(f"  Obs tokens: {len(obs_ids)} (loss_mask=0)")

    # === Turn 2: call a tool ===
    tool_names = re.findall(r'\d+\.\s+(mcp_tool_\w+)', tools_text)
    # Find a "get" or "list" or "search" tool
    read_tool = None
    for t in tool_names:
        if any(kw in t.lower() for kw in ['get', 'list', 'search', 'find', 'fetch']):
            read_tool = t
            break
    if not read_tool and tool_names:
        read_tool = tool_names[0]

    if read_tool:
        print(f"\n--- Turn 2: call_tool({read_tool}) ---")
        agent_response_2 = (
            f'Let me search for information.\n'
            f'<tool_call>\n{{"name": "call_tool", "arguments": {{"tool_name": "{read_tool}", "arguments": "{{}}"}}}}\n</tool_call>'
        )

        if tokenizer:
            resp_ids_2 = tokenizer.encode(agent_response_2, add_special_tokens=False)
        else:
            resp_ids_2 = list(range(20))

        all_tokens.extend(resp_ids_2)
        response_tokens.extend(resp_ids_2)
        loss_mask.extend([1] * len(resp_ids_2))
        log_probs.extend([-0.8] * len(resp_ids_2))
        print(f"  Model tokens: {len(resp_ids_2)} (loss_mask=1)")

        obs2, done2, info2 = await env.step(agent_response_2)
        tool_result = obs2["obs_str"]
        print(f"  Tool result: {tool_result[:200]}...")

        if not done2:
            # Tokenize observation (simplified)
            if tokenizer:
                obs_ids_2 = tokenizer.encode(tool_result[:500], add_special_tokens=False)[:200]
            else:
                obs_ids_2 = list(range(30))

            all_tokens.extend(obs_ids_2)
            response_tokens.extend(obs_ids_2)
            loss_mask.extend([0] * len(obs_ids_2))
            log_probs.extend([0.0] * len(obs_ids_2))
            print(f"  Obs tokens: {len(obs_ids_2)} (loss_mask=0)")

    # === Turn 3: final answer ===
    print(f"\n--- Turn 3: final answer ---")
    agent_response_3 = "Based on my analysis, here is the result."

    if tokenizer:
        resp_ids_3 = tokenizer.encode(agent_response_3, add_special_tokens=False)
    else:
        resp_ids_3 = list(range(10))

    all_tokens.extend(resp_ids_3)
    response_tokens.extend(resp_ids_3)
    loss_mask.extend([1] * len(resp_ids_3))
    log_probs.extend([-1.0] * len(resp_ids_3))
    print(f"  Model tokens: {len(resp_ids_3)} (loss_mask=1)")

    obs3, done3, info3 = await env.step(agent_response_3)
    assert done3, f"Should be done after no tool call, got done={done3}"
    print(f"  Done: {done3}, reason: {info3.get('done_reason')}")

    # === Compute reward ===
    reward = env.compute_reward()
    print(f"\n--- Reward ---")
    print(f"  Reward: {reward}")
    print(f"  Verify: {env._verify_result}")

    # === Validate data structures ===
    print(f"\n--- Validation ---")
    total_response = len(response_tokens)
    model_tokens = sum(loss_mask)
    env_tokens = total_response - model_tokens

    print(f"  Total tokens: {len(all_tokens)}")
    print(f"  Prompt tokens: {len(prompt_ids)}")
    print(f"  Response tokens: {total_response}")
    print(f"    Model tokens (loss=1): {model_tokens}")
    print(f"    Env tokens (loss=0): {env_tokens}")
    print(f"  Loss mask length: {len(loss_mask)}")
    print(f"  Log probs length: {len(log_probs)}")

    assert len(loss_mask) == total_response, \
        f"loss_mask ({len(loss_mask)}) != response_tokens ({total_response})"
    assert len(log_probs) == total_response, \
        f"log_probs ({len(log_probs)}) != response_tokens ({total_response})"
    assert len(all_tokens) == len(prompt_ids) + total_response, \
        f"all_tokens ({len(all_tokens)}) != prompt ({len(prompt_ids)}) + response ({total_response})"
    assert model_tokens > 0, "Should have some model tokens"
    assert env_tokens > 0, "Should have some env tokens"
    assert reward is not None, "Reward should be computed"

    print(f"  ALL ASSERTIONS PASSED")

    # Cleanup
    await env.close()
    print(f"  Env closed")

    return {
        "total_tokens": len(all_tokens),
        "prompt_tokens": len(prompt_ids),
        "response_tokens": total_response,
        "model_tokens": model_tokens,
        "env_tokens": env_tokens,
        "reward": reward,
        "num_turns": env._iteration,
    }


def test_concurrent_envs(scenarios_data):
    """Test two envs running concurrently (simulates parallel rollout)."""
    import threading

    print(f"\n{'='*60}")
    print("TEST: Concurrent envs (2 parallel)")
    print(f"{'='*60}")

    results = [None, None]
    errors = [None, None]

    def run_env(idx, scenario, task, task_idx):
        try:
            results[idx] = asyncio.run(simulate_rollout(scenario, task, task_idx))
        except Exception as e:
            errors[idx] = str(e)
            import traceback
            traceback.print_exc()

    s1 = scenarios_data[0]
    s2 = scenarios_data[1] if len(scenarios_data) > 1 else scenarios_data[0]

    t1 = threading.Thread(target=run_env, args=(0, s1[0], s1[1], s1[2]))
    t2 = threading.Thread(target=run_env, args=(1, s2[0], s2[1], s2[2]))

    t0 = time.time()
    t1.start()
    t2.start()
    t1.join(timeout=120)
    t2.join(timeout=120)

    print(f"\nConcurrent test completed in {time.time()-t0:.1f}s")
    for i in range(2):
        if errors[i]:
            print(f"  Env {i}: FAILED - {errors[i]}")
        else:
            r = results[i]
            print(f"  Env {i}: OK (tokens={r['total_tokens']}, reward={r['reward']}, turns={r['num_turns']})")

    assert all(e is None for e in errors), f"Some envs failed: {errors}"
    print("  CONCURRENT TEST PASSED")


def find_multiple_scenarios(n=2):
    """Find n different test scenarios."""
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

    results = []
    for scenario, tasks in tasks_by_scenario.items():
        if len(results) >= n:
            break
        for idx, task in enumerate(tasks):
            if f"{scenario}::{idx}" in verifier_keys:
                db_path = os.path.join(AWM_ROOT, "outputs/databases", f"{scenario}.db")
                if os.path.exists(db_path):
                    results.append((scenario, task, idx))
                    break
    return results


if __name__ == "__main__":
    print("Preloading AWM data...")
    preload_awm_data(
        db_schema_path=os.path.join(AWM_ROOT, "outputs/gen_db.jsonl"),
        sample_path=os.path.join(AWM_ROOT, "outputs/gen_sample.jsonl"),
        verifier_path=os.path.join(AWM_ROOT, "outputs/gen_verifier.pure_code.jsonl"),
        envs_path=os.path.join(AWM_ROOT, "outputs/gen_envs.jsonl"),
        db_dir=os.path.join(AWM_ROOT, "outputs/databases"),
    )
    print(f"Loaded {len(_DATA_CACHE.db_schemas)} schemas, {len(_DATA_CACHE.verifiers)} verifiers\n")

    # Single rollout test
    scenarios = find_multiple_scenarios(2)
    result = asyncio.run(simulate_rollout(*scenarios[0]))
    print(f"\nSingle rollout result: {json.dumps(result, indent=2)}")

    # Concurrent test
    if len(scenarios) >= 2:
        test_concurrent_envs(scenarios)

    print(f"\n{'='*60}")
    print("ALL ROLLOUT TESTS PASSED")
    print(f"{'='*60}")
