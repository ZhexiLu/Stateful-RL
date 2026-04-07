"""
Test the full ESA reward scoring pipeline on one scenario (10 tasks).

For each task, runs rollouts with both AWM and ESA prompts, then shows:
  - Per-turn: action, is_write, R_prog, signature, Φ
  - Predicate-anchored group advantage
  - Final composite reward

Usage:
  cd slime/
  PYTHONPATH=../agent-world-model:examples/awm_esa:. python examples/awm_esa/test_reward_scoring.py \
    --sglang_url http://127.0.0.1:8000
"""
import asyncio
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import textwrap
import time

import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SLIME_DIR = os.path.join(SCRIPT_DIR, "..", "..")
AWM_ROOT = os.path.join(SLIME_DIR, "..", "agent-world-model")
if AWM_ROOT not in sys.path:
    sys.path.insert(0, AWM_ROOT)

from awm.tools import tools_jsonl_load, normalize_scenario_name, check_mcp_server
from awm.core.agent import (
    MCPToolExecutor, get_system_prompt, parse_tool_calls,
    parse_call_tool_arguments, format_tools_for_response,
)
from awm.core.db import create_sqlite_database
from awm.core.sample import execute_sample_data
from awm.core.verifier import execute_verification_code, VerificationMode

sys.path.insert(0, SCRIPT_DIR)
from generate_with_esa import (
    get_esa_system_prompt, _evaluate_predicate_signature, _is_write_operation,
    _compute_predicate_anchored_rewards, ESA_CONFIGS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Simplified single-task rollout (no server pool, sequential)
# ═══════════════════════════════════════════════════════════════════════════════
async def run_rollout(
    scenario, task_idx, task_text, system_prompt, prompt_type,
    sglang_url, model_name, tokenizer, db_path, mcp, tools_text,
    predicates, verifiers, obs_template="think_status",
):
    """Run one rollout and return detailed per-turn info."""
    cfg = ESA_CONFIGS

    # Reset DB
    template_dir = os.path.join(cfg["db_dir"], "_templates")
    template_path = os.path.join(template_dir, f"{scenario}.db")
    shutil.copy2(template_path, db_path)
    initial_db = db_path + ".init_test"
    shutil.copy2(db_path, initial_db)

    # Build prompt
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_text},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )

    # Filter to useful predicates
    def _is_tautological(p):
        iv, ev, ct = p.get("initial_value"), p.get("expected"), p.get("check_type", "existence")
        if ct == "count_gte":
            try: return float(iv) >= float(ev) if iv is not None and ev is not None else iv == ev
            except: return iv == ev
        return iv == ev

    useful_preds = [p for p in predicates
                    if p.get("executable", True) and not _is_tautological(p)]

    response = ""
    final_answer = ""
    prev_phi = 0.0
    prev_sig = ()
    if useful_preds:
        prev_phi, prev_sig = _evaluate_predicate_signature(initial_db, predicates)

    turns = []  # detailed per-turn info
    turn_rewards = []
    turn_signatures = []
    turn_token_counts = []
    prev_was_error = False

    STATUS_RE = re.compile(r"<status>\s*(CONTINUE|VERIFY|REPLAN|STOP)\s*</status>", re.IGNORECASE)

    for turn_idx in range(cfg["max_turns"]):
        total_len = len(tokenizer.encode(prompt_text + response, add_special_tokens=False))
        if total_len + 100 >= 32768:
            break

        # Use vLLM OpenAI-compatible completions API (raw text, not chat)
        payload = {
            "model": model_name,
            "prompt": prompt_text + response,
            "temperature": 0.6,
            "max_tokens": 2048,
            "top_p": 0.95,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{sglang_url}/v1/completions", json=payload,
                                        timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    output = await resp.json()
            cur_response = output["choices"][0]["text"]
        except Exception as e:
            logger.error("API error at turn %d: %s", turn_idx + 1, e)
            break
        cur_tokens = len(tokenizer.encode(cur_response, add_special_tokens=False))

        # Parse status
        status_match = STATUS_RE.search(cur_response)
        status = status_match.group(1).upper() if status_match else ""

        # Record signature BEFORE action
        turn_signatures.append(list(prev_sig))

        # Check STOP
        if status == "STOP":
            final_answer = cur_response
            turn_info = {
                "turn": turn_idx + 1, "action": "STOP", "is_write": False,
                "status": status, "R_prog": 0.0, "Phi": prev_phi,
                "signature": list(prev_sig), "tokens": cur_tokens,
                "model_output": cur_response, "tool_response": "",
            }
            turns.append(turn_info)
            turn_rewards.append(0.0)
            turn_token_counts.append(cur_tokens)
            response += cur_response
            break

        # Parse tool call
        tcs = parse_tool_calls(cur_response)
        if not tcs:
            final_answer = cur_response
            turn_info = {
                "turn": turn_idx + 1, "action": "NO_TOOL_CALL", "is_write": False,
                "status": status, "R_prog": 0.0, "Phi": prev_phi,
                "signature": list(prev_sig), "tokens": cur_tokens,
                "model_output": cur_response, "tool_response": "",
            }
            turns.append(turn_info)
            turn_rewards.append(0.0)
            turn_token_counts.append(cur_tokens)
            response += cur_response
            break

        tc = tcs[0]
        tc_name = tc["name"]
        action_desc = tc_name

        # Execute
        is_write = False
        if tc_name == "list_tools":
            obs_text = tools_text
            action_desc = "list_tools"
        elif tc_name == "call_tool":
            try:
                tool_name, tool_args = parse_call_tool_arguments(tc["arguments"])
                action_desc = tool_name
                is_write = _is_write_operation(tool_name, tool_args)
                result = await mcp.call_tool(tool_name, tool_args)
                obs_text = result
            except Exception as e:
                obs_text = f"Error: {e}"
        else:
            obs_text = f"Error: Unknown tool"

        # Compute reward
        turn_reward = 0.0
        new_phi = prev_phi
        new_sig = prev_sig

        is_env_error = "Error:" in obs_text and any(kw in obs_text for kw in ["500", "timed out"])

        if is_env_error:
            turn_reward = 0.0
        elif is_write and useful_preds:
            new_phi, new_sig = _evaluate_predicate_signature(db_path, predicates)
            turn_reward = new_phi - prev_phi
            prev_was_error = (turn_reward < 0) or ("Error" in obs_text)
            prev_phi = new_phi
            prev_sig = new_sig
        elif not is_write and prev_was_error:
            turn_reward = cfg["gamma_verify"]
            prev_was_error = False
        else:
            prev_was_error = "Error" in obs_text

        if status == "VERIFY" and is_write:
            turn_reward += cfg["verify_violation_penalty"]

        turn_info = {
            "turn": turn_idx + 1, "action": action_desc, "is_write": is_write,
            "status": status, "R_prog": round(turn_reward, 4),
            "Phi": round(new_phi if is_write else prev_phi, 4),
            "signature": list(new_sig if is_write else prev_sig),
            "tokens": cur_tokens,
            "model_output": cur_response,
            "tool_response": obs_text,
        }
        turns.append(turn_info)
        turn_rewards.append(turn_reward)
        turn_token_counts.append(cur_tokens)

        response += cur_response
        # Append observation — template varies by prompt type and config
        if prompt_type == "awm":
            next_obs = (
                f"<|im_start|>user\n<tool_response>\n{obs_text}\n</tool_response><|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
        elif obs_template == "think_status":
            next_obs = (
                f"<|im_start|>user\n<tool_response>\n{obs_text}\n</tool_response><|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n<status>"
            )
        elif obs_template == "think_only":
            next_obs = (
                f"<|im_start|>user\n<tool_response>\n{obs_text}\n</tool_response><|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n"
            )
        else:  # bare
            next_obs = (
                f"<|im_start|>user\n<tool_response>\n{obs_text}\n</tool_response><|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        response += next_obs

    # Outcome reward
    key = f"{scenario}::{task_idx}"
    verifier = verifiers.get(key)
    outcome = 0.0
    if verifier:
        code = verifier.get("verification", {}).get("code", "")
        if code:
            func_name = "verify_task_completion"
            for line in code.split("\n"):
                line = line.strip()
                if line.startswith("def verify_") and "(" in line:
                    func_name = line.split("(")[0].replace("def ", "").strip()
                    break
            mode = VerificationMode.code if "final_answer" in code else VerificationMode.sql
            try:
                result = execute_verification_code(
                    python_code=code, function_name=func_name,
                    initial_db_path=initial_db, mode=mode,
                    final_db_path=db_path, final_answer=final_answer,
                )
                if result.get("execution_status") == "success":
                    inner = result.get("result", {})
                    if isinstance(inner, dict) and inner.get("result", "").lower() == "complete":
                        outcome = 1.0
            except Exception:
                pass

    # Cleanup
    try:
        os.remove(initial_db)
    except OSError:
        pass

    # Compute composite (turn-level return-to-go from step 0)
    eta = cfg["eta"]
    lam = cfg["lambda_prog"]
    if turn_rewards:
        T = len(turn_rewards)
        step_r = [lam * r for r in turn_rewards]
        step_r[-1] += outcome
        rtg = [0.0] * T
        rtg[T-1] = step_r[T-1]
        for t in range(T-2, -1, -1):
            rtg[t] = step_r[t] + eta * rtg[t+1]
        composite = rtg[0]
    else:
        composite = outcome

    return {
        "scenario": scenario, "task_idx": task_idx, "task": task_text,
        "prompt_type": prompt_type, "turns": turns,
        "turn_rewards": turn_rewards, "turn_signatures": turn_signatures,
        "turn_token_counts": turn_token_counts,
        "outcome_reward": outcome, "composite_reward": round(composite, 4),
        "num_useful_predicates": len(useful_preds),
    }


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sglang_url", default="http://127.0.0.1:8000")
    parser.add_argument("--scenario", default="e_commerce_33")
    parser.add_argument("--port", type=int, default=9300)
    parser.add_argument("--output", default=os.path.join(SCRIPT_DIR, "data", "test_reward_scoring.json"))
    parser.add_argument("--obs_template", default="think_status", choices=["think_status", "think_only", "bare"],
                        help="Observation template: think_status=<think>\\n<status>, think_only=<think>\\n, bare=nothing")
    args = parser.parse_args()

    # Load data
    schemas, sample_data, verifiers, envs_data, tasks_map, all_preds = {}, {}, {}, {}, {}, {}
    for item in tools_jsonl_load(os.path.join(AWM_ROOT, "outputs/gen_db.jsonl")):
        schemas[normalize_scenario_name(item["scenario"])] = item
    for item in tools_jsonl_load(os.path.join(AWM_ROOT, "outputs/gen_sample.jsonl")):
        sample_data[normalize_scenario_name(item["scenario"])] = item
    for item in tools_jsonl_load(os.path.join(AWM_ROOT, "outputs/gen_verifier.pure_code.jsonl")):
        s = normalize_scenario_name(item["scenario"])
        verifiers[f"{s}::{item['task_idx']}"] = item
    for item in tools_jsonl_load(os.path.join(AWM_ROOT, "outputs/gen_envs.jsonl")):
        envs_data[normalize_scenario_name(item["scenario"])] = item
    for item in tools_jsonl_load(os.path.join(AWM_ROOT, "outputs/gen_tasks.jsonl")):
        s = normalize_scenario_name(item["scenario"])
        tasks_map[s] = item.get("tasks", [])
    pred_path = os.path.join(SCRIPT_DIR, "data", "predicates.jsonl")
    for line in open(pred_path):
        r = json.loads(line)
        if r.get("status") == "success":
            all_preds[f"{r['scenario']}::{r['task_idx']}"] = r.get("predicates", [])

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(SLIME_DIR, "..", "models/Qwen3-4B"), trust_remote_code=True)

    scenario = normalize_scenario_name(args.scenario)
    task_list = tasks_map.get(scenario, [])
    logger.info("Scenario: %s, Tasks: %d", scenario, len(task_list))

    # Prepare DB
    db_dir = ESA_CONFIGS["db_dir"]
    template_dir = os.path.join(db_dir, "_templates")
    os.makedirs(template_dir, exist_ok=True)
    template_path = os.path.join(template_dir, f"{scenario}.db")
    if not os.path.exists(template_path):
        schema = schemas[scenario]
        tmp, _, _, _ = create_sqlite_database(scenario, schema["db_schema"], template_dir)
        sd = sample_data.get(scenario)
        if sd:
            execute_sample_data(tmp, sd["sample_data"], scenario)
        if tmp != template_path:
            os.rename(tmp, template_path)

    db_path = os.path.join(db_dir, f"test_scoring_{scenario}.db")

    # Start MCP server
    shutil.copy2(template_path, db_path)
    env_item = envs_data[scenario]
    code = env_item["full_code"]
    new_lines = ["import warnings", 'warnings.filterwarnings("ignore", category=DeprecationWarning)',
                 "from sqlalchemy.pool import NullPool"]
    for line in code.split("\n"):
        if "create_engine(" in line:
            left = line.split("create_engine(")[0]
            line = f"{left}create_engine('sqlite:///{db_path}', connect_args={{'check_same_thread': False}}, poolclass=NullPool)"
        if "uvicorn.run(app" in line:
            setup = textwrap.indent(textwrap.dedent("""\
                from fastapi_mcp import FastApiMCP
                mcp = FastApiMCP(app)
                mcp.mount_http()
            """), "    ")
            new_lines.extend(setup.rstrip().split("\n"))
            line = f"    uvicorn.run(app, host='127.0.0.1', port={args.port})"
        new_lines.append(line)
    script_dir = os.path.join("/dev/shm", f"awm_test_scripts_{os.getuid()}")
    os.makedirs(script_dir, exist_ok=True)
    script_path = os.path.join(script_dir, f"test_{scenario}.py")
    with open(script_path, "w") as f:
        f.write("\n".join(new_lines))

    subprocess.run(["fuser", "-k", f"{args.port}/tcp"], capture_output=True, timeout=3)
    await asyncio.sleep(0.5)

    env = os.environ.copy()
    for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
        env[k] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    proc = subprocess.Popen([sys.executable, script_path],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)

    mcp_url = f"http://127.0.0.1:{args.port}/mcp"
    deadline = time.time() + 60
    mcp = None
    while time.time() < deadline:
        try:
            running, tc, _, _ = await check_mcp_server(url=mcp_url, timeout=2.0)
            if running and tc > 0:
                mcp = MCPToolExecutor(mcp_url, timeout=10.0)
                await mcp.__aenter__()
                tools = await mcp.list_tools()
                tools_text = format_tools_for_response(tools)
                break
        except Exception:
            pass
        await asyncio.sleep(0.5)

    if not mcp:
        proc.kill()
        logger.error("MCP server failed to start")
        return

    awm_prompt = get_system_prompt()
    esa_prompt = get_esa_system_prompt()

    # Detect model name from vLLM
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{args.sglang_url}/v1/models") as resp:
            models = await resp.json()
    model_name = models["data"][0]["id"]
    logger.info("Using model: %s", model_name)

    # Run rollouts
    all_results = []
    for task_idx in range(min(10, len(task_list))):
        task_text = task_list[task_idx]
        preds = all_preds.get(f"{scenario}::{task_idx}", [])

        for pt, prompt in [("awm", awm_prompt), ("esa", esa_prompt)]:
            logger.info("Running %s::%d [%s]...", scenario, task_idx, pt)
            result = await run_rollout(
                scenario, task_idx, task_text, prompt, pt,
                args.sglang_url, model_name, tokenizer, db_path, mcp, tools_text,
                preds, verifiers, obs_template=args.obs_template,
            )
            all_results.append(result)

            # Print per-turn details
            print(f"\n{'='*80}")
            print(f"Task {task_idx} [{pt.upper()}]: {task_text[:80]}...")
            print(f"Useful predicates: {result['num_useful_predicates']}")
            print(f"{'─'*80}")
            for t in result["turns"]:
                sig_str = "".join(str(b) for b in t["signature"]) if t["signature"] else "-"
                write_mark = "W" if t["is_write"] else "R"
                status_mark = f"[{t['status']}]" if t["status"] else ""
                print(f"  Turn {t['turn']:2d} {write_mark} {status_mark:10s} "
                      f"Φ={t['Phi']:.2f} R_prog={t['R_prog']:+.3f} "
                      f"σ={sig_str:8s} | {t['action']}")
            print(f"{'─'*80}")
            print(f"  Outcome: {result['outcome_reward']}  "
                  f"Composite: {result['composite_reward']}")

    # Now simulate group reward (predicate-anchored advantage)
    # Group by task_idx
    print(f"\n{'='*80}")
    print("PREDICATE-ANCHORED GROUP ADVANTAGE")
    print(f"{'='*80}")

    from collections import defaultdict

    for task_idx in range(min(10, len(task_list))):
        task_results = [r for r in all_results if r["task_idx"] == task_idx]
        if len(task_results) < 2:
            continue

        # Simulate what reward_func would do with a group
        # Build fake Sample objects
        class FakeSample:
            def __init__(self, r):
                self.metadata = {
                    "turn_rewards": r["turn_rewards"],
                    "turn_signatures": r["turn_signatures"],
                    "turn_token_counts": r["turn_token_counts"],
                    "outcome_reward": r["outcome_reward"],
                }
                self.reward = r["composite_reward"]

        fake_samples = [FakeSample(r) for r in task_results]
        anchored_rewards = _compute_predicate_anchored_rewards(fake_samples)

        print(f"\nTask {task_idx}: {task_list[task_idx][:60]}...")
        for r, ar in zip(task_results, anchored_rewards):
            print(f"  [{r['prompt_type'].upper()}] outcome={r['outcome_reward']:.1f} "
                  f"composite={r['composite_reward']:.3f} → anchored={ar:.3f}")

    # Cleanup
    await mcp.__aexit__(None, None, None)
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
    subprocess.run(["fuser", "-k", f"{args.port}/tcp"], capture_output=True, timeout=3)

    # Save results
    out_path = args.output
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    os.environ["OTEL_SDK_DISABLED"] = "true"
    asyncio.run(main())
