"""
Evaluate Qwen3-4B on AWM tasks with two different system prompts:
  1. AWM official prompt (baseline)
  2. ESA prompt (with execution-status control block)

Records per-task rewards for both prompts to compare reward distributions.

Usage:
  python examples/awm_esa/eval_prompt_comparison.py \
    --num_tasks 200 --repeats 8 --num_gpus 8
"""
import argparse
import asyncio
import json
import logging
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Suppress noisy OpenTelemetry context errors from MCP SDK async usage
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)
import os as _os
_os.environ.setdefault("OTEL_SDK_DISABLED", "true")

# ── AWM imports ──
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

# ── ESA system prompt ──
sys.path.insert(0, SCRIPT_DIR)
from generate_with_esa import get_esa_system_prompt

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════
EVAL_CONFIG = {
    "max_turns": 15,
    "tool_timeout": 10.0,
    "server_startup_timeout": 60.0,
    "db_dir": "/dev/shm/awm_eval_databases",
    "db_schema_path": os.path.join(AWM_ROOT, "outputs/gen_db.jsonl"),
    "sample_path": os.path.join(AWM_ROOT, "outputs/gen_sample.jsonl"),
    "verifier_path": os.path.join(AWM_ROOT, "outputs/gen_verifier.pure_code.jsonl"),
    "envs_path": os.path.join(AWM_ROOT, "outputs/gen_envs.jsonl"),
    "tasks_path": os.path.join(AWM_ROOT, "outputs/gen_tasks.jsonl"),
    "temperature": 0.6,
    "top_k": 20,
    "top_p": 0.95,
    "max_new_tokens": 2048,
    "max_context_len": 32768,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════
def load_data():
    cfg = EVAL_CONFIG
    db_schemas, sample_data, verifiers, envs_data, tasks = {}, {}, {}, {}, {}

    for item in tools_jsonl_load(cfg["db_schema_path"]):
        db_schemas[normalize_scenario_name(item["scenario"])] = item
    for item in tools_jsonl_load(cfg["sample_path"]):
        sample_data[normalize_scenario_name(item["scenario"])] = item
    for item in tools_jsonl_load(cfg["verifier_path"]):
        s = normalize_scenario_name(item["scenario"])
        verifiers[f"{s}::{item['task_idx']}"] = item
    for item in tools_jsonl_load(cfg["envs_path"]):
        envs_data[normalize_scenario_name(item["scenario"])] = item
    for item in tools_jsonl_load(cfg["tasks_path"]):
        s = normalize_scenario_name(item["scenario"])
        tasks[s] = item.get("tasks", [])

    logger.info("Loaded: %d schemas, %d verifiers, %d envs, %d scenario tasks",
                len(db_schemas), len(verifiers), len(envs_data), len(tasks))
    return db_schemas, sample_data, verifiers, envs_data, tasks


def select_tasks(verifiers, tasks, num_tasks, seed=42):
    """Select num_tasks verified tasks randomly."""
    all_keys = []
    for key in verifiers:
        scenario, task_idx_str = key.split("::")
        task_idx = int(task_idx_str)
        task_list = tasks.get(scenario, [])
        if task_idx < len(task_list):
            all_keys.append((scenario, task_idx, task_list[task_idx]))

    random.seed(seed)
    random.shuffle(all_keys)
    selected = all_keys[:num_tasks]
    logger.info("Selected %d tasks from %d available", len(selected), len(all_keys))
    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# Environment management
# ═══════════════════════════════════════════════════════════════════════════════
def create_db(scenario, db_schemas, sample_data, db_dir):
    """Create a fresh database for a scenario."""
    os.makedirs(db_dir, exist_ok=True)
    schema = db_schemas[scenario]
    db_path = os.path.join(db_dir, f"{scenario}_eval.db")
    tmp_path, _, _, _ = create_sqlite_database(scenario, schema["db_schema"], db_dir)
    sd = sample_data.get(scenario)
    if sd:
        execute_sample_data(tmp_path, sd["sample_data"], scenario)
    if tmp_path != db_path:
        shutil.move(tmp_path, db_path)
    return db_path


def write_server_script(scenario, db_path, host, port, envs_data):
    env_item = envs_data[scenario]
    code = env_item["full_code"]
    new_lines = [
        "import warnings",
        'warnings.filterwarnings("ignore", category=DeprecationWarning)',
        "from sqlalchemy.pool import NullPool",
    ]
    for line in code.split("\n"):
        if "create_engine(" in line:
            left = line.split("create_engine(")[0]
            line = (f"{left}create_engine('sqlite:///{db_path}', "
                    f"connect_args={{'check_same_thread': False}}, poolclass=NullPool)")
        if "uvicorn.run(app" in line:
            setup = textwrap.indent(textwrap.dedent(f"""\
                from fastapi_mcp import FastApiMCP
                mcp = FastApiMCP(app)
                mcp.mount_http()
            """), "    ")
            new_lines.extend(setup.rstrip().split("\n"))
            line = f"    uvicorn.run(app, host='{host}', port={port})"
        new_lines.append(line)

    script_dir = os.path.join("/dev/shm", f"awm_eval_scripts_{os.getuid()}")
    os.makedirs(script_dir, exist_ok=True)
    script_path = os.path.join(script_dir, f"eval_server_{scenario}_{port}.py")
    with open(script_path, "w") as f:
        f.write("\n".join(new_lines))
    return script_path


async def start_server(scenario, db_path, port, envs_data):
    """Start MCP server, return (proc, mcp_executor, tools_text)."""
    # Kill old process on port
    try:
        subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, timeout=5)
    except Exception:
        pass
    await asyncio.sleep(0.5)

    script_path = write_server_script(scenario, db_path, "127.0.0.1", port, envs_data)

    env = os.environ.copy()
    for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
        env[k] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"

    log_dir = os.path.join("/dev/shm", f"awm_eval_logs_{os.getuid()}")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"eval_{scenario}_{port}.log")

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen([sys.executable, script_path],
                                stdout=log_f, stderr=log_f, env=env)

    mcp_url = f"http://127.0.0.1:{port}/mcp"
    deadline = time.time() + EVAL_CONFIG["server_startup_timeout"]
    while time.time() < deadline:
        try:
            running, tools_count, _, _ = await check_mcp_server(url=mcp_url, timeout=2.0)
            if running and tools_count > 0:
                mcp = MCPToolExecutor(mcp_url, timeout=EVAL_CONFIG["tool_timeout"])
                await mcp.__aenter__()
                tools = await mcp.list_tools()
                tools_text = format_tools_for_response(tools)
                return proc, mcp, tools_text
        except Exception:
            pass
        await asyncio.sleep(0.5)

    proc.kill()
    raise RuntimeError(f"Server failed to start for {scenario} on port {port}")


async def stop_server(proc, mcp, port):
    if mcp:
        try:
            await mcp.__aexit__(None, None, None)
        except Exception:
            pass
    if proc:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    try:
        subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, timeout=3)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Verification
# ═══════════════════════════════════════════════════════════════════════════════
def compute_reward(scenario, task_idx, initial_db, final_db, final_answer, verifiers):
    key = f"{scenario}::{task_idx}"
    verifier = verifiers.get(key)
    if not verifier:
        return 0.0
    code = verifier.get("verification", {}).get("code", "")
    if not code:
        return 0.0

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
            final_db_path=final_db, final_answer=final_answer,
        )
        if result.get("execution_status") == "success":
            inner = result.get("result", {})
            if isinstance(inner, dict):
                val = inner.get("result", "")
                if isinstance(val, str) and val.lower() == "complete":
                    return 1.0
    except Exception:
        pass
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-turn rollout
# ═══════════════════════════════════════════════════════════════════════════════
STATUS_PATTERN = re.compile(r"<status>\s*(CONTINUE|VERIFY|REPLAN|STOP)\s*</status>", re.IGNORECASE)


async def run_single_rollout(
    scenario, task_idx, task_text, system_prompt, prompt_type,
    vllm_url, tokenizer, db_path, mcp, tools_text, verifiers,
):
    """Run one multi-turn rollout. Returns a result dict."""
    cfg = EVAL_CONFIG

    # Reset DB
    template_dir = os.path.join(cfg["db_dir"], "_templates")
    template_path = os.path.join(template_dir, f"{scenario}.db")
    shutil.copy2(template_path, db_path)

    initial_db = db_path + ".init_eval"
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

    response = ""
    final_answer = ""
    num_turns = 0
    tool_calls_made = 0
    status_sequence = []
    errors = []

    for _turn in range(cfg["max_turns"]):
        num_turns += 1

        payload = {
            "text": prompt_text + response,
            "sampling_params": {
                "temperature": cfg["temperature"],
                "top_k": cfg["top_k"],
                "top_p": cfg["top_p"],
                "max_new_tokens": cfg["max_new_tokens"],
            },
        }

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{vllm_url}/generate", json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    output = await resp.json()
        except Exception as e:
            errors.append(f"Generation error: {e}")
            break

        if output.get("meta_info", {}).get("finish_reason", {}).get("type") == "abort":
            break

        cur_response = output["text"]
        response += cur_response

        # Parse ESA status if present
        status_match = STATUS_PATTERN.search(cur_response)
        if status_match:
            status_sequence.append(status_match.group(1).upper())

        # Check for STOP
        if status_match and status_match.group(1).upper() == "STOP":
            final_answer = cur_response
            break

        # Parse tool calls
        tcs = parse_tool_calls(cur_response)
        if not tcs:
            final_answer = cur_response
            break

        tool_calls_made += 1
        tc = tcs[0]
        name, arguments = tc["name"], tc["arguments"]

        # Execute
        if name == "list_tools":
            obs_text = tools_text
        elif name == "call_tool":
            try:
                tool_name, tool_args = parse_call_tool_arguments(arguments)
                result = await mcp.call_tool(tool_name, tool_args)
                obs_text = result
            except asyncio.TimeoutError:
                obs_text = f"Error: Tool call timed out"
                errors.append("timeout")
            except Exception as e:
                obs_text = f"Error: {e}"
                errors.append(str(e))
        else:
            obs_text = f"Error: Unknown tool '{name}'"

        # Append observation
        next_obs = (
            f"<|im_start|>user\n<tool_response>\n{obs_text}\n</tool_response><|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )
        response += next_obs

        # Context check
        total_len = len(tokenizer.encode(prompt_text + response, add_special_tokens=False))
        if total_len + 100 >= cfg["max_context_len"]:
            break

    # Compute reward
    reward = compute_reward(scenario, task_idx, initial_db, db_path, final_answer, verifiers)

    # Cleanup
    try:
        os.remove(initial_db)
    except OSError:
        pass

    return {
        "scenario": scenario,
        "task_idx": task_idx,
        "task": task_text,
        "prompt_type": prompt_type,
        "reward": reward,
        "num_turns": num_turns,
        "tool_calls": tool_calls_made,
        "status_sequence": status_sequence,
        "errors": errors,
        "response_len": len(response),
        "trace": response,  # full conversation trace
    }


# ═══════════════════════════════════════════════════════════════════════════════
# vLLM server management
# ═══════════════════════════════════════════════════════════════════════════════
def start_vllm_server(model_path, num_gpus, port=8000):
    """Start a vLLM/SGLang server."""
    logger.info("Starting SGLang server with %d GPUs on port %d...", num_gpus, port)

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--tp", str(num_gpus),
        "--port", str(port),
        "--mem-fraction-static", "0.85",
        "--trust-remote-code",
    ]

    log_dir = os.path.join("/dev/shm", f"awm_eval_logs_{os.getuid()}")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "sglang_server.log")

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=log_f)

    # Wait for server
    import urllib.request
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            req = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            if req.status == 200:
                logger.info("SGLang server ready on port %d", port)
                return proc
        except Exception:
            pass
        time.sleep(3)
        if proc.poll() is not None:
            with open(log_path) as f:
                logger.error("Server died. Last log:\n%s", f.read()[-500:])
            raise RuntimeError("SGLang server died during startup")

    proc.kill()
    raise RuntimeError("SGLang server failed to start in 300s")


# ═══════════════════════════════════════════════════════════════════════════════
# Persistent Server Pool — start once, reuse across rollouts
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class _EvalSlot:
    index: int
    port: int
    scenario: str = ""
    proc: subprocess.Popen | None = None
    db_path: str = ""
    tools_text: str = ""
    mcp: MCPToolExecutor | None = None
    busy: bool = False


class ServerPool:
    """Pool of persistent MCP servers. Servers are reused when the scenario matches;
    only restarted when switching to a different scenario. DB is reset via file copy."""

    def __init__(self, num_slots, base_port, db_dir, envs_data):
        self.slots = [
            _EvalSlot(index=i, port=base_port + i,
                      db_path=os.path.join(db_dir, f"eval_slot_{i}.db"))
            for i in range(num_slots)
        ]
        self.envs_data = envs_data
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(num_slots)
        self._startup_sem = asyncio.Semaphore(8)  # limit concurrent startups

    async def acquire(self, scenario):
        """Get a slot for the given scenario. Prefers reusing same-scenario slots."""
        await self.semaphore.acquire()
        async with self.lock:
            # Prefer slot already running this scenario
            for slot in self.slots:
                if not slot.busy and slot.scenario == scenario and slot.proc and slot.proc.poll() is None:
                    slot.busy = True
                    return slot
            # Otherwise any free slot
            for slot in self.slots:
                if not slot.busy:
                    slot.busy = True
                    return slot
        self.semaphore.release()
        raise RuntimeError("No free eval slots")

    def release(self, slot):
        slot.busy = False
        self.semaphore.release()

    async def ensure_ready(self, slot, scenario, template_dir):
        """Ensure slot has a running server for the scenario. Reuse if possible."""
        template_path = os.path.join(template_dir, f"{scenario}.db")

        if (slot.scenario == scenario and slot.proc is not None
                and slot.proc.poll() is None and slot.mcp is not None):
            # Same scenario, server alive — just reset DB
            shutil.copy2(template_path, slot.db_path)
            return

        # Need to start a new server
        async with self._startup_sem:
            await self._start_server(slot, scenario, template_path)

    async def _start_server(self, slot, scenario, template_path):
        # Close old MCP
        if slot.mcp is not None:
            try:
                await slot.mcp.__aexit__(None, None, None)
            except Exception:
                pass
            slot.mcp = None

        # Kill old server
        if slot.proc is not None:
            try:
                slot.proc.terminate()
                slot.proc.wait(timeout=3)
            except Exception:
                try:
                    slot.proc.kill()
                except Exception:
                    pass
            slot.proc = None
        try:
            subprocess.run(["fuser", "-k", f"{slot.port}/tcp"], capture_output=True, timeout=3)
        except Exception:
            pass

        # Copy template DB
        shutil.copy2(template_path, slot.db_path)

        # Write and launch server
        script_path = write_server_script(scenario, slot.db_path, "127.0.0.1", slot.port, self.envs_data)

        env = os.environ.copy()
        for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
            env[k] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"

        log_dir = os.path.join("/dev/shm", f"awm_eval_logs_{os.getuid()}")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f"eval_{slot.index}.log"), "w") as log_f:
            slot.proc = subprocess.Popen([sys.executable, script_path],
                                         stdout=log_f, stderr=log_f, env=env)

        # Wait for ready
        mcp_url = f"http://127.0.0.1:{slot.port}/mcp"
        deadline = time.time() + EVAL_CONFIG["server_startup_timeout"]
        while time.time() < deadline:
            try:
                running, tools_count, _, _ = await check_mcp_server(url=mcp_url, timeout=2.0)
                if running and tools_count > 0:
                    slot.mcp = MCPToolExecutor(mcp_url, timeout=EVAL_CONFIG["tool_timeout"])
                    await slot.mcp.__aenter__()
                    tools = await slot.mcp.list_tools()
                    slot.tools_text = format_tools_for_response(tools)
                    slot.scenario = scenario
                    return
            except Exception:
                pass
            await asyncio.sleep(0.3)
        raise RuntimeError(f"Server failed to start: slot={slot.index} scenario={scenario}")

    async def shutdown(self):
        for slot in self.slots:
            if slot.mcp:
                try:
                    await slot.mcp.__aexit__(None, None, None)
                except Exception:
                    pass
            if slot.proc:
                try:
                    slot.proc.terminate()
                    slot.proc.wait(timeout=3)
                except Exception:
                    try:
                        slot.proc.kill()
                    except Exception:
                        pass
                try:
                    subprocess.run(["fuser", "-k", f"{slot.port}/tcp"], capture_output=True, timeout=3)
                except Exception:
                    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ═══════════════════════════════════════════════════════════════════════════════
async def run_eval(args):
    # Load data
    db_schemas, sample_data, verifiers, envs_data, tasks = load_data()
    selected_tasks = select_tasks(verifiers, tasks, args.num_tasks)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Prepare prompts
    awm_prompt = get_system_prompt()
    esa_prompt = get_esa_system_prompt()

    vllm_url = f"http://127.0.0.1:{args.vllm_port}"

    # Prepare template DBs
    template_dir = os.path.join(EVAL_CONFIG["db_dir"], "_templates")
    os.makedirs(template_dir, exist_ok=True)

    needed_scenarios = set(s for s, _, _ in selected_tasks)
    logger.info("Preparing template DBs for %d scenarios...", len(needed_scenarios))
    for scenario in needed_scenarios:
        template_path = os.path.join(template_dir, f"{scenario}.db")
        if not os.path.exists(template_path):
            create_db(scenario, db_schemas, sample_data, template_dir)
            src = os.path.join(template_dir, f"{scenario}_eval.db")
            if os.path.exists(src) and not os.path.exists(template_path):
                shutil.move(src, template_path)

    # Results storage
    results = []
    output_path = os.path.join(SCRIPT_DIR, "data", "prompt_comparison_results.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _write_lock = asyncio.Lock()

    # Create persistent server pool
    base_port = 9200
    num_workers = args.num_workers
    pool = ServerPool(num_workers, base_port, EVAL_CONFIG["db_dir"], envs_data)

    total_runs = len(selected_tasks) * args.repeats * 2  # 2 prompts
    completed = 0
    t_start = time.time()

    async def run_one_task(scenario, task_idx, task_text, system_prompt, prompt_type):
        nonlocal completed
        slot = await pool.acquire(scenario)
        try:
            await pool.ensure_ready(slot, scenario, template_dir)

            result = await run_single_rollout(
                scenario, task_idx, task_text, system_prompt, prompt_type,
                vllm_url, tokenizer, slot.db_path, slot.mcp, slot.tools_text, verifiers,
            )
            results.append(result)

            async with _write_lock:
                with open(output_path, "a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

            completed += 1
            if completed % 50 == 0:
                elapsed = time.time() - t_start
                rate = completed / elapsed
                eta = (total_runs - completed) / rate if rate > 0 else 0
                logger.info("Progress: %d/%d (%.1f%%) | %.1f runs/min | ETA: %.0fs",
                            completed, total_runs, 100*completed/total_runs,
                            rate * 60, eta)

        except Exception as e:
            logger.error("Task %s[%d] %s failed: %s", scenario, task_idx, prompt_type, e)
            completed += 1
        finally:
            pool.release(slot)

    # Clear previous results
    if os.path.exists(output_path):
        os.remove(output_path)

    # Build task list grouped by scenario for maximum server reuse
    # Each task × repeats × 2 prompts
    task_queue = []
    for scenario, task_idx, task_text in selected_tasks:
        for repeat in range(args.repeats):
            task_queue.append((scenario, task_idx, task_text, awm_prompt, "awm"))
            task_queue.append((scenario, task_idx, task_text, esa_prompt, "esa"))

    # Sort by scenario to maximize server reuse, then shuffle within scenario
    from itertools import groupby
    task_queue.sort(key=lambda x: x[0])
    grouped = []
    for _, group in groupby(task_queue, key=lambda x: x[0]):
        g = list(group)
        random.seed(123)
        random.shuffle(g)
        grouped.extend(g)
    task_queue = grouped

    logger.info("Starting evaluation: %d tasks × %d repeats × 2 prompts = %d total runs (%d workers)",
                args.num_tasks, args.repeats, total_runs, num_workers)

    # Launch all tasks with asyncio concurrency (pool.semaphore limits actual parallelism)
    batch_size = num_workers * 4  # larger batches, pool handles concurrency
    for i in range(0, len(task_queue), batch_size):
        batch = task_queue[i:i+batch_size]
        coros = [run_one_task(s, ti, tt, sp, pt) for s, ti, tt, sp, pt in batch]
        await asyncio.gather(*coros, return_exceptions=True)

    # Shutdown pool
    await pool.shutdown()

    # ── Analysis ──
    elapsed = time.time() - t_start
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE in %.1f minutes (%.1f runs/min)", elapsed/60, len(results)/(elapsed/60))
    logger.info("=" * 60)

    awm_results = [r for r in results if r["prompt_type"] == "awm"]
    esa_results = [r for r in results if r["prompt_type"] == "esa"]

    def stats(rs):
        if not rs:
            return {}
        rewards = [r["reward"] for r in rs]
        n_completed = sum(1 for r in rewards if r == 1.0)
        turns = [r["num_turns"] for r in rs]
        tool_calls = [r["tool_calls"] for r in rs]
        return {
            "count": len(rs),
            "reward_mean": round(sum(rewards) / len(rewards), 4),
            "completion_rate": round(n_completed / len(rs), 4),
            "reward_nonzero_frac": round(sum(1 for r in rewards if r > 0) / len(rs), 4),
            "turns_mean": round(sum(turns) / len(turns), 2),
            "tool_calls_mean": round(sum(tool_calls) / len(tool_calls), 2),
        }

    awm_stats = stats(awm_results)
    esa_stats = stats(esa_results)

    logger.info("\nAWM Prompt:  %s", json.dumps(awm_stats, indent=2))
    logger.info("\nESA Prompt:  %s", json.dumps(esa_stats, indent=2))

    # ESA-specific: status distribution
    if esa_results:
        all_statuses = []
        for r in esa_results:
            all_statuses.extend(r.get("status_sequence", []))
        if all_statuses:
            from collections import Counter
            status_dist = Counter(all_statuses)
            logger.info("\nESA Status Distribution: %s", dict(status_dist))

    # Per-task comparison
    task_comparison = {}
    for r in results:
        key = f"{r['scenario']}::{r['task_idx']}"
        if key not in task_comparison:
            task_comparison[key] = {"awm": [], "esa": []}
        task_comparison[key][r["prompt_type"]].append(r["reward"])

    improved = degraded = same = 0
    for key, v in task_comparison.items():
        awm_mean = sum(v["awm"]) / max(1, len(v["awm"])) if v["awm"] else 0
        esa_mean = sum(v["esa"]) / max(1, len(v["esa"])) if v["esa"] else 0
        if esa_mean > awm_mean + 0.05:
            improved += 1
        elif esa_mean < awm_mean - 0.05:
            degraded += 1
        else:
            same += 1

    logger.info("\nPer-task comparison (threshold +/-0.05):")
    logger.info("  ESA better: %d  |  Same: %d  |  AWM better: %d", improved, same, degraded)

    # Save summary
    summary = {
        "awm": awm_stats,
        "esa": esa_stats,
        "per_task": {"esa_better": improved, "same": same, "awm_better": degraded},
        "elapsed_minutes": round(elapsed / 60, 1),
    }
    summary_path = os.path.join(SCRIPT_DIR, "data", "prompt_comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("\nResults saved to: %s", output_path)
    logger.info("Summary saved to: %s", summary_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=os.path.join(SLIME_DIR, "..", "models/Qwen3-4B"))
    parser.add_argument("--num_tasks", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=16, help="concurrent MCP servers")
    parser.add_argument("--vllm_port", type=int, default=8000)
    parser.add_argument("--skip_vllm", action="store_true", help="assume vLLM already running")
    args = parser.parse_args()

    vllm_proc = None
    if not args.skip_vllm:
        vllm_proc = start_vllm_server(args.model_path, args.num_gpus, args.vllm_port)

    try:
        asyncio.run(run_eval(args))
    finally:
        if vllm_proc:
            logger.info("Shutting down SGLang server...")
            vllm_proc.terminate()
            try:
                vllm_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                vllm_proc.kill()
        # Clean up MCP servers
        subprocess.run("pkill -9 -f 'eval_server_'", shell=True, capture_output=True)


if __name__ == "__main__":
    main()
