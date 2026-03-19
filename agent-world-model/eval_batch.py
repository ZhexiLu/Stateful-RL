"""
Parallel batch evaluation script for AWM 1K environments.

Usage:
  1. Start vLLM:
     CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-4B --host 127.0.0.1 --port 8000 --max-model-len 32768

  2. Run evaluation (parallel with 8 workers):
     python eval_batch.py \
       --model Qwen/Qwen3-4B \
       --vllm_url http://localhost:8000/v1 \
       --num_workers 8 \
       --max_scenarios 100
"""

import argparse
import asyncio
import json
import os
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from awm.tools import (
    tools_jsonl_load,
    tools_robust_json_loads,
    normalize_scenario_name,
    check_mcp_server,
)
from awm.core.agent import (
    MCPToolExecutor,
    get_system_prompt,
    generate_response,
    parse_tool_calls,
    parse_call_tool_arguments,
    format_tools_for_response,
    Config as AgentConfig,
)
from awm.core.server import Config as ServerConfig, run_server
from awm.core.verifier import execute_verification_code, VerificationMode
from awm.core.db import create_sqlite_database
from awm.core.sample import execute_sample_data


# ── Global data (loaded once, shared across threads) ──────────────────────────
DB_SCHEMAS: dict[str, dict] = {}
SAMPLE_DATA: dict[str, dict] = {}

# Thread-safe lock for writing results
RESULTS_LOCK = threading.Lock()

# Global counters
COUNTERS_LOCK = threading.Lock()
COUNTERS = {"total_tasks": 0, "total_complete": 0, "total_agent_error": 0, "scenarios_done": 0}


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel batch evaluation on AWM-1K")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--envs_path", type=str, default="outputs/gen_envs.jsonl")
    parser.add_argument("--tasks_path", type=str, default="outputs/gen_tasks.jsonl")
    parser.add_argument("--verifier_path", type=str, default="outputs/gen_verifier.pure_code.jsonl")
    parser.add_argument("--sql_verifier_path", type=str, default="outputs/gen_verifier.jsonl",
                        help="SQL-mode verifier for LLM-as-a-Judge fallback")
    parser.add_argument("--judge_model", type=str, default="",
                        help="Model for LLM judge fallback (empty=same as agent model)")
    parser.add_argument("--judge_url", type=str, default="",
                        help="vLLM URL for judge model (empty=same as vllm_url)")
    parser.add_argument("--db_dir", type=str, default="outputs/databases")
    parser.add_argument("--db_schema_path", type=str, default="outputs/gen_db.jsonl")
    parser.add_argument("--sample_path", type=str, default="outputs/gen_sample.jsonl")
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--max_scenarios", type=int, default=0, help="0 = all scenarios")
    parser.add_argument("--max_iterations", type=int, default=15)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--base_port", type=int, default=8001, help="Base port for MCP servers")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--server_startup_timeout", type=float, default=12.0)
    parser.add_argument("--llm_timeout", type=float, default=60.0, help="Timeout per LLM request in seconds")
    parser.add_argument("--task_timeout", type=float, default=300.0, help="Timeout per task in seconds")
    parser.add_argument("--verifier_timeout", type=float, default=60.0, help="Timeout per verifier execution in seconds")
    parser.add_argument("--resume", action="store_true", help="Resume from existing results")
    parser.add_argument("--enable_thinking", action="store_true", default=True)
    parser.add_argument("--no_thinking", action="store_true", help="Disable thinking mode")
    parser.add_argument("--worker_mode", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker_payload", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker_output", type=str, default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def load_verifiers(verifier_path: str) -> dict[str, dict]:
    verifiers = {}
    data = tools_jsonl_load(verifier_path)
    for item in data:
        scenario = normalize_scenario_name(item["scenario"])
        task_idx = item["task_idx"]
        verifiers[f"{scenario}::{task_idx}"] = item
    return verifiers


def preload_global_data(db_schema_path: str, sample_path: str):
    """Load DB schemas and sample data once into global dicts."""
    global DB_SCHEMAS, SAMPLE_DATA
    for item in tools_jsonl_load(db_schema_path):
        DB_SCHEMAS[normalize_scenario_name(item["scenario"])] = item
    for item in tools_jsonl_load(sample_path):
        SAMPLE_DATA[normalize_scenario_name(item["scenario"])] = item
    logger.info(f"Preloaded {len(DB_SCHEMAS)} schemas, {len(SAMPLE_DATA)} sample data")


def reset_db_fast(scenario: str, db_dir: str) -> str:
    """Reset DB using preloaded global data (no re-reading JSONL)."""
    schema_item = DB_SCHEMAS.get(scenario)
    if schema_item is None:
        raise RuntimeError(f"Schema not found for {scenario}")
    db_path, _, _, _ = create_sqlite_database(scenario, schema_item["db_schema"], db_dir)
    sample_item = SAMPLE_DATA.get(scenario)
    if sample_item:
        execute_sample_data(db_path, sample_item["sample_data"], scenario)
    return db_path


def start_mcp_server(scenario: str, envs_path: str, db_dir: str, port: int) -> subprocess.Popen:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "outputs", "server_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{scenario}_{port}.log")
    log_handle = open(log_path, "wb")
    cmd = [
        sys.executable, "-c",
        f"""
import sys
sys.path.insert(0, '.')
from awm.core.server import Config, run_server
config = Config(
    scenario='{scenario}',
    envs_load_path='{envs_path}',
    db_path='{os.path.join(db_dir, scenario + ".db")}',
    host='127.0.0.1',
    port={port},
)
config.pre_process()
run_server(config)
"""
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=script_dir,
        stdout=log_handle,
        stderr=log_handle,
    )
    proc._log_handle = log_handle
    proc._log_path = log_path
    return proc


def is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.1)
        return sock.connect_ex((host, port)) == 0


def wait_for_port_close(port: int, timeout: float = 0.5, poll_interval: float = 0.05) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not is_port_open(port):
            return True
        time.sleep(poll_interval)
    return not is_port_open(port)


def kill_port(port: int):
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"], capture_output=True, text=True, timeout=5,
        )
        killed = False
        for pid in result.stdout.strip().split():
            if pid:
                os.kill(int(pid), signal.SIGKILL)
                killed = True
        if killed:
            wait_for_port_close(port)
    except Exception:
        pass


def stop_server(proc: subprocess.Popen, port: int):
    try:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1)
    finally:
        log_handle = getattr(proc, "_log_handle", None) if proc else None
        if log_handle is not None:
            try:
                log_handle.close()
            except Exception:
                pass
        kill_port(port)


async def wait_for_server(
    url: str,
    timeout: float = 12.0,
    poll_interval: float = 0.25,
    probe_timeout: float = 1.5,
) -> bool:
    deadline = time.time() + timeout
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return False
        try:
            running, tools_count, _, _ = await check_mcp_server(
                url=url,
                timeout=min(probe_timeout, remaining),
            )
            if running and tools_count > 0:
                return True
        except Exception:
            pass
        await asyncio.sleep(min(poll_interval, max(deadline - time.time(), 0)))


async def run_agent_on_task(
    task: str, mcp_url: str, vllm_url: str, model: str,
    max_iterations: int = 30, max_tokens: int = 2048,
    temperature: float = 0.6, enable_thinking: bool = True,
    llm_timeout: float = 60.0,
) -> dict:
    from openai import AsyncOpenAI

    config = AgentConfig(
        task=task, mcp_url=mcp_url, vllm_url=vllm_url, model=model,
        max_iterations=max_iterations, max_tokens=max_tokens,
        temperature=temperature, request_timeout=llm_timeout,
        enable_thinking=enable_thinking, verbose=False,
    )

    vllm_client = None
    try:
        vllm_client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
            base_url=vllm_url,
            timeout=llm_timeout,
        )

        async with MCPToolExecutor(mcp_url, timeout=llm_timeout) as mcp:
            tools = await mcp.list_tools()
            tools_response_text = format_tools_for_response(tools)

            messages = [
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": task},
            ]

            trajectory = []
            final_answer = ""
            num_iterations = 0

            for iteration in range(1, max_iterations + 1):
                num_iterations = iteration
                content, tool_calls = await generate_response(
                    vllm_client, model, messages, config
                )

                messages.append({"role": "assistant", "content": content})
                trajectory.append({"role": "assistant", "content": content, "tool_calls": len(tool_calls)})

                if not tool_calls:
                    final_answer = content
                    break

                tc = tool_calls[0]
                name, arguments, tool_call_id = tc["name"], tc["arguments"], tc["id"]

                if name == "list_tools":
                    response_text = tools_response_text
                elif name == "call_tool":
                    tool_name, tool_args = parse_call_tool_arguments(arguments)
                    try:
                        response_text = await mcp.call_tool(tool_name, tool_args)
                    except asyncio.TimeoutError:
                        response_text = "Error: Tool call timed out"
                    except Exception as e:
                        response_text = f"Error: {e}"
                else:
                    response_text = f"Error: Unknown tool '{name}'"

                messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": response_text})
                tool_detail = {"role": "tool", "tool_name": name, "content": response_text}
                if name == "call_tool":
                    tool_detail["called_tool"] = tool_name
                    tool_detail["called_args"] = tool_args
                trajectory.append(tool_detail)

        return {"success": True, "final_answer": final_answer,
                "num_iterations": num_iterations, "trajectory": trajectory}
    except asyncio.TimeoutError:
        return {"success": False, "error": f"LLM request timed out after {llm_timeout}s",
                "traceback": traceback.format_exc(),
                "final_answer": "", "num_iterations": 0, "trajectory": []}
    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc(),
                "final_answer": "", "num_iterations": 0, "trajectory": []}
    finally:
        if vllm_client is not None:
            close = getattr(vllm_client, "close", None)
            if close is not None:
                try:
                    await asyncio.wait_for(close(), timeout=2.0)
                except Exception:
                    pass


def _worker_success_result(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _run_worker_subprocess(worker_mode: str, payload: dict, timeout: float) -> dict:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    with tempfile.TemporaryDirectory(prefix=f"eval_{worker_mode}_") as tmpdir:
        payload_path = os.path.join(tmpdir, "payload.json")
        output_path = os.path.join(tmpdir, "output.json")

        with open(payload_path, "w") as f:
            json.dump(payload, f, ensure_ascii=False)

        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    script_path,
                    "--worker_mode", worker_mode,
                    "--worker_payload", payload_path,
                    "--worker_output", output_path,
                ],
                cwd=script_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=max(1.0, float(timeout)),
            )
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(f"{worker_mode} timed out after {timeout}s") from e

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            if len(stderr) > 500:
                stderr = stderr[:500] + "..."
            raise RuntimeError(
                f"{worker_mode} worker failed with exit code {completed.returncode}: {stderr or 'no stderr'}"
            )

        if not os.path.exists(output_path):
            raise RuntimeError(f"{worker_mode} worker produced no output")

        return _worker_success_result(output_path)


def run_agent_on_task_subprocess(
    task: str,
    mcp_url: str,
    vllm_url: str,
    model: str,
    max_iterations: int,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    llm_timeout: float,
    task_timeout: float,
) -> dict:
    payload = {
        "task": task,
        "mcp_url": mcp_url,
        "vllm_url": vllm_url,
        "model": model,
        "max_iterations": max_iterations,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "enable_thinking": enable_thinking,
        "llm_timeout": llm_timeout,
    }
    try:
        return _run_worker_subprocess("agent_task", payload, timeout=task_timeout)
    except TimeoutError:
        return {
            "success": False,
            "error": f"Task timed out after {task_timeout}s",
            "final_answer": "",
            "num_iterations": 0,
            "trajectory": [],
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "final_answer": "",
            "num_iterations": 0,
            "trajectory": [],
        }


def run_verifier_subprocess(
    verifier_item: dict,
    initial_db_path: str,
    final_db_path: str,
    final_answer: str,
    timeout: float,
) -> dict:
    payload = {
        "verifier_item": verifier_item,
        "initial_db_path": initial_db_path,
        "final_db_path": final_db_path,
        "final_answer": final_answer,
    }
    try:
        return _run_worker_subprocess("verifier_task", payload, timeout=timeout)
    except TimeoutError:
        return {"status": "error", "error": f"Verifier timed out after {timeout}s"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _run_worker_mode(args) -> int:
    if args.worker_mode == "agent_task":
        with open(args.worker_payload, "r") as f:
            payload = json.load(f)
        result = asyncio.run(run_agent_on_task(**payload))
    elif args.worker_mode == "verifier_task":
        with open(args.worker_payload, "r") as f:
            payload = json.load(f)
        result = run_verifier(
            payload["verifier_item"],
            initial_db_path=payload["initial_db_path"],
            final_db_path=payload["final_db_path"],
            final_answer=payload.get("final_answer", ""),
        )
    else:
        raise RuntimeError(f"Unknown worker mode: {args.worker_mode}")

    with open(args.worker_output, "w") as f:
        json.dump(result, f, ensure_ascii=False)
    return 0


def run_verifier(verifier_item: dict, initial_db_path: str, final_db_path: str, final_answer: str = "") -> dict:
    code = verifier_item.get("verification", {}).get("code", "")
    if not code:
        return {"status": "no_verifier", "result": None}

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
            initial_db_path=initial_db_path, mode=mode,
            final_db_path=final_db_path,
            final_answer=final_answer,
        )
        return {"status": "executed", "result": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def check_task_complete(verify_result: dict) -> bool:
    if verify_result.get("status") != "executed":
        return False
    exec_result = verify_result.get("result", {})
    if exec_result.get("execution_status") != "success":
        return False
    inner = exec_result.get("result", {})
    if isinstance(inner, dict):
        result_val = inner.get("result", "")
        if isinstance(result_val, str) and result_val.lower() == "complete":
            return True
    return False


def load_completed_scenarios(output_dir: str) -> set[str]:
    completed = set()
    results_file = os.path.join(output_dir, "results.jsonl")
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            for line in f:
                try:
                    completed.add(json.loads(line.strip())["scenario"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return completed


def eval_single_scenario(
    scenario: str,
    tasks: list[str],
    verifiers: dict,
    args,
    port: int,
    total_scenarios: int,
) -> list[dict]:
    """Evaluate all tasks for a single scenario. Runs in a thread."""
    worker_id = port - args.base_port
    mcp_url = f"http://127.0.0.1:{port}/mcp"
    enable_thinking = args.enable_thinking and not args.no_thinking

    loop = asyncio.new_event_loop()
    scenario_results = []

    for task_idx, task in enumerate(tasks):
        db_path = os.path.join(args.db_dir, f"{scenario}.db")
        initial_db_snapshot = os.path.join(args.db_dir, f"{scenario}_initial_w{worker_id}.db")
        server_proc = None

        try:
            reset_db_fast(scenario, args.db_dir)
        except Exception as e:
            logger.error(f"[W{worker_id}] DB reset failed for {scenario}[{task_idx}]: {e}")
            scenario_results.append({
                "scenario": scenario,
                "task_idx": task_idx,
                "task": task,
                "agent_success": False,
                "num_iterations": 0,
                "is_complete": False,
                "verify_status": "no_verifier",
                "final_answer": "",
                "trajectory": [],
                "agent_error": f"DB reset failed: {e}",
            })
            continue

        # Recreate the server after each reset to avoid stale SQLite connections.
        shutil.copy2(db_path, initial_db_snapshot)

        kill_port(port)
        server_proc = start_mcp_server(scenario, args.envs_path, args.db_dir, port)
        server_ready = loop.run_until_complete(
            wait_for_server(mcp_url, timeout=args.server_startup_timeout)
        )
        if not server_ready:
            logger.error(f"[W{worker_id}] MCP server failed to start for {scenario}[{task_idx}]")
            stop_server(server_proc, port)
            if server_proc.poll() is not None:
                log_path = getattr(server_proc, "_log_path", "")
                if log_path and os.path.exists(log_path):
                    try:
                        with open(log_path, "rb") as f:
                            f.seek(0, os.SEEK_END)
                            f.seek(max(f.tell() - 300, 0))
                            preview = f.read().decode(errors="replace")
                        logger.error(f"[W{worker_id}] server log: {preview[-300:]}")
                    except Exception:
                        pass

            try:
                os.remove(initial_db_snapshot)
            except OSError:
                pass

            scenario_results.append({
                "scenario": scenario,
                "task_idx": task_idx,
                "task": task,
                "agent_success": False,
                "num_iterations": 0,
                "is_complete": False,
                "verify_status": "no_verifier",
                "final_answer": "",
                "trajectory": [],
                "agent_error": "MCP server failed to start",
            })
            continue

        try:
            agent_result = run_agent_on_task_subprocess(
                task=task, mcp_url=mcp_url, vllm_url=args.vllm_url, model=args.model,
                max_iterations=args.max_iterations, max_tokens=args.max_tokens,
                temperature=args.temperature, enable_thinking=enable_thinking,
                llm_timeout=args.llm_timeout, task_timeout=args.task_timeout,
            )
        finally:
            stop_server(server_proc, port)

        verifier_key = f"{scenario}::{task_idx}"
        verifier_item = verifiers.get(verifier_key, {})
        verify_result = {"status": "no_verifier"}
        if agent_result["success"] and verifier_item:
            verify_result = run_verifier_subprocess(
                verifier_item,
                initial_db_path=initial_db_snapshot,
                final_db_path=db_path,
                final_answer=agent_result.get("final_answer", ""),
                timeout=args.verifier_timeout,
            )

        # Clean up snapshot
        try:
            os.remove(initial_db_snapshot)
        except OSError:
            pass

        is_complete = check_task_complete(verify_result)

        task_result = {
            "scenario": scenario, "task_idx": task_idx, "task": task,
            "agent_success": agent_result["success"],
            "num_iterations": agent_result["num_iterations"],
            "is_complete": is_complete,
            "verify_status": verify_result.get("status"),
            "final_answer": agent_result.get("final_answer", ""),
            "trajectory": agent_result.get("trajectory", []),
        }
        if not agent_result["success"]:
            task_result["agent_error"] = agent_result.get("error", "")

        scenario_results.append(task_result)

        status = "COMPLETE" if is_complete else ("ERROR" if not agent_result["success"] else "INCOMPLETE")
        logger.info(f"[W{worker_id}] {scenario}[{task_idx}] -> {status} (iters={agent_result['num_iterations']})")

    loop.close()

    # Update global counters
    sc_complete = sum(1 for r in scenario_results if r["is_complete"])
    sc_errors = sum(1 for r in scenario_results if r.get("agent_error"))

    with COUNTERS_LOCK:
        COUNTERS["total_tasks"] += len(scenario_results)
        COUNTERS["total_complete"] += sc_complete
        COUNTERS["total_agent_error"] += sc_errors
        COUNTERS["scenarios_done"] += 1
        done = COUNTERS["scenarios_done"]
        tc = COUNTERS["total_complete"]
        tt = COUNTERS["total_tasks"]

    logger.info(f"[W{worker_id}] {scenario} done: {sc_complete}/{len(tasks)} | "
                f"Overall: {tc}/{tt} ({tc/max(tt,1)*100:.1f}%) | "
                f"Scenarios: {done}/{total_scenarios}")

    return scenario_results


def main():
    args = parse_args()
    if args.worker_mode:
        raise SystemExit(_run_worker_mode(args))

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.db_dir, exist_ok=True)

    # Load all data once
    logger.info("Loading data...")
    preload_global_data(args.db_schema_path, args.sample_path)

    tasks_data = tools_jsonl_load(args.tasks_path)
    tasks_by_scenario = {
        normalize_scenario_name(item["scenario"]): item["tasks"]
        for item in tasks_data
    }

    envs_data = tools_jsonl_load(args.envs_path)
    available_scenarios = [normalize_scenario_name(e["scenario"]) for e in envs_data]

    verifiers = load_verifiers(args.verifier_path)

    # Resume support
    completed_scenarios = set()
    if args.resume:
        completed_scenarios = load_completed_scenarios(args.output_dir)
        logger.info(f"Resuming: {len(completed_scenarios)} scenarios already done")

    if args.max_scenarios > 0:
        # Take the first max_scenarios from available, then filter out completed
        candidate_scenarios = available_scenarios[:args.max_scenarios]
        scenarios = [s for s in candidate_scenarios if s not in completed_scenarios]
    else:
        scenarios = [s for s in available_scenarios if s not in completed_scenarios]

    if not scenarios:
        logger.info("No scenarios left to evaluate.")
        summary = {
            "model": args.model,
            "total_scenarios": 0,
            "total_tasks": 0,
            "total_complete": 0,
            "total_agent_error": 0,
            "completion_rate": 0.0,
            "enable_thinking": args.enable_thinking and not args.no_thinking,
            "max_iterations": args.max_iterations,
            "temperature": args.temperature,
            "llm_timeout": args.llm_timeout,
            "task_timeout": args.task_timeout,
            "verifier_timeout": args.verifier_timeout,
            "num_workers": 0,
        }
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return

    num_workers = min(args.num_workers, len(scenarios))
    logger.info(f"Evaluating {len(scenarios)} scenarios with {num_workers} parallel workers")
    logger.info(f"Model: {args.model} | Ports: {args.base_port}-{args.base_port + num_workers - 1}")

    results_file = os.path.join(args.output_dir, "results.jsonl")
    summary_file = os.path.join(args.output_dir, "summary.json")

    # Assign each scenario a dedicated port via round-robin
    # But we must ensure only one scenario runs on each port at a time.
    # We split scenarios into batches of num_workers and process each batch in parallel.
    batches = [scenarios[i:i + num_workers] for i in range(0, len(scenarios), num_workers)]

    for batch_idx, batch in enumerate(batches):
        logger.info(f"\n--- Batch {batch_idx+1}/{len(batches)} ({len(batch)} scenarios) ---")

        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = {}
            for worker_idx, scenario in enumerate(batch):
                tasks = tasks_by_scenario.get(scenario, [])
                if not tasks:
                    continue
                port = args.base_port + worker_idx
                future = executor.submit(
                    eval_single_scenario, scenario, tasks, verifiers,
                    args, port, len(scenarios),
                )
                futures[future] = scenario

            for future in as_completed(futures):
                scenario = futures[future]
                try:
                    scenario_results = future.result()
                except Exception as e:
                    logger.error(f"Scenario {scenario} crashed: {e}")
                    continue

                if not scenario_results:
                    continue

                # Save results (thread-safe)
                with RESULTS_LOCK:
                    with open(results_file, "a") as f:
                        for r in scenario_results:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")

                    detail_file = os.path.join(args.output_dir, f"detail_{scenario}.json")
                    with open(detail_file, "w") as f:
                        json.dump(scenario_results, f, ensure_ascii=False, indent=2)

    # Final summary
    with COUNTERS_LOCK:
        summary = {
            "model": args.model,
            "total_scenarios": COUNTERS["scenarios_done"],
            "total_tasks": COUNTERS["total_tasks"],
            "total_complete": COUNTERS["total_complete"],
            "total_agent_error": COUNTERS["total_agent_error"],
            "completion_rate": COUNTERS["total_complete"] / max(COUNTERS["total_tasks"], 1),
            "enable_thinking": args.enable_thinking and not args.no_thinking,
            "max_iterations": args.max_iterations,
            "temperature": args.temperature,
            "llm_timeout": args.llm_timeout,
            "task_timeout": args.task_timeout,
            "verifier_timeout": args.verifier_timeout,
            "num_workers": num_workers,
        }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Workers: {num_workers}")
    logger.info(f"Scenarios: {summary['total_scenarios']}")
    logger.info(f"Tasks: {summary['total_tasks']}")
    logger.info(f"Complete: {summary['total_complete']} ({summary['completion_rate']*100:.1f}%)")
    logger.info(f"Agent Errors: {summary['total_agent_error']}")
    logger.info(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
