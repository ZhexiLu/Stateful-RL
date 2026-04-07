# ESA (Execution-Status Awareness) environment for slime RL training.
#
# Extends AWM multi-turn tool-calling with:
#   1. Structured control block (CONTINUE/VERIFY/REPLAN/STOP) scaffold
#   2. State-based progress reward from database state diffs
#   3. Turn-level reward encoding for fine-grained credit assignment
#
# Based on examples/awm/generate_with_awm.py

import asyncio
import json
import logging
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import dataclass, field

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# ── AWM imports ──────────────────────────────────────────────────────────────
AWM_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "agent-world-model")
if AWM_ROOT not in sys.path:
    sys.path.insert(0, AWM_ROOT)

from awm.tools import tools_jsonl_load, normalize_scenario_name, check_mcp_server
from awm.core.agent import (
    MCPToolExecutor, parse_tool_calls,
    parse_call_tool_arguments, format_tools_for_response,
)
from awm.core.db import create_sqlite_database
from awm.core.sample import execute_sample_data
from awm.core.verifier import execute_verification_code, VerificationMode

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════
ESA_CONFIGS = {
    # ── AWM base ──
    "max_turns": 15,
    "pool_size": 32,
    "max_concurrent_starts": 16,
    "server_startup_timeout": 60.0,
    "tool_timeout": 10.0,
    "db_schema_path": os.path.join(AWM_ROOT, "outputs/gen_db.jsonl"),
    "sample_path": os.path.join(AWM_ROOT, "outputs/gen_sample.jsonl"),
    "verifier_path": os.path.join(AWM_ROOT, "outputs/gen_verifier.pure_code.jsonl"),
    "envs_path": os.path.join(AWM_ROOT, "outputs/gen_envs.jsonl"),
    "db_dir": "/dev/shm/awm_databases",
    "sql_verifier_path": os.path.join(AWM_ROOT, "outputs/gen_verifier.jsonl"),
    "judge_url": "http://127.0.0.1:8001",
    "return_logprob": True,
    # ── ESA-specific ──
    "predicate_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "predicates.jsonl"),
    "lambda_prog": 0.5,        # weight for progress reward component
    "gamma_verify": 0.05,      # verification bonus (β in paper)
    "eta": 0.95,               # discount factor for turn-level return-to-go
    "format_error_penalty": -1.0,  # AWM: hard penalty for format errors + early termination
    "format_correct_bonus": 0.1,   # reward for outputting valid <status>...</status> each turn
    "verify_violation_penalty": -0.3,  # penalty when VERIFY status but action is write
    "duplicate_penalty": -0.1,         # penalty for repeating exact same tool call
    "omega_step": 0.3,         # weight for step-level advantage in composite reward
    "enable_turn_level": True,  # encode turn indices in loss_mask
}

POOL_SEMAPHORE = asyncio.Semaphore(ESA_CONFIGS["pool_size"])
_STARTUP_SEMAPHORE = asyncio.Semaphore(ESA_CONFIGS["max_concurrent_starts"])


# ═══════════════════════════════════════════════════════════════════════════════
# ESA System Prompt
# ═══════════════════════════════════════════════════════════════════════════════
def get_esa_system_prompt() -> str:
    """Build ESA system prompt with status-inside-think format.

    Status tag goes at the beginning of each <think> block:
        <think>
        <status>CONTINUE</status>
        reasoning...
        </think>
        <tool_call>...</tool_call>

    This matches the observation template seed (<think>\\n<status>)
    and the training data in train.jsonl.
    """
    from textwrap import dedent

    return dedent("""\
        # MCP Tools

        You are at a MCP environment. You need to call MCP tools to assist with the user query. At each step, you can only call one function. You have already logged in, and your user id is 1 if required for the MCP tool.

        You are provided with TWO functions within <tools></tools> XML tags:
        <tools>
        1. list_tools
    - Description: List all available MCP tools for the current environment to help you finish the user task.
    - Arguments: None
    - Output: A list of MCP environment-specific tools and their descriptions

2. call_tool
    - Description: Call a MCP environment-specific tool
    - Arguments:
        - tool_name: str, required, the tool name in the list_tools output
        - arguments: str, required, the arguments for calling <tool_name>. You must pass a valid JSON string without any markdown fences or additional commentary. This JSON str will be parsed by the tool and executed. You can pass an empty JSON str if no arguments are required by <tool_name>.
    - Output: The result of the <tool_name> tool call
        </tools>

        You should always call list_tools function first to get the available tools, and should only call it once. You should always directly output the answer or summary at the final step instead of calling any function.

        For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
        <tool_call>
        {"name": <function-name>, "arguments": <args-json-object>}
        </tool_call>

        ## Execution Status Assessment

        IMPORTANT: At the beginning of each <think> block, you MUST first declare your execution status using one of these four tags, then provide your reasoning:

        <status>CONTINUE</status> - Current approach is viable, proceed with next step.
        <status>VERIFY</status> - Need to confirm current state or check a precondition before proceeding. Next action must be a read-only query.
        <status>REPLAN</status> - Current approach invalidated by execution feedback. Switch strategy, tool, or subgoal.
        <status>STOP</status> - Task objective satisfied or further execution cannot improve outcome. Output final answer directly (no tool call).

        Example step:
        <think>
        <status>CONTINUE</status>
        The previous query returned the user profile. I have the info I need to update the email.
        </think>

        <tool_call>
        {"name": "call_tool", "arguments": {"tool_name": "update_profile", "arguments": "{\\"email\\": \\"new@example.com\\"}"}}
        </tool_call>

        Example verification step:
        <think>
        <status>VERIFY</status>
        The update returned success, but I should confirm the change was persisted.
        </think>

        <tool_call>
        {"name": "call_tool", "arguments": {"tool_name": "get_profile", "arguments": "{}"}}
        </tool_call>

        Example replan step:
        <think>
        <status>REPLAN</status>
        The endpoint returned 404. I need to find an alternative tool.
        </think>

        <tool_call>
        {"name": "list_tools", "arguments": null}
        </tool_call>

        Example stop step:
        <think>
        <status>STOP</status>
        The task is complete. Email updated and verified.
        </think>

        The task is done. I have successfully updated the email to new@example.com and verified the change.""")


# ════════════════��══════════════════════════════════════════════════════════════
# Data Cache (reused from AWM)
# ════════════════════════════════════��══════════════════════════════════���═══════
@dataclass
class _DataCache:
    db_schemas: dict = field(default_factory=dict)
    sample_data: dict = field(default_factory=dict)
    verifiers: dict = field(default_factory=dict)       # code mode (pure_code)
    sql_verifiers: dict = field(default_factory=dict)    # sql mode (for LLM judge)
    envs_data: dict = field(default_factory=dict)
    predicates: dict = field(default_factory=dict)
    loaded: bool = False

_CACHE = _DataCache()
_load_lock = threading.Lock()


def _load_cache():
    if _CACHE.loaded:
        return
    with _load_lock:
        if _CACHE.loaded:
            return
        cfg = ESA_CONFIGS
        for item in tools_jsonl_load(cfg["db_schema_path"]):
            _CACHE.db_schemas[normalize_scenario_name(item["scenario"])] = item
        for item in tools_jsonl_load(cfg["sample_path"]):
            _CACHE.sample_data[normalize_scenario_name(item["scenario"])] = item
        for item in tools_jsonl_load(cfg["verifier_path"]):
            s = normalize_scenario_name(item["scenario"])
            _CACHE.verifiers[f"{s}::{item['task_idx']}"] = item
        for item in tools_jsonl_load(cfg["envs_path"]):
            _CACHE.envs_data[normalize_scenario_name(item["scenario"])] = item
        # Load SQL verifiers (for LLM judge evidence)
        sql_path = cfg.get("sql_verifier_path", "")
        if os.path.exists(sql_path):
            for item in tools_jsonl_load(sql_path):
                s = normalize_scenario_name(item["scenario"])
                key = f"{s}::{item['task_idx']}"
                raw = item.get("verification", {}).get("raw_response", "")
                code = item.get("verification", {}).get("code", "")
                try:
                    parsed_raw = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    parsed_raw = {}
                _CACHE.sql_verifiers[key] = {
                    "code": code,
                    "function_name": parsed_raw.get("function_name", "verify_task"),
                    "reasoning": parsed_raw.get("reasoning", ""),
                    "success_criteria": parsed_raw.get("success_criteria", ""),
                    "failure_criteria": parsed_raw.get("failure_criteria", ""),
                }
            logger.info("Loaded %d SQL verifiers", len(_CACHE.sql_verifiers))
        # Load predicates
        pred_path = cfg["predicate_path"]
        if os.path.exists(pred_path):
            for item in tools_jsonl_load(pred_path):
                s = normalize_scenario_name(item["scenario"])
                _CACHE.predicates[f"{s}::{item['task_idx']}"] = item
            logger.info("Loaded %d predicate sets", len(_CACHE.predicates))
        else:
            logger.warning("No predicates file at %s — progress reward disabled", pred_path)
        _CACHE.loaded = True
        logger.info("ESA data cache: %d schemas, %d verifiers, %d envs, %d predicates",
                    len(_CACHE.db_schemas), len(_CACHE.verifiers),
                    len(_CACHE.envs_data), len(_CACHE.predicates))


# ═══════════════════════════════════════════════════════════════════════════════
# Template DB (reused from AWM)
# ══════════════════════════════════════════════════════════��════════════════════
_TEMPLATE_DIR = os.path.join(ESA_CONFIGS["db_dir"], "_templates")
_template_lock = asyncio.Lock()


async def _ensure_template(scenario):
    os.makedirs(_TEMPLATE_DIR, exist_ok=True)
    path = os.path.join(_TEMPLATE_DIR, f"{scenario}.db")
    if os.path.exists(path):
        return path
    async with _template_lock:
        if os.path.exists(path):
            return path
        schema = _CACHE.db_schemas.get(scenario)
        if not schema:
            raise RuntimeError(f"No schema for scenario: {scenario}")
        await asyncio.to_thread(_create_template_sync, scenario, schema, path)
    return path


def _create_template_sync(scenario, schema, path):
    db_path, _, _, _ = create_sqlite_database(scenario, schema["db_schema"], os.path.dirname(path))
    sample = _CACHE.sample_data.get(scenario)
    if sample:
        execute_sample_data(db_path, sample["sample_data"], scenario)
    if db_path != path:
        shutil.move(db_path, path)


# ═══════════════════════════════════════════════════════════════════════════════
# Server Pool (reused from AWM)
# ═════════════════════════���═════════════════════════════════════════════════════
@dataclass
class _Slot:
    index: int
    port: int
    scenario: str = ""
    proc: subprocess.Popen | None = None
    db_path: str = ""
    tools_text: str = ""
    mcp: MCPToolExecutor | None = None
    busy: bool = False

_pool: list[_Slot] = []
_pool_lock = asyncio.Lock()
_pool_initialized = False


def _init_pool():
    global _pool, _pool_initialized
    if _pool_initialized:
        return
    base_port = 9100
    db_dir = ESA_CONFIGS["db_dir"]
    os.makedirs(db_dir, exist_ok=True)
    for i in range(ESA_CONFIGS["pool_size"]):
        _pool.append(_Slot(index=i, port=base_port + i,
                           db_path=os.path.join(db_dir, f"slot_{i}.db")))
    _pool_initialized = True
    logger.info("ESA server pool: %d slots, ports %d-%d",
                len(_pool), base_port, base_port + len(_pool) - 1)


async def _acquire_slot(scenario):
    await POOL_SEMAPHORE.acquire()
    async with _pool_lock:
        for slot in _pool:
            if not slot.busy and slot.scenario == scenario:
                slot.busy = True
                return slot
        for slot in _pool:
            if not slot.busy:
                slot.busy = True
                return slot
    POOL_SEMAPHORE.release()
    raise RuntimeError("No free server slots")


def _release_slot(slot):
    slot.busy = False
    POOL_SEMAPHORE.release()


def _kill_port(port):
    try:
        subprocess.run(["fuser", "-k", f"{port}/tcp"],
                        capture_output=True, text=True, timeout=5)
    except Exception:
        pass


def _write_server_script(scenario, db_path, host, port):
    env_item = _CACHE.envs_data.get(scenario)
    if not env_item:
        raise RuntimeError(f"No envs data for scenario: {scenario}")
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
    script_dir = os.path.join("/dev/shm", f"awm_scripts_{os.getuid()}")
    os.makedirs(script_dir, exist_ok=True)
    script_path = os.path.join(script_dir, f"server_{scenario}_{port}.py")
    with open(script_path, "w") as f:
        f.write("\n".join(new_lines))
    return script_path


async def _close_slot_mcp(slot):
    if slot.mcp is not None:
        try:
            await slot.mcp.__aexit__(None, None, None)
        except Exception:
            pass
        slot.mcp = None


async def _start_server(slot, scenario):
    await _close_slot_mcp(slot)
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
    _kill_port(slot.port)

    template_path = await _ensure_template(scenario)
    await asyncio.to_thread(shutil.copy2, template_path, slot.db_path)
    script_path = _write_server_script(scenario, slot.db_path, "127.0.0.1", slot.port)

    log_dir = os.path.join("/dev/shm", f"awm_logs_{os.getuid()}")
    os.makedirs(log_dir, exist_ok=True)
    env = os.environ.copy()
    for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"]:
        env[k] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"

    with open(os.path.join(log_dir, f"server_{slot.index}.log"), "w") as log_f:
        slot.proc = subprocess.Popen([sys.executable, script_path],
                                     stdout=log_f, stderr=log_f, env=env)

    mcp_url = f"http://127.0.0.1:{slot.port}/mcp"
    deadline = time.time() + ESA_CONFIGS["server_startup_timeout"]
    while time.time() < deadline:
        try:
            running, tools_count, _, _ = await check_mcp_server(
                url=mcp_url, timeout=min(2.0, deadline - time.time()))
            if running and tools_count > 0:
                slot.mcp = MCPToolExecutor(mcp_url, timeout=ESA_CONFIGS["tool_timeout"])
                await slot.mcp.__aenter__()
                tools = await slot.mcp.list_tools()
                slot.tools_text = format_tools_for_response(tools)
                slot.scenario = scenario
                return
        except Exception:
            pass
        await asyncio.sleep(0.5)
    raise RuntimeError(f"Server failed to start: slot={slot.index} scenario={scenario}")


async def _ensure_server_ready(slot, scenario):
    if slot.scenario == scenario and slot.proc is not None and slot.proc.poll() is None:
        template_path = await _ensure_template(scenario)
        await asyncio.to_thread(shutil.copy2, template_path, slot.db_path)
        return
    async with _STARTUP_SEMAPHORE:
        await _start_server(slot, scenario)


def _shutdown_pool():
    for slot in _pool:
        if slot.proc is not None:
            try:
                slot.proc.terminate()
                slot.proc.wait(timeout=3)
            except Exception:
                try:
                    slot.proc.kill()
                except Exception:
                    pass
            _kill_port(slot.port)

import atexit
atexit.register(_shutdown_pool)


# ═══════════════════════════════════════════════════════════════════════════════
# Tool execution (reused from AWM)
# ═══════════════════════════════════════════════════════════════════════════════
async def _execute_tool_call(mcp, response_text, tools_text):
    """Execute tool call, return (result_text, done, is_write_op)."""
    tool_calls = parse_tool_calls(response_text)
    if not tool_calls:
        return "", True, False

    tc = tool_calls[0]
    name, arguments = tc["name"], tc["arguments"]

    if name == "list_tools":
        return tools_text, False, False

    if name == "call_tool":
        tool_name, tool_args = parse_call_tool_arguments(arguments)
        is_write = _is_write_operation(tool_name, tool_args)
        try:
            result = await mcp.call_tool(tool_name, tool_args)
            return result, False, is_write
        except asyncio.TimeoutError:
            return f"Error: Tool call timed out after {ESA_CONFIGS['tool_timeout']}s", False, is_write
        except Exception as e:
            return f"Error: {e}", False, is_write

    return f"Error: Unknown tool '{name}'. Use 'list_tools' or 'call_tool'.", False, False


def _is_write_operation(tool_name: str, tool_args: dict) -> bool:
    """Heuristic to detect state-mutating operations."""
    write_keywords = {"create", "add", "insert", "update", "delete", "remove",
                      "post", "put", "patch", "set", "modify", "change",
                      "book", "cancel", "submit", "send", "publish", "edit",
                      "purchase", "buy", "order", "checkout", "pay", "register",
                      "subscribe", "unsubscribe", "assign", "unassign",
                      "approve", "reject", "close", "resolve", "archive",
                      "restore", "enable", "disable", "activate", "deactivate",
                      "reset", "revoke", "grant", "transfer", "move",
                      "upload", "import", "export", "schedule", "complete",
                      "start", "stop", "pause", "resume", "apply",
                      "link", "unlink", "connect", "disconnect",
                      "follow", "unfollow", "block", "unblock", "mute",
                      "pin", "unpin", "save", "unsave", "mark", "flag",
                      "rate", "review", "comment", "reply", "react",
                      "invite", "accept", "decline", "join", "leave",
                      "rename", "configure", "upsert", "merge", "split"}
    name_lower = tool_name.lower()
    for kw in write_keywords:
        if kw in name_lower:
            return True
    # Check HTTP method if present in args
    method = str(tool_args.get("method", "")).upper()
    if method in {"POST", "PUT", "PATCH", "DELETE"}:
        return True
    return False


# ═══════════════════════════════════════════���═══════════════════════════════════
# ESA: Control Block Parsing
# ═══════════════════════════════════════════════��═══════════════════════════════
VALID_STATUSES = {"CONTINUE", "VERIFY", "REPLAN", "STOP"}

_STATUS_PATTERN = re.compile(
    r"(?:<status>)?\s*(CONTINUE|VERIFY|REPLAN|STOP)\s*</status>", re.IGNORECASE
)


def _parse_control_block(response_text: str) -> tuple[str, bool]:
    """Extract execution-status from model output.

    Returns (status_str, is_valid_format).
    If no valid control block found, returns ("CONTINUE", False).
    """
    match = _STATUS_PATTERN.search(response_text)
    if match:
        status = match.group(1).upper()
        if status in VALID_STATUSES:
            return status, True
    return "CONTINUE", False


# ═══════════════════════════════════════════════════════════════════════════════
# ESA: State-Based Progress Reward
# ═══════════════════════════════════════════════════════════════════════════════
def _evaluate_predicates(db_path: str, predicates: list[dict]) -> float:
    """Evaluate predicates against current database state.

    Uses the format from predicates.jsonl (LLM-extracted):
      - sql: self-contained SQL query
      - check_type: "existence" | "count_gte" | "exact_match" | "text_contains"
      - expected: ground truth expected value
      - initial_value: value on initial DB (used to filter tautological predicates)
      - executable: bool

    Only evaluates non-tautological predicates (initial_value != expected).
    Returns Phi(s_t) in [0, 1].
    """
    if not predicates:
        return 0.0

    # Filter to executable, non-tautological predicates only
    # A predicate is tautological if it's already satisfied on the initial DB
    def _is_tautological(p):
        iv = p.get("initial_value")
        ev = p.get("expected")
        ct = p.get("check_type", "existence")
        if ct == "count_gte":
            # count_gte is satisfied if initial_value >= expected
            if iv is not None and ev is not None:
                try:
                    return float(iv) >= float(ev)
                except (TypeError, ValueError):
                    return iv == ev
        # For other types, tautological if initial == expected
        return iv == ev

    useful = [p for p in predicates
              if p.get("executable", True) and not _is_tautological(p)]
    if not useful:
        return 0.0

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
    except Exception:
        return 0.0

    scores = []
    for pred in useful:
        try:
            cursor.execute(pred["sql"])
            result = cursor.fetchone()
            actual = result[0] if result else None

            check_type = pred.get("check_type", "existence")
            expected = pred.get("expected")

            if check_type == "existence":
                # Expected: 1 (exists) or 0 (not exists)
                # actual is typically COUNT(*)>0 returning 0 or 1
                if expected is not None:
                    scores.append(1.0 if actual == expected else 0.0)
                else:
                    scores.append(1.0 if actual else 0.0)

            elif check_type == "count_gte":
                # Expected: minimum count
                if actual is not None and expected is not None:
                    scores.append(1.0 if actual >= expected else 0.0)
                else:
                    scores.append(0.0)

            elif check_type == "exact_match":
                # Expected: exact value (string, int, float)
                if actual is not None and expected is not None:
                    # Handle numeric comparison with tolerance
                    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                        scores.append(1.0 if abs(actual - expected) < 0.01 else 0.0)
                    else:
                        scores.append(1.0 if str(actual) == str(expected) else 0.0)
                else:
                    scores.append(0.0)

            elif check_type == "text_contains":
                # Expected: substring that should be present
                if actual is not None and expected is not None:
                    scores.append(1.0 if str(expected).lower() in str(actual).lower() else 0.0)
                else:
                    scores.append(0.0)

            else:
                # Unknown check_type — fallback to equality
                if expected is not None:
                    scores.append(1.0 if actual == expected else 0.0)
                else:
                    scores.append(1.0 if actual else 0.0)

        except Exception:
            scores.append(0.0)

    conn.close()
    return sum(scores) / len(scores) if scores else 0.0


def _evaluate_predicate_signature(db_path: str, predicates: list[dict]) -> tuple[float, tuple]:
    """Evaluate predicates and return both Phi score and binary signature.

    Returns:
        (phi, signature) where phi is in [0,1] and signature is a tuple of 0/1
        indicating which non-tautological predicates are satisfied.
        signature can be used as a hashable key for grouping.
    """
    if not predicates:
        return 0.0, ()

    def _is_tautological(p):
        iv = p.get("initial_value")
        ev = p.get("expected")
        ct = p.get("check_type", "existence")
        if ct == "count_gte":
            if iv is not None and ev is not None:
                try:
                    return float(iv) >= float(ev)
                except (TypeError, ValueError):
                    return iv == ev
        return iv == ev

    useful = [p for p in predicates
              if p.get("executable", True) and not _is_tautological(p)]
    if not useful:
        return 0.0, ()

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
    except Exception:
        return 0.0, tuple(0 for _ in useful)

    bits = []
    for pred in useful:
        try:
            cursor.execute(pred["sql"])
            result = cursor.fetchone()
            actual = result[0] if result else None
            check_type = pred.get("check_type", "existence")
            expected = pred.get("expected")

            satisfied = False
            if check_type == "existence":
                satisfied = (actual == expected) if expected is not None else bool(actual)
            elif check_type == "count_gte":
                satisfied = (actual is not None and expected is not None and actual >= expected)
            elif check_type == "exact_match":
                if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                    satisfied = abs(actual - expected) < 0.01
                else:
                    satisfied = str(actual) == str(expected) if actual is not None else False
            elif check_type == "text_contains":
                satisfied = (str(expected).lower() in str(actual).lower()) if actual and expected else False
            else:
                satisfied = (actual == expected) if expected is not None else bool(actual)
            bits.append(1 if satisfied else 0)
        except Exception:
            bits.append(0)

    conn.close()
    phi = sum(bits) / len(bits) if bits else 0.0
    return phi, tuple(bits)


def _compute_progress_reward(
    db_path: str,
    prev_phi: float,
    predicate_sqls: list[dict],
) -> tuple[float, float, tuple]:
    """Compute R_prog = Phi(s_{t+1}) - Phi(s_t).

    Returns (R_prog, current_phi, current_signature).
    """
    current_phi, signature = _evaluate_predicate_signature(db_path, predicate_sqls)
    r_prog = current_phi - prev_phi
    return r_prog, current_phi, signature


# ═══════════════════════════════════════════════════════════════════════════════
# ESA: Compute Turn-Level Return-to-Go
# ════════════════��══════════════════════════════════════════════════════════════
def _compute_turn_level_returns(
    turn_rewards: list[float],
    outcome_reward: float,
    eta: float,
    lambda_prog: float,
) -> list[float]:
    """Compute discounted return-to-go per turn.

    r^(t) = lambda * R_prog^(t) + gamma * R_verify^(t) + 1[t=T] * R_out
    G_{t} = sum_{u=t}^{T} eta^{u-t} * r^(u)

    Returns list of G_t values, one per turn.
    """
    T = len(turn_rewards)
    if T == 0:
        return []

    # The outcome reward is added to the last turn
    step_rewards = [lambda_prog * r for r in turn_rewards]
    step_rewards[-1] += outcome_reward

    # Compute discounted return-to-go from back to front
    returns = [0.0] * T
    returns[T - 1] = step_rewards[T - 1]
    for t in range(T - 2, -1, -1):
        returns[t] = step_rewards[t] + eta * returns[t + 1]

    return returns


# ═══════════════════════════════════════════════════════════��═══════════════════
# Verification / Reward (from AWM)
# ═══════════════════════════════════════════════════════════════════════════════
def _run_sql_verifier(scenario, task_idx, initial_db, final_db):
    """Run SQL mode verifier to collect evidence for LLM judge."""
    key = f"{scenario}::{task_idx}"
    sv = _CACHE.sql_verifiers.get(key)
    if not sv or not sv["code"]:
        return {"status": "no_verifier"}

    try:
        result = execute_verification_code(
            python_code=sv["code"],
            function_name=sv["function_name"],
            initial_db_path=initial_db,
            mode=VerificationMode.sql,
            final_db_path=final_db,
        )
        return {
            "status": "executed",
            "results": result.get("result", {}),
            "success_criteria": sv["success_criteria"],
            "failure_criteria": sv["failure_criteria"],
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "success_criteria": sv.get("success_criteria", ""),
            "failure_criteria": sv.get("failure_criteria", ""),
        }


_JUDGE_PROMPT = """You are an impartial evaluator for tool-use agent task results. Based on the agent trajectory AND database verification evidence, classify the outcome.

Categories:
- Completed: task done, database confirms it.
- Environment Error: agent blocked by API bugs (500, 422 for valid params, route conflicts).
- Agent Error: agent made mistakes (wrong params, hallucination, incomplete).

Output valid JSON only:
{"reasoning": "<brief>", "classification": "<Completed | Environment Error | Agent Error>", "confidence": <0-1>}"""


async def _compute_outcome_reward(scenario, task_idx, initial_db, final_db, final_answer, trajectory):
    """Code-augmented LLM-as-a-Judge outcome reward.

    Step 1: Run SQL verifier to collect DB evidence
    Step 2: Send evidence + trajectory to LLM judge
    Step 3: Map classification to reward
    """
    cfg = ESA_CONFIGS

    # Step 1: SQL verifier evidence
    verification = await asyncio.to_thread(
        _run_sql_verifier, scenario, task_idx, initial_db, final_db)

    # Build evidence string
    if verification["status"] == "executed":
        results_str = json.dumps(verification.get("results", {}), indent=2, default=str)
        if len(results_str) > 2000:
            results_str = results_str[:2000] + "\n... [truncated]"
        evidence = (
            f"Database evidence:\n{results_str}\n\n"
            f"Success criteria: {verification.get('success_criteria', 'N/A')}\n"
            f"Failure criteria: {verification.get('failure_criteria', 'N/A')}"
        )
    else:
        evidence = f"Verification failed: {verification.get('error', 'no verifier')}"

    # Truncate trajectory
    traj = trajectory
    if len(traj) > 4000:
        traj = traj[:1500] + "\n\n... [truncated] ...\n\n" + traj[-2500:]

    user_msg = f"Task: {_CACHE.sql_verifiers.get(f'{scenario}::{task_idx}', {}).get('reasoning', '')}\n\nTrajectory:\n{traj}\n\n{evidence}"

    # Step 2: Call LLM judge
    import aiohttp
    payload = {
        "model": "",  # will be filled by first /v1/models call
        "messages": [
            {"role": "system", "content": _JUDGE_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.1,
        "max_tokens": 2000,
        "chat_template_kwargs": {"enable_thinking": True},
    }

    try:
        async with aiohttp.ClientSession() as session:
            # Get model name
            async with session.get(f"{cfg['judge_url']}/v1/models") as resp:
                models = await resp.json()
            payload["model"] = models["data"][0]["id"]

            async with session.post(
                f"{cfg['judge_url']}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                result = await resp.json()

        raw = result["choices"][0]["message"]["content"]

        # Parse JSON from response (skip thinking)
        json_text = raw.strip()
        last_brace = json_text.rfind("{")
        if last_brace >= 0:
            depth = 0
            for idx in range(last_brace, len(json_text)):
                if json_text[idx] == "{": depth += 1
                elif json_text[idx] == "}":
                    depth -= 1
                    if depth == 0:
                        json_text = json_text[last_brace:idx+1]
                        break

        parsed = json.loads(json_text)
        classification = parsed.get("classification", "Agent Error")

        reward_map = {"Completed": 1.0, "Environment Error": 0.0, "Agent Error": 0.0}
        return reward_map.get(classification, 0.0)

    except Exception as e:
        logger.error("LLM judge FAILED for %s::%d: %s", scenario, task_idx, e)
        raise RuntimeError(f"LLM judge failed for {scenario}::{task_idx}: {e}. "
                          f"Check judge server at {cfg['judge_url']}")


# ═══════════════════════════════════════════════════════════════════════════════
# generate() — the main ESA rollout function
# ═══════════════════════════════════════════════════════════════════════════════
async def generate(args, sample: Sample, sampling_params) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for ESA."

    await asyncio.to_thread(_load_cache)
    _init_pool()

    state = GenerateState(args)
    tokenizer = state.tokenizer
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    metadata = sample.metadata or {}
    scenario = normalize_scenario_name(metadata["scenario"])
    task_idx = metadata["task_idx"]

    prompt_text = sample.prompt
    prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    cfg = ESA_CONFIGS
    pred_key = f"{scenario}::{task_idx}"
    predicate_sqls = _CACHE.predicates.get(pred_key, {}).get("predicates", [])
    has_predicates = len(predicate_sqls) > 0

    max_retries = 10  # retry aggressively — infra failures are transient
    retry_timeout = 120.0  # hard cap: give up after 2 minutes of retrying
    last_error = None
    retry_start = time.monotonic()
    rollout_success = False

    for attempt in range(max_retries):
        # Reset all mutable state for each attempt
        response = ""
        response_token_ids = []
        loss_mask = []
        rollout_log_probs = [] if cfg["return_logprob"] else None
        turn_rewards = []
        turn_token_counts = []
        turn_signatures = []
        current_turn = 0
        prev_phi = 0.0
        prev_signature = ()
        prev_was_error = False
        format_error = False
        prev_tool_call = None
        sample.status = Sample.Status.PENDING

        slot = await _acquire_slot(scenario)

        try:
            await _ensure_server_ready(slot, scenario)

            initial_db = slot.db_path + ".init"
            await asyncio.to_thread(shutil.copy2, slot.db_path, initial_db)

            # Evaluate initial Phi(s_0) and signature
            if has_predicates:
                prev_phi, prev_signature = await asyncio.to_thread(
                    _evaluate_predicate_signature, initial_db, predicate_sqls)

            mcp = slot.mcp
            final_answer = ""
            max_ctx = getattr(args, "rollout_max_context_len", None) or 32768
            max_new = sampling_params.get("max_new_tokens", 2048)

            for _turn in range(cfg["max_turns"]):
                total_tokens = len(prompt_token_ids) + len(response_token_ids)
                remaining = max_ctx - total_tokens
                if remaining <= 0:
                    sample.status = Sample.Status.TRUNCATED
                    break

                cur_sampling_params = sampling_params.copy()
                cur_sampling_params["max_new_tokens"] = min(max_new, remaining)

                payload = {"text": prompt_text + response,
                           "sampling_params": cur_sampling_params}
                if cfg["return_logprob"]:
                    payload["return_logprob"] = True

                output = await post(url, payload)

                if output["meta_info"]["finish_reason"]["type"] == "abort":
                    sample.status = Sample.Status.ABORTED
                    break

                cur_response = output["text"]

                if cfg["return_logprob"]:
                    cur_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
                    cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
                else:
                    cur_token_ids = tokenizer(cur_response, add_special_tokens=False)["input_ids"]

                # ── ESA: Parse control block ──
                control_status, valid_format = _parse_control_block(cur_response)
                current_turn += 1
                turn_token_count = len(cur_token_ids)

                # Encode turn index in loss_mask (>0 means trainable, value = turn index)
                if cfg["enable_turn_level"]:
                    loss_mask += [current_turn] * turn_token_count
                else:
                    loss_mask += [1] * turn_token_count

                response += cur_response
                response_token_ids += cur_token_ids
                if rollout_log_probs is not None:
                    rollout_log_probs += cur_log_probs

                if output["meta_info"]["finish_reason"]["type"] == "length":
                    turn_signatures.append(prev_signature)
                    turn_rewards.append(0.0)
                    turn_token_counts.append(turn_token_count)
                    sample.status = Sample.Status.TRUNCATED
                    break

                # ── ESA: Reward/penalize <status> usage ──
                if valid_format:
                    turn_format_bonus = cfg["format_correct_bonus"]
                elif current_turn <= 1:
                    turn_format_bonus = 0.0  # exempt first turn
                else:
                    turn_format_bonus = -cfg["format_correct_bonus"]

                # ── ESA: Handle STOP ──
                if control_status == "STOP":
                    final_answer = cur_response
                    turn_signatures.append(prev_signature)
                    turn_rewards.append(turn_format_bonus)
                    turn_token_counts.append(turn_token_count)
                    sample.status = Sample.Status.COMPLETED
                    break

                # ── AWM-style format validation (strict) ──
                tool_calls = parse_tool_calls(cur_response)
                if not tool_calls:
                    turn_signatures.append(prev_signature)
                    turn_rewards.append(cfg["format_error_penalty"])
                    turn_token_counts.append(turn_token_count)
                    format_error = True
                    sample.status = Sample.Status.COMPLETED
                    break

                tc = tool_calls[0]
                tc_name = tc.get("name", "")

                if tc_name not in ("list_tools", "call_tool"):
                    turn_signatures.append(prev_signature)
                    turn_rewards.append(cfg["format_error_penalty"])
                    turn_token_counts.append(turn_token_count)
                    format_error = True
                    sample.status = Sample.Status.COMPLETED
                    break

                if tc_name == "call_tool":
                    try:
                        tool_name, tool_args = parse_call_tool_arguments(tc["arguments"])
                        if not tool_name:
                            raise ValueError("empty tool_name")
                    except Exception:
                        turn_signatures.append(prev_signature)
                        turn_rewards.append(cfg["format_error_penalty"])
                        turn_token_counts.append(turn_token_count)
                        format_error = True
                        sample.status = Sample.Status.COMPLETED
                        break

                # ── Detect duplicate tool call ──
                current_tool_call = json.dumps(tc, sort_keys=True)
                is_duplicate = (current_tool_call == prev_tool_call)
                prev_tool_call = current_tool_call

                # ── Execute tool call ──
                obs_text, done, is_write = await _execute_tool_call(
                    mcp, cur_response, slot.tools_text)

                is_env_error = obs_text.startswith("Error:") and any(
                    kw in obs_text for kw in ["Internal Server Error", "500", "timed out"])

                # ── ESA: Record signature BEFORE action ──
                turn_signatures.append(prev_signature)

                # ── ESA: Compute per-turn reward ──
                turn_reward = turn_format_bonus

                if is_env_error:
                    prev_was_error = False
                elif has_predicates:
                    r_prog, new_phi, new_sig = await asyncio.to_thread(
                        _compute_progress_reward, slot.db_path, prev_phi, predicate_sqls)
                    turn_reward += r_prog
                    prev_was_error = (r_prog < 0) or ("Error" in obs_text)
                    prev_phi = new_phi
                    prev_signature = new_sig
                else:
                    prev_was_error = "Error" in obs_text

                if is_duplicate:
                    turn_reward += cfg["duplicate_penalty"]

                if control_status == "VERIFY" and is_write:
                    turn_reward += cfg["verify_violation_penalty"]

                turn_rewards.append(turn_reward)
                turn_token_counts.append(turn_token_count)

                if done:
                    final_answer = cur_response
                    sample.status = Sample.Status.COMPLETED
                    break

                # Append observation (loss_mask=0, no gradient)
                next_obs = (
                    f"<|im_start|>user\n<tool_response>\n{obs_text}\n</tool_response><|im_end|>\n"
                    f"<|im_start|>assistant\n<think>\n<status>"
                )
                obs_token_ids = tokenizer(next_obs, add_special_tokens=False)["input_ids"]

                response += next_obs
                response_token_ids += obs_token_ids
                loss_mask += [0] * len(obs_token_ids)
                if rollout_log_probs is not None:
                    rollout_log_probs += [0.0] * len(obs_token_ids)

                if len(prompt_token_ids) + len(response_token_ids) + 1 >= max_ctx:
                    sample.status = Sample.Status.TRUNCATED
                    break

            # ── Compute rewards ──
            if sample.status == Sample.Status.PENDING:
                sample.status = Sample.Status.COMPLETED

            if format_error:
                outcome_reward = cfg["format_error_penalty"]
            else:
                outcome_reward = await _compute_outcome_reward(
                    scenario, task_idx, initial_db, slot.db_path, final_answer, response)

            eta = cfg["eta"]
            lam = cfg["lambda_prog"]

            if turn_rewards:
                turn_returns = _compute_turn_level_returns(
                    turn_rewards, outcome_reward, eta, lam)
                composite_reward = turn_returns[0] if turn_returns else outcome_reward
            else:
                composite_reward = outcome_reward
                turn_returns = []

            sample.reward = composite_reward
            sample.metadata["turn_rewards"] = turn_rewards
            sample.metadata["turn_token_counts"] = turn_token_counts
            sample.metadata["turn_returns"] = turn_returns
            sample.metadata["turn_signatures"] = [list(s) for s in turn_signatures]
            sample.metadata["outcome_reward"] = outcome_reward

            rollout_success = True
            break  # Success, exit retry loop

        except Exception as e:
            last_error = e
            elapsed = time.monotonic() - retry_start
            logger.warning(
                "ESA rollout attempt %d/%d failed for %s[%d] (%.1fs elapsed): %s",
                attempt + 1, max_retries, scenario, task_idx, elapsed, e,
            )
            if elapsed >= retry_timeout:
                break  # timeout, give up
            if attempt < max_retries - 1:
                await asyncio.sleep(min(3, 1 + attempt))  # backoff: 1s, 2s, 3s, ...

        finally:
            _release_slot(slot)
            initial_db = slot.db_path + ".init"
            if os.path.exists(initial_db):
                try:
                    os.remove(initial_db)
                except OSError:
                    pass

    if not rollout_success:
        logger.error(
            "ESA rollout failed after %d attempts (%.1fs) for %s[%d]: %s",
            attempt + 1, time.monotonic() - retry_start,
            scenario, task_idx, last_error, exc_info=True,
        )
        sample.reward = 0.0
        sample.remove_sample = True
        sample.metadata["infra_error"] = f"{type(last_error).__name__}: {last_error}"
        sample.status = Sample.Status.FAILED

    # Fill sample (covers both success and failure paths)
    sample.tokens = prompt_token_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_mask
    sample.prompt = prompt_text
    if rollout_log_probs is not None:
        sample.rollout_log_probs = rollout_log_probs

    return sample


# ═══════════════════════════════════════════════════════════════════════════════
# Official Rollout Logging (via --custom-rollout-log-function-path)
#
# Called by the SLIME driver process after all rollout samples are collected.
# This runs on the main node, not inside Ray workers, so file I/O is reliable.
# ═══════════════════════════════════════════════════════════════════════════════
_ROLLOUT_LOG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "logs",
    time.strftime("%Y%m%d_%H%M%S")
)


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """Custom rollout log function (--custom-rollout-log-function-path).

    Saves per-sample traces with full reward decomposition to JSONL files.
    Returns False so SLIME's default WandB/tensorboard logging also runs.
    """
    log_dir = _ROLLOUT_LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    filename = os.path.join(log_dir, f"rollout_{rollout_id:04d}.jsonl")
    with open(filename, "w") as f:
        for sample in samples:
            meta = sample.metadata or {}
            turn_rewards = meta.get("turn_rewards", [])
            turn_returns = meta.get("turn_returns", [])
            turn_token_counts = meta.get("turn_token_counts", [])
            turn_signatures = meta.get("turn_signatures", [])
            outcome_reward = meta.get("outcome_reward")

            record = {
                # ── Identity ──
                "scenario": meta.get("scenario", ""),
                "task_idx": meta.get("task_idx", ""),
                "task": meta.get("task", ""),
                # ── Rewards ──
                "composite_reward": sample.reward,
                "outcome_reward": outcome_reward,
                # ── Per-turn reward decomposition ──
                "turn_rewards": turn_rewards,
                "turn_returns": turn_returns,
                "turn_token_counts": turn_token_counts,
                "turn_signatures": turn_signatures,
                # ── Status ──
                "status": sample.status.value if sample.status else None,
                "infra_error": meta.get("infra_error"),  # None if no infra failure
                "response_length": sample.response_length,
                "num_turns": len(turn_rewards),
                # ── Full trace (prompt + response = complete conversation) ──
                "prompt": sample.prompt or "",
                "response": sample.response or "",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Log summary to console
    rewards = [s.reward for s in samples if s.reward is not None]
    outcomes = [s.metadata.get("outcome_reward", 0) for s in samples if s.metadata]
    n_completed = sum(1 for o in outcomes if o == 1.0)
    n_format_err = sum(1 for o in outcomes if o == ESA_CONFIGS["format_error_penalty"])
    n_infra_err = sum(1 for s in samples if s.metadata and s.metadata.get("infra_error"))
    logger.info(
        "ESA rollout %d: %d samples, reward=%.3f, completion=%d/%d, format_err=%d, "
        "infra_err=%d, time=%.1fs -> %s",
        rollout_id, len(samples),
        sum(rewards) / max(1, len(rewards)),
        n_completed, len(samples), n_format_err,
        n_infra_err, rollout_time, filename,
    )
    return False  # Continue with default WandB logging


# ═══════════════════════════════════════════════════════════════════════════════
# reward_func() — Group-aware with predicate-anchored advantage
#
# When called with --group-rm, receives all rollouts for one task (list[Sample]).
# Computes predicate-anchored step-level advantage and encodes it into the
# scalar reward for each sample. Standard GRPO then does trajectory-level
# normalization on top.
# ═══════════════════════════════════════════════════════════════════════════════
def _compute_predicate_anchored_rewards(samples: list) -> list[float]:
    """Compute rewards with predicate-anchored step-level advantage.

    For each (rollout, turn), we know:
      - The predicate signature σ at that turn (state before action)
      - The return-to-go G from that turn onward

    We group all (rollout, turn) pairs by signature σ, compute group-relative
    advantage within each group, then aggregate per-rollout as:
      adjusted_reward = outcome_reward + ω * mean(step_advantages)

    This encodes step-level credit into the scalar reward that GRPO sees.
    """
    cfg = ESA_CONFIGS
    omega = cfg.get("omega_step", 0.3)  # weight for step-level component
    eta = cfg["eta"]
    lam = cfg["lambda_prog"]

    # Collect all (rollout_idx, turn_idx, signature, return_to_go) tuples
    all_entries = []  # (rollout_idx, turn_idx, signature_key, G_value)

    for i, sample in enumerate(samples):
        meta = sample.metadata or {}
        turn_rewards = meta.get("turn_rewards", [])
        turn_sigs = meta.get("turn_signatures", [])
        outcome = meta.get("outcome_reward", 0.0)

        if not turn_rewards:
            continue

        # Compute return-to-go per turn
        T = len(turn_rewards)
        step_r = [lam * r for r in turn_rewards]
        step_r[-1] += outcome

        rtg = [0.0] * T
        rtg[T - 1] = step_r[T - 1]
        for t in range(T - 2, -1, -1):
            rtg[t] = step_r[t] + eta * rtg[t + 1]

        for t in range(T):
            sig = tuple(turn_sigs[t]) if t < len(turn_sigs) else ()
            all_entries.append((i, t, sig, rtg[t]))

    if not all_entries:
        return [s.reward if s.reward is not None else 0.0 for s in samples]

    # Group by signature
    from collections import defaultdict
    sig_groups = defaultdict(list)  # signature → [(rollout_idx, turn_idx, G)]
    for rollout_idx, turn_idx, sig, g_val in all_entries:
        sig_groups[sig].append((rollout_idx, turn_idx, g_val))

    # Compute group-relative advantage within each signature group
    step_advantages = {}  # (rollout_idx, turn_idx) → advantage
    for sig, entries in sig_groups.items():
        if len(entries) < 2:
            # Can't normalize with < 2 samples; step advantage = 0
            for ri, ti, g in entries:
                step_advantages[(ri, ti)] = 0.0
            continue

        g_values = [g for _, _, g in entries]
        mu = sum(g_values) / len(g_values)
        var = sum((g - mu) ** 2 for g in g_values) / len(g_values)
        std = var ** 0.5 + 1e-8

        for ri, ti, g in entries:
            step_advantages[(ri, ti)] = (g - mu) / std

    # Aggregate per-rollout: mean of step advantages
    rollout_step_advs = defaultdict(list)
    for (ri, ti), adv in step_advantages.items():
        rollout_step_advs[ri].append(adv)

    # Final reward = outcome + omega * mean(step_advantages)
    adjusted_rewards = []
    for i, sample in enumerate(samples):
        outcome = (sample.metadata or {}).get("outcome_reward", 0.0)
        if outcome is None:
            outcome = 0.0
        step_advs = rollout_step_advs.get(i, [])
        mean_step_adv = sum(step_advs) / len(step_advs) if step_advs else 0.0
        adjusted = outcome + omega * mean_step_adv
        adjusted_rewards.append(adjusted)

    return adjusted_rewards


async def reward_func(args, samples_or_sample, **kwargs):
    """Reward function supporting both single-sample and group modes.

    When --group-rm is enabled, receives list[Sample] for one task group.
    Otherwise receives a single Sample.
    """
    if isinstance(samples_or_sample, list):
        # Group mode: predicate-anchored advantage
        samples = samples_or_sample
        rewards = _compute_predicate_anchored_rewards(samples)

        # Neutralize infra-failed samples: set their reward to the group mean
        # so they don't distort GRPO group normalization (their loss_mask is
        # empty anyway, so no gradient flows through them).
        valid_mask = [s.metadata.get("infra_error") is None for s in samples]
        valid_rewards = [r for r, v in zip(rewards, valid_mask) if v]
        if valid_rewards and not all(valid_mask):
            group_mean = sum(valid_rewards) / len(valid_rewards)
            rewards = [r if v else group_mean for r, v in zip(rewards, valid_mask)]

        return rewards
    else:
        # Single sample mode: return pre-computed reward
        sample = samples_or_sample
        if sample.reward is not None:
            return sample.reward
        return 0.0
