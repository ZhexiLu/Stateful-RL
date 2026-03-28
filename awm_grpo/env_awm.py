"""
AWM Environment wrapper for slime multi-turn rollout.

Wraps AWM's MCP tool execution into slime's BaseInteractionEnv interface.
Each env instance manages its own MCP server subprocess, DB reset, and tool calling.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import re
from dataclasses import dataclass, field
from typing import Any

try:
    from slime.utils.types import Sample
except ImportError:
    Sample = None  # Allow standalone usage without slime

# ── AWM imports (add agent-world-model to sys.path) ──────────────────────────
AWM_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent-world-model")
if AWM_ROOT not in sys.path:
    sys.path.insert(0, AWM_ROOT)

from awm.tools import (
    tools_robust_json_loads,
    normalize_scenario_name,
    check_mcp_server,
)
from awm.core.agent import (
    MCPToolExecutor,
    get_system_prompt,
    parse_tool_calls,
    parse_call_tool_arguments,
    format_tools_for_response,
)
from awm.core.db import create_sqlite_database
from awm.core.sample import execute_sample_data
from awm.core.verifier import execute_verification_code, VerificationMode


# ── Port management ──────────────────────────────────────────────────────────

class PortPool:
    """Thread-safe port pool for MCP servers."""

    def __init__(self, base_port: int = 9000, pool_size: int = 256):
        import threading
        self._lock = threading.Lock()
        self._available = list(range(base_port, base_port + pool_size))
        self._in_use: set[int] = set()

    def acquire(self) -> int:
        with self._lock:
            # Try to find a port that's not in use by the OS
            for _ in range(len(self._available)):
                port = self._available.pop(0)
                if not _is_port_open(port):
                    self._in_use.add(port)
                    return port
                self._available.append(port)
            raise RuntimeError("No available ports in pool")

    def release(self, port: int):
        with self._lock:
            self._in_use.discard(port)
            self._available.append(port)


# Global port pool (initialized lazily)
_PORT_POOL: PortPool | None = None


def get_port_pool(base_port: int = 9000, pool_size: int = 256) -> PortPool:
    global _PORT_POOL
    if _PORT_POOL is None:
        _PORT_POOL = PortPool(base_port, pool_size)
    return _PORT_POOL


def _is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.1)
        return sock.connect_ex((host, port)) == 0


def _kill_port(port: int):
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"], capture_output=True, text=True, timeout=5,
        )
        for pid in result.stdout.strip().split():
            if pid:
                os.kill(int(pid), signal.SIGKILL)
    except Exception:
        pass


# ── Global data cache ────────────────────────────────────────────────────────

@dataclass
class AWMDataCache:
    """Holds preloaded AWM data shared across all env instances."""
    db_schemas: dict[str, dict] = field(default_factory=dict)
    sample_data: dict[str, dict] = field(default_factory=dict)
    verifiers: dict[str, dict] = field(default_factory=dict)
    envs_data: dict[str, dict] = field(default_factory=dict)   # full_code per scenario
    envs_path: str = ""
    db_dir: str = ""
    template_db_dir: str = ""
    template_snapshots: dict[str, str] = field(default_factory=dict)
    loaded: bool = False


_DATA_CACHE = AWMDataCache()
_TEMPLATE_DB_LOCK = threading.Lock()


_THREAD_LIMIT_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


def preload_awm_data(
    db_schema_path: str,
    sample_path: str,
    verifier_path: str,
    envs_path: str,
    db_dir: str,
):
    """Load AWM data once into global cache. Call before creating any envs."""
    from awm.tools import tools_jsonl_load

    if _DATA_CACHE.loaded:
        return

    for item in tools_jsonl_load(db_schema_path):
        _DATA_CACHE.db_schemas[normalize_scenario_name(item["scenario"])] = item
    for item in tools_jsonl_load(sample_path):
        _DATA_CACHE.sample_data[normalize_scenario_name(item["scenario"])] = item
    for item in tools_jsonl_load(verifier_path):
        scenario = normalize_scenario_name(item["scenario"])
        task_idx = item["task_idx"]
        _DATA_CACHE.verifiers[f"{scenario}::{task_idx}"] = item

    # Cache envs data (full_code) so server subprocesses don't re-read the 100MB file
    for item in tools_jsonl_load(envs_path):
        _DATA_CACHE.envs_data[normalize_scenario_name(item["scenario"])] = item

    _DATA_CACHE.envs_path = envs_path
    _DATA_CACHE.db_dir = db_dir
    _DATA_CACHE.template_db_dir = os.path.join(db_dir, "_templates")
    if os.path.isdir(_DATA_CACHE.template_db_dir):
        for filename in os.listdir(_DATA_CACHE.template_db_dir):
            if not filename.endswith("_template.db"):
                continue
            scenario = filename[: -len("_template.db")]
            _DATA_CACHE.template_snapshots[scenario] = os.path.join(_DATA_CACHE.template_db_dir, filename)
    _DATA_CACHE.loaded = True


def _ensure_template_db(scenario: str) -> str:
    template_path = _DATA_CACHE.template_snapshots.get(scenario)
    if template_path and os.path.exists(template_path):
        return template_path

    with _TEMPLATE_DB_LOCK:
        template_path = _DATA_CACHE.template_snapshots.get(scenario)
        if template_path and os.path.exists(template_path):
            return template_path

        schema_item = _DATA_CACHE.db_schemas.get(scenario)
        if schema_item is None:
            raise RuntimeError(f"Schema not found for {scenario}")

        os.makedirs(_DATA_CACHE.template_db_dir, exist_ok=True)
        template_name = f"{scenario}__template"
        template_filename = f"{normalize_scenario_name(template_name)}.db"
        template_path = os.path.join(_DATA_CACHE.template_db_dir, template_filename)

        if not os.path.exists(template_path):
            template_path, _, _, _ = create_sqlite_database(
                template_name, schema_item["db_schema"], _DATA_CACHE.template_db_dir
            )
            sample_item = _DATA_CACHE.sample_data.get(scenario)
            if sample_item:
                execute_sample_data(template_path, sample_item["sample_data"], template_name)

        _DATA_CACHE.template_snapshots[scenario] = template_path
        return template_path


def _reset_db_file(scenario: str, db_path: str) -> str:
    """Reset a scenario database file to its initial state."""
    db_dir = os.path.dirname(db_path)
    os.makedirs(db_dir, exist_ok=True)
    template_path = _ensure_template_db(scenario)
    shutil.copy2(template_path, db_path)
    return db_path


# ── Pre-fork server launcher ──────────────────────────────────────────────

class _ServerLauncher:
    """Persistent launcher that pre-imports heavy Python modules once,
    then fork()s children that inherit them instantly (~0s vs ~4s)."""

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def _ensure_started(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        launcher_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "_server_launcher.py",
        )
        self._proc = subprocess.Popen(
            [sys.executable, launcher_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        line = self._proc.stdout.readline()
        resp = json.loads(line)
        if resp.get("status") != "ready":
            raise RuntimeError(f"Launcher failed to start: {resp}")

    def start_server(
        self, script_path: str, log_path: str, env_vars: dict | None = None,
    ) -> int:
        """Fork a child to run *script_path*.  Returns child PID."""
        with self._lock:
            self._ensure_started()
            self._proc.stdin.write(json.dumps({
                "script": script_path,
                "log": log_path,
                "env": env_vars or {},
            }) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
            resp = json.loads(line)
            if "error" in resp:
                raise RuntimeError(f"Launcher error: {resp['error']}")
            return resp["pid"]

    def shutdown(self):
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None


class _PidProcess:
    """Thin wrapper around a PID to match subprocess.Popen interface."""

    def __init__(self, pid: int):
        self.pid = pid
        self.returncode = None
        self._log_handle = None
        self._log_path = None

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        try:
            os.kill(self.pid, 0)
            return None
        except ProcessLookupError:
            self.returncode = 0
            return 0

    def kill(self):
        for sig_func, sig in [(os.killpg, signal.SIGKILL), (os.kill, signal.SIGKILL)]:
            try:
                sig_func(self.pid, sig)
                return
            except (ProcessLookupError, PermissionError):
                continue

    def terminate(self):
        for sig_func, sig in [(os.killpg, signal.SIGTERM), (os.kill, signal.SIGTERM)]:
            try:
                sig_func(self.pid, sig)
                return
            except (ProcessLookupError, PermissionError):
                continue

    def wait(self, timeout=None):
        deadline = time.time() + (timeout if timeout is not None else 3600)
        while time.time() < deadline:
            if self.poll() is not None:
                return self.returncode
            time.sleep(0.05)
        raise subprocess.TimeoutExpired(cmd="server", timeout=timeout)


_LAUNCHER: _ServerLauncher | None = None


def get_server_launcher() -> _ServerLauncher:
    global _LAUNCHER
    if _LAUNCHER is None:
        _LAUNCHER = _ServerLauncher()
    return _LAUNCHER


# ── Server pool: reuse MCP servers across tasks of the same scenario ─────────

def _write_server_script(scenario: str, db_path: str, host: str, port: int) -> str:
    """Generate a temp server script from cached envs data.

    Avoids the 100MB gen_envs.jsonl read per subprocess by using
    data already loaded into _DATA_CACHE.  Uses a unique filename
    per port to prevent race conditions between concurrent starts.
    """
    import textwrap

    env_item = _DATA_CACHE.envs_data.get(scenario)
    if env_item is None:
        raise RuntimeError(f"No cached envs data for scenario: {scenario}")

    code = env_item["full_code"]
    new_code = [
        "import warnings",
        'warnings.filterwarnings("ignore", category=DeprecationWarning)',
        "from sqlalchemy.pool import NullPool",
    ]
    for line in code.split("\n"):
        if "create_engine(" in line:
            left = line.split("create_engine(")[0]
            sql_path = f"'sqlite:///{db_path}'"
            right = (
                f"create_engine({sql_path}, "
                f"connect_args={{'check_same_thread': False}}, poolclass=NullPool)"
            )
            line = f"{left}{right}"

        if "uvicorn.run(app" in line:
            mcp_setup = textwrap.dedent(f"""\
                import os
                host = os.environ.get('HOST', '{host}')
                port = os.environ.get('PORT', {port})
                print(f'Server starting on port={{port}}')
                from fastapi_mcp import FastApiMCP
                mcp = FastApiMCP(app)
                mcp.mount_http()
                print("MCP server enabled on http://{host}:{port}/mcp")
            """)
            new_code.extend(
                textwrap.indent(mcp_setup, "    ").rstrip().split("\n")
            )
            line = "    uvicorn.run(app, host=host, port=int(port))"

        new_code.append(line)

    # Use /dev/shm (RAM) for temp scripts — avoids NFS metadata latency
    script_dir = os.path.join("/dev/shm", f"awm_server_scripts_{os.getuid()}")
    os.makedirs(script_dir, exist_ok=True)
    script_path = os.path.join(script_dir, f"temp_server_{scenario}_{port}.py")
    with open(script_path, "w") as f:
        f.write("\n".join(new_code))

    return script_path


@dataclass
class _ServerSlot:
    """A running MCP server for a specific scenario."""
    scenario: str
    port: int
    db_path: str
    proc: subprocess.Popen
    mcp_url: str
    tools_text: str  # cached list_tools output (same per scenario)
    script_path: str = ""
    last_used: float = 0.0


class ServerPool:
    """
    Thread-safe pool of MCP servers, keyed by scenario.

    When a task requests a scenario that already has a warm server,
    we reuse it only after the previous lease is released. This keeps
    every rollout sample on an exclusive MCP server / SQLite file pair.
    When the scenario differs, we start a new server (or evict the LRU
    if we've hit capacity).

    With NullPool in server.py, replacing the .db file between tasks
    produces correct results without restarting the server process.
    """

    def __init__(
        self,
        max_servers: int = 8,
        startup_timeout: float = 30.0,
        acquire_poll_interval: float = 0.1,
        startup_max_retries: int = 2,
    ):
        self._lock = threading.Lock()
        self._slots: dict[str, list[_ServerSlot]] = {}  # scenario -> slots
        self._max_servers = max_servers
        self._startup_timeout = startup_timeout
        self._acquire_poll_interval = acquire_poll_interval
        self._startup_max_retries = startup_max_retries
        # Track which concrete slots are busy (leased to envs)
        self._busy: set[int] = set()
        # Track in-flight server starts per scenario.
        self._starting: dict[str, int] = {}

    async def acquire(self, scenario: str) -> _ServerSlot:
        """Get an exclusive server slot for the given scenario."""
        slots_to_stop: list[_ServerSlot] = []

        while True:
            should_start = False
            slot_to_return: _ServerSlot | None = None
            with self._lock:
                slots_to_stop.extend(self._prune_dead_slots_locked(scenario))
                slots = self._slots.get(scenario, [])

                for slot in slots:
                    if slot.port not in self._busy:
                        slot.last_used = time.time()
                        self._busy.add(slot.port)
                        slot_to_return = slot
                        break

                if slot_to_return is None:
                    if self._total_slots_locked() + self._total_starting_locked() >= self._max_servers:
                        evicted = self._evict_one_locked()
                        if evicted is not None:
                            slots_to_stop.append(evicted)

                    if self._total_slots_locked() + self._total_starting_locked() < self._max_servers:
                        self._starting[scenario] = self._starting.get(scenario, 0) + 1
                        should_start = True

            for slot in slots_to_stop:
                self._stop_slot(slot)
            slots_to_stop = []

            if slot_to_return is not None:
                return slot_to_return

            if should_start:
                break

            await asyncio.sleep(self._acquire_poll_interval)

        try:
            slot = await self._start_server(scenario, max_retries=self._startup_max_retries)
        except Exception:
            with self._lock:
                self._decrement_starting_locked(scenario)
            raise

        with self._lock:
            self._decrement_starting_locked(scenario)
            slot.last_used = time.time()
            self._slots.setdefault(scenario, []).append(slot)
            self._busy.add(slot.port)

        return slot

    def release(self, slot: _ServerSlot):
        """Mark a server slot as no longer busy."""
        with self._lock:
            self._busy.discard(slot.port)
            slot.last_used = time.time()

    def shutdown_all(self):
        """Stop all servers and the pre-fork launcher. Call on process exit."""
        with self._lock:
            for slots in self._slots.values():
                for slot in slots:
                    self._stop_slot(slot)
            self._slots.clear()
            self._busy.clear()
        if _LAUNCHER is not None:
            _LAUNCHER.shutdown()

    def _evict_one_locked(self):
        """Evict the least-recently-used non-busy slot. Must hold _lock."""
        candidates = [
            (scenario, slot)
            for scenario, slots in self._slots.items()
            for slot in slots
            if slot.port not in self._busy
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1].last_used)
        evict_scenario, evict_slot = candidates[0]
        slots = [slot for slot in self._slots.get(evict_scenario, []) if slot.port != evict_slot.port]
        if slots:
            self._slots[evict_scenario] = slots
        else:
            self._slots.pop(evict_scenario, None)
        return evict_slot

    def _prune_dead_slots_locked(self, scenario: str) -> list[_ServerSlot]:
        dead_slots: list[_ServerSlot] = []
        healthy_slots: list[_ServerSlot] = []
        for slot in self._slots.get(scenario, []):
            if slot.proc.poll() is None:
                healthy_slots.append(slot)
            else:
                dead_slots.append(slot)
                self._busy.discard(slot.port)

        if healthy_slots:
            self._slots[scenario] = healthy_slots
        else:
            self._slots.pop(scenario, None)
        return dead_slots

    def _total_slots_locked(self) -> int:
        return sum(len(slots) for slots in self._slots.values())

    def _total_starting_locked(self) -> int:
        return sum(self._starting.values())

    def _decrement_starting_locked(self, scenario: str) -> None:
        count = self._starting.get(scenario, 0)
        if count <= 1:
            self._starting.pop(scenario, None)
        else:
            self._starting[scenario] = count - 1

    async def _start_server(self, scenario: str, max_retries: int = 2) -> _ServerSlot:
        """Start a new MCP server subprocess for the given scenario.

        Retries on failure with a fresh port each time.
        """
        last_error: Exception | None = None
        pool = get_port_pool()

        for attempt in range(1 + max_retries):
            port = pool.acquire()
            db_path = os.path.join(_DATA_CACHE.db_dir, f"{scenario}_slot_p{port}.db")
            log_handle = None
            proc = None

            try:
                await asyncio.to_thread(_reset_db_file, scenario, db_path)
                script_dir = AWM_ROOT
                log_dir = os.path.join("/dev/shm", f"awm_server_logs_{os.getuid()}")
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(log_dir, f"{scenario}_{port}.log")

                _kill_port(port)

                # Generate server script from cached data (avoids 100MB
                # gen_envs.jsonl read per subprocess — critical on shared FS).
                server_script = _write_server_script(
                    scenario, db_path, "127.0.0.1", port,
                )

                # Use pre-fork launcher: child inherits pre-imported
                # modules (~0s import vs ~4s cold start).
                launcher = get_server_launcher()
                child_pid = await asyncio.to_thread(
                    launcher.start_server,
                    server_script, log_path, _THREAD_LIMIT_ENV,
                )
                proc = _PidProcess(child_pid)
                proc._log_path = log_path

                mcp_url = f"http://127.0.0.1:{port}/mcp"

                # Wait for server readiness
                ready = await self._wait_for_server(mcp_url, self._startup_timeout)

                if not ready:
                    raise RuntimeError(
                        f"MCP server for {scenario} failed to start on port {port} "
                        f"(attempt {attempt + 1}/{1 + max_retries})"
                    )

                # Fetch and cache tool listing (same for all tasks in a scenario)
                tools_text = await self._fetch_tools(mcp_url)

                return _ServerSlot(
                    scenario=scenario, port=port, db_path=db_path, proc=proc,
                    mcp_url=mcp_url, tools_text=tools_text,
                    script_path=server_script, last_used=time.time(),
                )
            except Exception as e:
                last_error = e
                # Clean up failed attempt
                if proc is not None:
                    try:
                        proc.kill()
                        proc.wait(timeout=3)
                    except Exception:
                        pass
                if log_handle is not None:
                    try:
                        log_handle.close()
                    except Exception:
                        pass
                _kill_port(port)
                pool.release(port)

                if attempt < max_retries:
                    import logging
                    logging.getLogger(__name__).warning(
                        "MCP server start failed for %s (attempt %d/%d): %s — retrying",
                        scenario, attempt + 1, 1 + max_retries, e,
                    )
                    await asyncio.sleep(0.5)

        raise RuntimeError(
            f"MCP server for {scenario} failed after {1 + max_retries} attempts: {last_error}"
        )

    def _stop_slot(self, slot: _ServerSlot):
        """Terminate a server and release its port."""
        try:
            slot.proc.terminate()
            try:
                slot.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                slot.proc.kill()
                slot.proc.wait(timeout=1)
        except Exception:
            pass
        log_handle = getattr(slot.proc, "_log_handle", None)
        if log_handle:
            try:
                log_handle.close()
            except Exception:
                pass
        if slot.db_path and os.path.exists(slot.db_path):
            try:
                os.remove(slot.db_path)
            except OSError:
                pass
        if slot.script_path and os.path.exists(slot.script_path):
            try:
                os.remove(slot.script_path)
            except OSError:
                pass
        _kill_port(slot.port)
        get_port_pool().release(slot.port)

    async def _wait_for_server(
        self, url: str, timeout: float, poll_interval: float = 0.3,
    ) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                running, tools_count, _, _ = await check_mcp_server(
                    url=url, timeout=min(1.5, deadline - time.time()),
                )
                if running and tools_count > 0:
                    return True
            except Exception:
                pass
            await asyncio.sleep(poll_interval)
        return False

    async def _fetch_tools(self, mcp_url: str) -> str:
        """Connect to MCP server, list tools, return formatted text."""
        async with MCPToolExecutor(mcp_url, timeout=15.0) as mcp:
            tools = await mcp.list_tools()
            return format_tools_for_response(tools)


# Global server pool
_SERVER_POOL: ServerPool | None = None


def get_server_pool(
    max_servers: int = 8,
    startup_timeout: float = 30.0,
    acquire_poll_interval: float = 0.1,
    startup_max_retries: int = 2,
) -> ServerPool:
    global _SERVER_POOL
    if _SERVER_POOL is None:
        _SERVER_POOL = ServerPool(
            max_servers=max_servers,
            startup_timeout=startup_timeout,
            acquire_poll_interval=acquire_poll_interval,
            startup_max_retries=startup_max_retries,
        )
    else:
        _SERVER_POOL._max_servers = max_servers
        _SERVER_POOL._startup_timeout = startup_timeout
        _SERVER_POOL._acquire_poll_interval = acquire_poll_interval
        _SERVER_POOL._startup_max_retries = startup_max_retries
    return _SERVER_POOL


# ── AWM Environment ─────────────────────────────────────────────────────────

class AWMEnv:
    """
    AWM environment for slime multi-turn rollout.

    Lifecycle:
        1. __init__: store metadata (scenario, task, task_idx)
        2. reset(): reset DB, start MCP server, list tools
        3. step(response_text): parse tool calls, execute via MCP, return observation
        4. close(): stop MCP server, run verification, compute reward
    """

    def __init__(
        self,
        scenario: str,
        task: str,
        task_idx: int,
        max_iterations: int = 15,
        server_startup_timeout: float = 30.0,
        server_startup_max_retries: int = 2,
        server_pool_max_servers: int = 8,
        server_pool_acquire_poll_interval: float = 0.1,
        tool_timeout: float = 30.0,
    ):
        self.scenario = normalize_scenario_name(scenario)
        self.task = task
        self.task_idx = task_idx
        self.max_iterations = max_iterations
        self.server_startup_timeout = server_startup_timeout
        self.server_startup_max_retries = server_startup_max_retries
        self.server_pool_max_servers = server_pool_max_servers
        self.server_pool_acquire_poll_interval = server_pool_acquire_poll_interval
        self.tool_timeout = tool_timeout

        # Runtime state
        self._slot: _ServerSlot | None = None
        self._mcp: MCPToolExecutor | None = None
        self._tools_text: str = ""
        self._tools_listed: bool = False
        self._iteration: int = 0

        # DB paths
        self._db_path: str = ""
        self._initial_db_path: str = ""

        # Result
        self.reward: float | None = None
        self.trajectory: list[dict] = []
        self._final_answer: str = ""
        self._verify_result: dict | None = None

    async def reset(self):
        """Reset DB, acquire/reuse MCP server, prepare for interaction.

        Uses the global ServerPool to reuse servers for the same scenario.
        With NullPool in SQLAlchemy, replacing the leased slot's .db file is
        sufficient — no server restart needed for same-scenario tasks.
        """
        assert _DATA_CACHE.loaded, "Call preload_awm_data() before creating envs"

        self._iteration = 0
        self.trajectory = []
        self.reward = None
        self._final_answer = ""
        self._tools_listed = False

        # Acquire an exclusive server slot for this scenario first, then
        # reset that slot's dedicated database file.
        server_pool = get_server_pool(
            max_servers=self.server_pool_max_servers,
            startup_timeout=self.server_startup_timeout,
            acquire_poll_interval=self.server_pool_acquire_poll_interval,
            startup_max_retries=self.server_startup_max_retries,
        )
        self._slot = await server_pool.acquire(self.scenario)
        self._tools_text = self._slot.tools_text  # cached from first startup
        self._db_path = self._slot.db_path
        await asyncio.to_thread(self._reset_db, self._db_path)

        # Save initial snapshot for verification (use port to avoid collisions)
        self._initial_db_path = os.path.join(
            _DATA_CACHE.db_dir, f"{self.scenario}_init_p{self._slot.port}.db"
        )
        await asyncio.to_thread(shutil.copy2, self._db_path, self._initial_db_path)

        # Connect MCP executor within the running rollout event loop.
        self._mcp = MCPToolExecutor(self._slot.mcp_url, timeout=self.tool_timeout)
        await self._mcp.__aenter__()

    async def step(self, response_text: str) -> tuple[dict, bool, dict]:
        """
        Process LLM response, execute tool calls, return observation.

        Returns:
            (observation, done, info)
            - observation: dict with 'obs_str' and 'role'
            - done: whether episode is over
            - info: additional info dict
        """
        self._iteration += 1
        is_final = self._iteration >= self.max_iterations

        # Parse tool calls from response
        tool_calls = parse_tool_calls(response_text)

        self.trajectory.append({
            "role": "assistant",
            "content": response_text,
            "tool_calls_count": len(tool_calls),
        })

        info: dict[str, Any] = {"iteration": self._iteration}

        # No tool calls -> task complete
        if not tool_calls:
            self._final_answer = response_text
            info["done_reason"] = "no_tool_call"
            return {"obs_str": "", "role": "tool"}, True, info

        # Execute first tool call only
        tc = tool_calls[0]
        name = tc["name"]
        arguments = tc["arguments"]
        tool_call_id = tc["id"]

        if len(tool_calls) > 1:
            info["skipped_tools"] = [t["name"] for t in tool_calls[1:]]

        # Execute tool
        if name == "list_tools":
            # Use cached tools text from server pool (same for all tasks in a scenario)
            response_text = self._tools_text
            self._tools_listed = True
            info["tool_name"] = "list_tools"

        elif name == "call_tool":
            tool_name, tool_args = parse_call_tool_arguments(arguments)
            info["tool_name"] = tool_name
            info["tool_args"] = tool_args
            try:
                response_text = await self._mcp.call_tool(tool_name, tool_args)
            except asyncio.TimeoutError:
                response_text = f"Error: Tool call timed out after {self.tool_timeout}s"
                info["tool_error"] = "timeout"
            except Exception as e:
                response_text = f"Error: {e}"
                info["tool_error"] = str(e)
        else:
            response_text = f"Error: Unknown tool '{name}'. Only 'list_tools' and 'call_tool' are available."
            info["tool_error"] = f"unknown_tool_{name}"

        self.trajectory.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "tool_name": info.get("tool_name", name),
            "content": response_text,
        })

        obs = {
            "obs_str": response_text,
            "role": "tool",
            "tool_call_id": tool_call_id,
        }

        done = is_final
        if done:
            info["done_reason"] = "max_iterations"

        return obs, done, info

    def compute_reward(self) -> float:
        """Run AWM verification and compute reward. Call after episode ends."""
        verifier_key = f"{self.scenario}::{self.task_idx}"
        verifier_item = _DATA_CACHE.verifiers.get(verifier_key)

        if not verifier_item:
            self.reward = 0.0
            return self.reward

        code = verifier_item.get("verification", {}).get("code", "")
        if not code:
            self.reward = 0.0
            return self.reward

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
                initial_db_path=self._initial_db_path,
                mode=mode,
                final_db_path=self._db_path,
                final_answer=self._final_answer,
            )
            self._verify_result = result

            # Check completion
            if result.get("execution_status") == "success":
                inner = result.get("result", {})
                if isinstance(inner, dict):
                    result_val = inner.get("result", "")
                    if isinstance(result_val, str) and result_val.lower() == "complete":
                        self.reward = 1.0
                        return self.reward

            # Partial completion check (if verifier returns partial info)
            self.reward = 0.0
            return self.reward

        except Exception as e:
            self._verify_result = {"error": str(e)}
            self.reward = 0.0
            return self.reward

    async def close(self):
        """Release MCP server back to pool, clean up per-task resources.

        The server process is NOT killed — it stays warm in the pool
        for reuse by subsequent tasks on the same scenario.
        """
        # Close MCP connection (per-task, not per-server)
        if self._mcp is not None:
            try:
                await self._mcp.__aexit__(None, None, None)
            except Exception:
                pass
            self._mcp = None

        # Release server slot back to pool (server stays alive)
        if self._slot is not None:
            get_server_pool().release(self._slot)
            self._slot = None

        # Clean up initial db snapshot
        if self._initial_db_path and os.path.exists(self._initial_db_path):
            try:
                os.remove(self._initial_db_path)
            except OSError:
                pass

    def _reset_db(self, db_path: str):
        """Reset database to initial state using cached schema/sample data."""
        _reset_db_file(self.scenario, db_path)



# ── Slime-compatible env builder ─────────────────────────────────────────────

def build_env(sample=None, args: Any = None, **kwargs) -> AWMEnv:
    """
    Build an AWMEnv from a slime Sample.

    Expected sample.metadata fields:
        - scenario: str
        - task: str
        - task_idx: int
    """
    metadata = sample.metadata if sample else {}
    max_turns = getattr(args, "max_turns", 15) if args else 15

    env = AWMEnv(
        scenario=metadata["scenario"],
        task=metadata["task"],
        task_idx=metadata["task_idx"],
        max_iterations=max_turns,
        server_startup_timeout=getattr(args, "server_startup_timeout", 30.0) if args else 30.0,
        server_startup_max_retries=getattr(args, "server_startup_max_retries", 2) if args else 2,
        server_pool_max_servers=getattr(args, "server_pool_max_servers", 8) if args else 8,
        server_pool_acquire_poll_interval=getattr(args, "server_pool_acquire_poll_interval", 0.05) if args else 0.05,
        tool_timeout=getattr(args, "tool_timeout", 30.0) if args else 30.0,
    )
    return env
