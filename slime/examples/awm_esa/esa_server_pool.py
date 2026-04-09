# ESA MCP server pool: slot management, server lifecycle, process control.

import asyncio
import atexit
import os
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass

from esa_config import ESA_CONFIGS, POOL_SEMAPHORE, _STARTUP_SEMAPHORE, logger
from esa_cache import _CACHE, _ensure_template

from awm.tools import check_mcp_server
from awm.core.agent import MCPToolExecutor, format_tools_for_response


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


atexit.register(_shutdown_pool)
