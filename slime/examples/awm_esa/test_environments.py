#!/usr/bin/env python3
"""Pre-test AWM environments to find broken scenarios.

Starts each scenario's MCP server, calls list_tools, runs a basic tool call,
then records the result. Outputs a JSON file of healthy/broken scenarios
that can be used to filter train.jsonl.

Usage:
    python test_environments.py [--workers 8] [--timeout 30] [--output env_health.json]
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time

# AWM imports
AWM_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "agent-world-model")
if AWM_ROOT not in sys.path:
    sys.path.insert(0, AWM_ROOT)

from awm.tools import tools_jsonl_load, normalize_scenario_name, check_mcp_server
from awm.core.agent import MCPToolExecutor, format_tools_for_response
from awm.core.db import create_sqlite_database
from awm.core.sample import execute_sample_data


def load_data():
    """Load all AWM data files."""
    db_schemas = {}
    sample_data = {}
    envs_data = {}

    for item in tools_jsonl_load(os.path.join(AWM_ROOT, "outputs/gen_db.jsonl")):
        db_schemas[normalize_scenario_name(item["scenario"])] = item
    for item in tools_jsonl_load(os.path.join(AWM_ROOT, "outputs/gen_sample.jsonl")):
        sample_data[normalize_scenario_name(item["scenario"])] = item
    for item in tools_jsonl_load(os.path.join(AWM_ROOT, "outputs/gen_envs.jsonl")):
        envs_data[normalize_scenario_name(item["scenario"])] = item

    return db_schemas, sample_data, envs_data


def create_db(scenario, db_schemas, sample_data, db_dir):
    """Create a fresh SQLite database for a scenario."""
    schema = db_schemas.get(scenario)
    if not schema:
        return None
    db_path, _, _, _ = create_sqlite_database(scenario, schema["db_schema"], db_dir)
    sample = sample_data.get(scenario)
    if sample:
        execute_sample_data(db_path, sample["sample_data"], scenario)
    return db_path


def write_server_script(scenario, db_path, port, envs_data):
    """Write the MCP server script for a scenario."""
    env_item = envs_data.get(scenario)
    if not env_item:
        return None

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
            setup = textwrap.indent(textwrap.dedent("""\
                from fastapi_mcp import FastApiMCP
                mcp = FastApiMCP(app)
                mcp.mount_http()
            """), "    ")
            new_lines.extend(setup.rstrip().split("\n"))
            line = f"    uvicorn.run(app, host='127.0.0.1', port={port})"
        new_lines.append(line)

    script_dir = os.path.join("/dev/shm", f"awm_env_test_{os.getuid()}")
    os.makedirs(script_dir, exist_ok=True)
    script_path = os.path.join(script_dir, f"test_{scenario}_{port}.py")
    with open(script_path, "w") as f:
        f.write("\n".join(new_lines))
    return script_path


async def test_scenario(scenario, port, db_schemas, sample_data, envs_data, db_dir, timeout):
    """Test a single scenario. Returns a result dict."""
    result = {
        "scenario": scenario,
        "healthy": False,
        "error": None,
        "tools_count": 0,
        "list_tools_ok": False,
        "startup_time": None,
    }

    # Create DB
    try:
        db_path = await asyncio.to_thread(create_db, scenario, db_schemas, sample_data, db_dir)
        if not db_path:
            result["error"] = "no_db_schema"
            return result
    except Exception as e:
        result["error"] = f"db_create_failed: {e}"
        return result

    # Write server script
    script_path = write_server_script(scenario, db_path, port, envs_data)
    if not script_path:
        result["error"] = "no_envs_data"
        return result

    # Start server
    env = os.environ.copy()
    for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
        env[k] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"

    proc = None
    mcp = None
    start_time = time.time()

    try:
        log_path = os.path.join(db_dir, f"test_{scenario}.log")
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(
                [sys.executable, script_path],
                stdout=log_f, stderr=log_f, env=env,
            )

        # Wait for server to start
        mcp_url = f"http://127.0.0.1:{port}/mcp"
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                running, tools_count, _, _ = await check_mcp_server(
                    url=mcp_url, timeout=min(2.0, deadline - time.time()))
                if running and tools_count > 0:
                    result["startup_time"] = round(time.time() - start_time, 1)
                    result["tools_count"] = tools_count
                    break
            except Exception:
                pass
            await asyncio.sleep(0.5)
        else:
            result["error"] = "startup_timeout"
            return result

        # Test list_tools via MCP
        mcp = MCPToolExecutor(mcp_url, timeout=10.0)
        await mcp.__aenter__()
        tools = await mcp.list_tools()
        tools_text = format_tools_for_response(tools)

        if not tools_text or len(tools_text) < 10:
            result["error"] = "empty_tools_list"
            return result

        result["list_tools_ok"] = True
        result["healthy"] = True

    except Exception as e:
        result["error"] = f"test_failed: {e}"

    finally:
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
        # Kill port
        try:
            subprocess.run(["fuser", "-k", f"{port}/tcp"],
                           capture_output=True, timeout=3)
        except Exception:
            pass
        # Cleanup DB
        if db_path and os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass

    return result


async def main():
    parser = argparse.ArgumentParser(description="Pre-test AWM environments")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent test workers")
    parser.add_argument("--timeout", type=int, default=30, help="Per-scenario timeout (seconds)")
    parser.add_argument("--output", type=str, default="data/env_health.json", help="Output file")
    args = parser.parse_args()

    print("Loading AWM data...")
    db_schemas, sample_data, envs_data = load_data()

    scenarios = sorted(set(db_schemas.keys()) & set(envs_data.keys()))
    print(f"Found {len(scenarios)} scenarios to test")

    db_dir = os.path.join("/dev/shm", f"awm_env_test_{os.getuid()}")
    os.makedirs(db_dir, exist_ok=True)

    base_port = 11000
    semaphore = asyncio.Semaphore(args.workers)
    results = []
    done_count = 0

    async def test_with_semaphore(scenario, port):
        nonlocal done_count
        async with semaphore:
            result = await test_scenario(
                scenario, port, db_schemas, sample_data, envs_data, db_dir, args.timeout)
            done_count += 1
            status = "✓" if result["healthy"] else f"✗ {result['error']}"
            print(f"  [{done_count}/{len(scenarios)}] {scenario}: {status}")
            return result

    print(f"Testing with {args.workers} workers, {args.timeout}s timeout...")
    tasks = [
        test_with_semaphore(scenario, base_port + i)
        for i, scenario in enumerate(scenarios)
    ]
    results = await asyncio.gather(*tasks)

    # Summary
    healthy = [r for r in results if r["healthy"]]
    broken = [r for r in results if not r["healthy"]]

    print(f"\n{'='*60}")
    print(f"Results: {len(healthy)} healthy, {len(broken)} broken ({len(broken)/len(results)*100:.1f}%)")

    if broken:
        from collections import Counter
        error_dist = Counter(r["error"] for r in broken)
        print(f"\nError distribution:")
        for err, cnt in error_dist.most_common():
            print(f"  {err}: {cnt}")

    # Save results
    output = {
        "healthy_scenarios": sorted(r["scenario"] for r in healthy),
        "broken_scenarios": {r["scenario"]: r["error"] for r in broken},
        "summary": {
            "total": len(results),
            "healthy": len(healthy),
            "broken": len(broken),
        },
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")

    # Cleanup
    shutil.rmtree(db_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
