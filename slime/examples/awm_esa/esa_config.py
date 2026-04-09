# ESA configuration, constants, AWM path setup, and system prompt.

import asyncio
import logging
import os
import sys

# ── AWM path setup (must happen before any awm.* imports in other modules) ──
AWM_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "agent-world-model")
if AWM_ROOT not in sys.path:
    sys.path.insert(0, AWM_ROOT)

logger = logging.getLogger("awm_esa")

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════
ESA_CONFIGS = {
    # ── AWM base ──
    "max_turns": 15,
    "pool_size": 64,
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
    "lambda_prog": 0.5,
    "gamma_verify": 0.05,
    "eta": 0.95,
    "format_error_penalty": -1.0,
    "format_correct_bonus": 0.05,  # per-check bonus: status +0.05, tool_call +0.05 = +0.1/turn
    "missing_status_penalty": -1.0, # same as format_error_penalty: hard terminate
    "verify_violation_penalty": -0.3,
    "duplicate_penalty": -0.1,
    "omega_step": 0.3,
    "enable_turn_level": True,
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
