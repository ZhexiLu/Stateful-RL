"""
Code-Augmented LLM-as-a-Judge for AWM task evaluation.

Follows AWM paper's approach:
  1. Run SQL verifier code to collect database evidence (initial vs final state)
  2. Pass evidence + success/failure criteria + agent trajectory to LLM judge
  3. LLM judge classifies: Completed / Environment Error / Agent Error

Usage:
  # Start vLLM judge model:
  python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen3.5-35B-A3B --tensor-parallel-size 8 --port 8001

  # Run on saved rollouts (offline, with DB replay):
  python examples/awm_esa/llm_judge.py \
    --judge_url http://127.0.0.1:8001 \
    --input examples/awm_esa/data/prompt_comparison_results.jsonl \
    --output examples/awm_esa/data/judge_results.jsonl \
    --limit 20

  # Run on full dataset:
  python examples/awm_esa/llm_judge.py \
    --judge_url http://127.0.0.1:8001
"""
import asyncio
import json
import logging
import os
import shutil
import sqlite3
import sys
import time

import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SLIME_DIR = os.path.join(SCRIPT_DIR, "..", "..")
AWM_ROOT = os.path.join(SLIME_DIR, "..", "agent-world-model")
if AWM_ROOT not in sys.path:
    sys.path.insert(0, AWM_ROOT)

from awm.tools import tools_jsonl_load, normalize_scenario_name
from awm.core.db import create_sqlite_database
from awm.core.sample import execute_sample_data
from awm.core.verifier import execute_verification_code, VerificationMode


# ═══════════════════════════════════════════════════════════════════════════════
# Judge System Prompt
# ═══════════════════════════════════════════════════════════════════════════════
JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator for tool-use agent task results with access to database verification. Based on the provided agent trajectory AND the code-based verification results from querying the database, decide the task outcome.

### Input
- task: the user task description
- agent trajectory: the full interaction history
- verification results: structured data collected by comparing database states before and after agent execution
- success criteria: what the verification results should look like if the task was completed
- failure criteria: what indicates the task was not completed

### Classification Categories
- Completed: all required steps were successfully executed, AND the database state confirms the task was completed.
- Environment Error: the agent is blocked by MCP server or environment error, e.g., 5xx errors such as 'Internal Server Error', or the MCP server cannot process valid tool calls (e.g., API route conflicts returning 422 for valid requests).
- Agent Error: the agent made mistakes, used invalid parameters, or failed to complete the user's instruction due to agent-side issues.

### Priority Order for Classification
1. Completed (trajectory shows success AND database confirms it)
2. Environment Error (blocked by MCP server or environment error)
3. Agent Error (agent-side issues, e.g., invalid tool arguments, hallucination)

### Key Considerations
- Use the verification results to help judge task completion, but they may be incomplete or inaccurate.
- Comprehensively consider BOTH the trajectory AND verification results.
- If the agent's approach was reasonable but failed due to API bugs (422 for valid parameters, 500 errors, route conflicts), classify as Environment Error.
- If the agent completed the task correctly but the code verifier shows minor mismatches (e.g., format differences like "Monday" vs "Mon"), still classify as Completed.

### Output Format (must be valid JSON, no markdown fences)
{
  "reasoning": "<brief explanation considering both trajectory and verification results>",
  "classification": "<Completed | Environment Error | Agent Error>",
  "confidence": <float 0-1>
}"""


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════
_cache = {}


def _load_awm_data():
    if _cache:
        return
    _cache["sql_verifiers"] = {}
    _cache["code_verifiers"] = {}
    _cache["db_schemas"] = {}
    _cache["sample_data"] = {}

    # SQL mode verifiers (for LLM judge evidence)
    for item in tools_jsonl_load(os.path.join(AWM_ROOT, "outputs/gen_verifier.jsonl")):
        s = normalize_scenario_name(item["scenario"])
        key = f"{s}::{item['task_idx']}"
        raw = item.get("verification", {}).get("raw_response", "")
        code = item.get("verification", {}).get("code", "")
        try:
            parsed_raw = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed_raw = {}
        _cache["sql_verifiers"][key] = {
            "code": code,
            "function_name": parsed_raw.get("function_name", "verify_task"),
            "reasoning": parsed_raw.get("reasoning", ""),
            "success_criteria": parsed_raw.get("success_criteria", ""),
            "failure_criteria": parsed_raw.get("failure_criteria", ""),
        }

    # Code mode verifiers (for binary outcome check)
    for item in tools_jsonl_load(os.path.join(AWM_ROOT, "outputs/gen_verifier.pure_code.jsonl")):
        s = normalize_scenario_name(item["scenario"])
        _cache["code_verifiers"][f"{s}::{item['task_idx']}"] = item

    for item in tools_jsonl_load(os.path.join(AWM_ROOT, "outputs/gen_db.jsonl")):
        _cache["db_schemas"][normalize_scenario_name(item["scenario"])] = item
    for item in tools_jsonl_load(os.path.join(AWM_ROOT, "outputs/gen_sample.jsonl")):
        _cache["sample_data"][normalize_scenario_name(item["scenario"])] = item

    logger.info("Loaded %d SQL verifiers, %d code verifiers",
                len(_cache["sql_verifiers"]), len(_cache["code_verifiers"]))


# ═══════════════════════════════════════════════════════════════════════════════
# SQL Verifier Execution (Step 1: collect evidence)
# ═══════════════════════════════════════════════════════════════════════════════
def run_sql_verifier(scenario, task_idx, initial_db, final_db):
    """Run SQL mode verifier to collect database evidence for LLM judge.

    Returns dict with verification results + criteria.
    """
    _load_awm_data()
    key = f"{scenario}::{task_idx}"
    verifier = _cache["sql_verifiers"].get(key)
    if not verifier or not verifier["code"]:
        return {"status": "no_verifier", "results": None}

    try:
        result = execute_verification_code(
            python_code=verifier["code"],
            function_name=verifier["function_name"],
            initial_db_path=initial_db,
            mode=VerificationMode.sql,
            final_db_path=final_db,
        )
        return {
            "status": "executed",
            "results": result.get("result", {}),
            "execution_status": result.get("execution_status", "error"),
            "reasoning": verifier["reasoning"],
            "success_criteria": verifier["success_criteria"],
            "failure_criteria": verifier["failure_criteria"],
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "reasoning": verifier["reasoning"],
            "success_criteria": verifier["success_criteria"],
            "failure_criteria": verifier["failure_criteria"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Build Judge Input (Step 2: combine trajectory + evidence)
# ═══════════════════════════════════════════════════════════════════════════════
def build_judge_input(rollout, verification):
    """Build the user message combining trajectory + SQL verification evidence."""
    task = rollout.get("task", "")
    scenario = rollout.get("scenario", "")
    trace = rollout.get("trace", "")
    num_turns = rollout.get("num_turns", 0)
    errors = rollout.get("errors", [])

    # Truncate trace
    if len(trace) > 4000:
        trace = trace[:1500] + "\n\n... [truncated] ...\n\n" + trace[-2500:]

    # Format verification evidence
    if verification["status"] == "executed":
        results_str = json.dumps(verification.get("results", {}), indent=2, default=str)
        if len(results_str) > 2000:
            results_str = results_str[:2000] + "\n... [truncated]"
        evidence_section = f"""### Database Verification Evidence
The following data was collected by comparing the database state before and after the agent's execution:

```json
{results_str}
```

### Success Criteria
{verification.get('success_criteria', 'Not available')}

### Failure Criteria
{verification.get('failure_criteria', 'Not available')}"""
    elif verification["status"] == "error":
        evidence_section = f"""### Database Verification Evidence
Verification code execution failed: {verification.get('error', 'unknown error')}

### Success Criteria
{verification.get('success_criteria', 'Not available')}

### Failure Criteria
{verification.get('failure_criteria', 'Not available')}"""
    else:
        evidence_section = "### Database Verification Evidence\nNo verification available for this task."

    user_msg = f"""### Task
Scenario: {scenario}
Task: {task}

### Agent Trajectory ({num_turns} turns)
{trace}

### Execution Errors
{json.dumps(errors) if errors else "None"}

{evidence_section}

### Your Classification"""
    return user_msg


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Judge Call (Step 3: classify)
# ═══════════════════════════════════════════════════════════════════════════════
async def judge_single(
    session, judge_url, model_name, rollout, verification, semaphore,
):
    """Call LLM judge with trajectory + SQL evidence."""
    user_msg = build_judge_input(rollout, verification)
    judge_input = f"[SYSTEM]\n{JUDGE_SYSTEM_PROMPT}\n\n[USER]\n{user_msg}"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.1,
        "max_tokens": 2000,
        "chat_template_kwargs": {"enable_thinking": True},
    }

    async with semaphore:
        raw = ""
        try:
            async with session.post(
                f"{judge_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                result = await resp.json()

            raw = result["choices"][0]["message"]["content"]

            # Parse JSON — find last JSON object (skip thinking text)
            json_text = raw.strip()
            if "```" in json_text:
                import re
                match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', json_text, re.DOTALL)
                if match:
                    json_text = match.group(1).strip()

            last_json_start = json_text.rfind("{")
            if last_json_start >= 0:
                depth = 0
                for idx in range(last_json_start, len(json_text)):
                    if json_text[idx] == "{":
                        depth += 1
                    elif json_text[idx] == "}":
                        depth -= 1
                        if depth == 0:
                            json_text = json_text[last_json_start:idx+1]
                            break

            parsed = json.loads(json_text)
            classification = parsed.get("classification", "Agent Error")
            confidence = parsed.get("confidence", 0.5)
            reasoning = parsed.get("reasoning", "")

            reward_map = {"Completed": 1.0, "Environment Error": 0.0, "Agent Error": 0.0}
            judge_reward = reward_map.get(classification, 0.0)

            return {
                "scenario": rollout.get("scenario"),
                "task_idx": rollout.get("task_idx"),
                "task": rollout.get("task", ""),
                "prompt_type": rollout.get("prompt_type"),
                "code_verifier_reward": rollout.get("reward", 0.0),
                "judge_classification": classification,
                "judge_reward": judge_reward,
                "judge_confidence": confidence,
                "judge_reasoning": reasoning,
                "judge_input": judge_input,
                "judge_raw_output": raw,
                "verification_status": verification["status"],
            }

        except Exception as e:
            logger.error("Judge error for %s::%s: %s",
                        rollout.get("scenario"), rollout.get("task_idx"), e)
            return {
                "scenario": rollout.get("scenario"),
                "task_idx": rollout.get("task_idx"),
                "task": rollout.get("task", ""),
                "prompt_type": rollout.get("prompt_type"),
                "code_verifier_reward": rollout.get("reward", 0.0),
                "judge_classification": "error",
                "judge_reward": 0.0,
                "judge_reasoning": f"Judge error: {e}",
                "judge_input": judge_input if 'judge_input' in dir() else "",
                "judge_raw_output": raw,
                "verification_status": verification["status"],
            }


# ═══════════════════════════════════════════════════════════════════════════════
# DB Replay — recreate initial/final DB states from rollout traces
# ═══════════════════════════════════════════════════════════════════════════════
def _ensure_template_db(scenario, template_dir):
    """Create template DB for a scenario if not exists."""
    _load_awm_data()
    db_path = os.path.join(template_dir, f"{scenario}.db")
    if os.path.exists(db_path):
        return db_path
    schema = _cache["db_schemas"].get(scenario)
    if not schema:
        return None
    tmp_path, _, _, _ = create_sqlite_database(scenario, schema["db_schema"], template_dir)
    sd = _cache["sample_data"].get(scenario)
    if sd:
        execute_sample_data(tmp_path, sd["sample_data"], scenario)
    if tmp_path != db_path:
        os.rename(tmp_path, db_path)
    return db_path


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
async def run_judge(args):
    _load_awm_data()

    # Load rollouts
    rollouts = []
    with open(args.input) as f:
        for line in f:
            r = json.loads(line)
            if r.get("trace"):
                rollouts.append(r)

    if args.limit:
        import random
        random.seed(42)
        by_task = {}
        for r in rollouts:
            key = f"{r['scenario']}::{r['task_idx']}::{r['prompt_type']}"
            by_task[key] = r
        unique = list(by_task.values())
        random.shuffle(unique)
        rollouts = unique[:args.limit]
        logger.info("Sampled %d unique (task, prompt) pairs", len(rollouts))

    logger.info("Judging %d rollouts", len(rollouts))

    # Detect model
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{args.judge_url}/v1/models") as resp:
            models = await resp.json()
    model_name = models["data"][0]["id"]
    logger.info("Judge model: %s", model_name)

    # Prepare template DBs
    template_dir = os.path.join(args.db_dir, "_templates")
    os.makedirs(template_dir, exist_ok=True)

    semaphore = asyncio.Semaphore(args.concurrency)
    results = []
    t_start = time.time()

    async with aiohttp.ClientSession() as session:
        for i, rollout in enumerate(rollouts):
            scenario = normalize_scenario_name(rollout["scenario"])
            task_idx = rollout.get("task_idx", 0)

            # For offline evaluation, we only have the initial DB (no final DB from rollout)
            # Run SQL verifier with initial_db as both (evidence will show no changes)
            # This is a limitation — in training, we'd have the actual final DB
            template_path = _ensure_template_db(scenario, template_dir)
            if template_path:
                verification = await asyncio.to_thread(
                    run_sql_verifier, scenario, task_idx, template_path, template_path)
            else:
                verification = {"status": "no_db", "results": None}

            result = await judge_single(
                session, args.judge_url, model_name, rollout, verification, semaphore)
            results.append(result)

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                eta = (len(rollouts) - i - 1) / rate if rate > 0 else 0
                logger.info("Progress: %d/%d (%.1f/sec, ETA %.0fs)",
                            i + 1, len(rollouts), rate, eta)

    # Write results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Stats
    elapsed = time.time() - t_start
    total = len(results)
    completed = sum(1 for r in results if r["judge_classification"] == "Completed")
    env_err = sum(1 for r in results if r["judge_classification"] == "Environment Error")
    agent_err = sum(1 for r in results if r["judge_classification"] == "Agent Error")
    errors = sum(1 for r in results if r["judge_classification"] == "error")
    code_completed = sum(1 for r in results if r["code_verifier_reward"] == 1.0)

    false_neg = sum(1 for r in results
                    if r["code_verifier_reward"] == 0.0 and r["judge_classification"] == "Completed")
    false_pos = sum(1 for r in results
                    if r["code_verifier_reward"] == 1.0 and r["judge_classification"] != "Completed"
                    and r["judge_classification"] != "error")

    logger.info("\n" + "=" * 60)
    logger.info("JUDGE RESULTS (%.1f seconds)", elapsed)
    logger.info("=" * 60)
    logger.info("Total: %d", total)
    logger.info("  Completed:         %d (%.1f%%)", completed, 100 * completed / max(1, total))
    logger.info("  Environment Error: %d (%.1f%%)", env_err, 100 * env_err / max(1, total))
    logger.info("  Agent Error:       %d (%.1f%%)", agent_err, 100 * agent_err / max(1, total))
    logger.info("  Judge errors:      %d", errors)
    logger.info("")
    logger.info("Code verifier completed: %d", code_completed)
    logger.info("LLM judge completed:     %d", completed)
    logger.info("False negatives (code=0, judge=Completed): %d", false_neg)
    logger.info("False positives (code=1, judge!=Completed): %d", false_pos)
    logger.info("\nOutput: %s", args.output)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_url", default="http://127.0.0.1:8001")
    parser.add_argument("--input", default="examples/awm_esa/data/prompt_comparison_results.jsonl")
    parser.add_argument("--output", default="examples/awm_esa/data/judge_results.jsonl")
    parser.add_argument("--db_dir", default="/dev/shm/awm_databases")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.environ["OTEL_SDK_DISABLED"] = "true"
    asyncio.run(run_judge(args))


if __name__ == "__main__":
    main()
