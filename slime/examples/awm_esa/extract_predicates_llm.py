"""
LLM-based predicate extraction from AWM verifier code.

Uses a strong LLM (e.g., Qwen3-235B-A22B) to analyze each verifier's Python code
and extract self-contained SQL predicates with ground truth expected values.

Each predicate can be independently evaluated at any intermediate database state
to compute progress reward: Phi(s_t) = mean(predicate_scores).

Usage:
  # Start SGLang server first:
  python -m sglang.launch_server --model-path ../models/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --tp 8 --port 8000 --mem-fraction-static 0.85 --trust-remote-code

  # Then run extraction:
  python examples/awm_esa/extract_predicates_llm.py \
    --awm_root ../agent-world-model \
    --sglang_url http://127.0.0.1:8000 \
    --output examples/awm_esa/data/predicates.jsonl
"""
import argparse
import asyncio
import json
import logging
import os
import sqlite3
import sys
import time

import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are an expert at analyzing Python database verification code. Your task is to extract **self-contained SQL predicates** from a verifier function.

## Context

The verifier below checks whether an AI agent successfully completed a task by comparing the database state before (initial_db) and after (final_db) the agent's execution. It contains multiple conditions that must ALL pass for the task to be marked "complete".

Your job: decompose the verifier into **independent, self-contained SQL predicates** that can each be evaluated on any intermediate database state to measure partial progress.

## Verifier Code

```python
{verifier_code}
```

## Task Description

{task_description}

## Database Schema (table names and columns)

{db_schema}

## Requirements

1. Each predicate must be a **single self-contained SQL query** — no Python variables, no f-strings, no external dependencies. Use subqueries to resolve any ID lookups.

2. Each predicate must return a single value that can be compared against an expected value.

3. Include the **ground truth expected value** extracted from the verifier code's hardcoded constants and logic.

4. Categorize each predicate:
   - "existence": Check if a row exists (expected: 1 for exists, 0 for not exists)
   - "count_gte": Check if count >= expected value
   - "exact_match": Check if a field equals an exact value
   - "text_contains": Check if a text field contains a substring

5. **Skip** any checks on `final_answer` text — those are not database predicates.

6. **Skip** any checks that only verify the initial state hasn't changed (sanity checks).

7. Each SQL must be executable against a SQLite database. Use SQLite syntax only.

8. Resolve all variable dependencies inline. For example, if the verifier does:
   ```python
   profile_id = query("SELECT id FROM profiles WHERE name='Kids'")[0][0]
   controls = query("SELECT * FROM controls WHERE profile_id=?", profile_id)
   ```
   Convert to:
   ```sql
   SELECT COUNT(*)>0 FROM controls WHERE profile_id=(SELECT id FROM profiles WHERE name='Kids')
   ```

## Output Format

Return a JSON object (no markdown fences):
{
  "predicates": [
    {
      "sql": "SELECT ... FROM ...",
      "check_type": "existence|count_gte|exact_match|text_contains",
      "expected": <expected_value>,
      "description": "Brief description of what this checks"
    }
  ],
  "task_type": "query|create|update|delete|combined",
  "num_db_conditions": <number of database-related conditions in verifier>
}"""


async def extract_predicates_for_task(
    session: aiohttp.ClientSession,
    api_url: str,
    model_name: str,
    verifier_code: str,
    task_description: str,
    db_schema: str,
    scenario: str,
    task_idx: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Call LLM via OpenAI-compatible API to extract predicates for one task."""
    prompt = (EXTRACTION_PROMPT
              .replace("{verifier_code}", verifier_code)
              .replace("{task_description}", task_description)
              .replace("{db_schema}", db_schema))

    messages = [
        {"role": "system", "content": "You are a precise code analysis assistant. Output valid JSON only, no markdown fences."},
        {"role": "user", "content": prompt},
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 2048,
        "top_p": 0.95,
    }

    async with semaphore:
        try:
            async with session.post(
                f"{api_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=180),
            ) as resp:
                result = await resp.json()

            if "error" in result:
                raise RuntimeError(f"API error: {result['error']}")
            raw_text = result["choices"][0]["message"]["content"]
            logger.info("Got response for %s::%d (%d chars)", scenario, task_idx, len(raw_text))

            # Parse JSON from response — robust extraction
            json_text = raw_text.strip()

            # Strip markdown fences
            if "```" in json_text:
                import re as _re
                match = _re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', json_text, _re.DOTALL)
                if match:
                    json_text = match.group(1).strip()

            # Find the outermost JSON object if there's extra text
            if not json_text.startswith("{"):
                start = json_text.find("{")
                if start >= 0:
                    json_text = json_text[start:]

            # Find matching closing brace
            if json_text.startswith("{"):
                depth = 0
                for idx, ch in enumerate(json_text):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            json_text = json_text[:idx+1]
                            break

            parsed = json.loads(json_text)
            return {
                "scenario": scenario,
                "task_idx": task_idx,
                "task": task_description,
                "predicates": parsed.get("predicates", []),
                "task_type": parsed.get("task_type", "unknown"),
                "num_db_conditions": parsed.get("num_db_conditions", 0),
                "raw_response": raw_text,
                "status": "success",
            }

        except json.JSONDecodeError as e:
            logger.warning("JSON parse error for %s::%d: %s", scenario, task_idx, e)
            return {
                "scenario": scenario,
                "task_idx": task_idx,
                "task": task_description,
                "predicates": [],
                "status": "json_error",
                "raw_response": raw_text if 'raw_text' in dir() else "",
            }
        except Exception as e:
            logger.error("Error for %s::%d: %s", scenario, task_idx, e)
            return {
                "scenario": scenario,
                "task_idx": task_idx,
                "task": task_description,
                "predicates": [],
                "status": "error",
                "error": str(e),
            }


def validate_predicates(predicates: list[dict], db_path: str) -> list[dict]:
    """Validate extracted predicates by executing against initial DB."""
    if not os.path.exists(db_path):
        return predicates

    valid = []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
    except Exception:
        return predicates

    for pred in predicates:
        sql = pred.get("sql", "")
        if not sql:
            continue
        try:
            cursor.execute(sql)
            result = cursor.fetchone()
            pred["initial_value"] = result[0] if result else None
            pred["executable"] = True
            valid.append(pred)
        except Exception as e:
            pred["executable"] = False
            pred["validation_error"] = str(e)
            # Still include it — might work on final DB with different schema state
            valid.append(pred)

    conn.close()
    return valid


def get_db_schema_text(db_path: str) -> str:
    """Extract schema from SQLite database."""
    if not os.path.exists(db_path):
        return ""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")
        tables = cursor.fetchall()
        conn.close()
        return "\n".join(t[0] for t in tables if t[0])
    except Exception:
        return ""


async def run_extraction(args):
    AWM_ROOT = args.awm_root
    if AWM_ROOT not in sys.path:
        sys.path.insert(0, AWM_ROOT)

    from awm.tools import tools_jsonl_load, normalize_scenario_name
    from awm.core.db import create_sqlite_database
    from awm.core.sample import execute_sample_data

    # Load AWM data
    verifiers = {}
    for item in tools_jsonl_load(os.path.join(AWM_ROOT, "outputs/gen_verifier.pure_code.jsonl")):
        s = normalize_scenario_name(item["scenario"])
        verifiers[f"{s}::{item['task_idx']}"] = item

    tasks = {}
    for item in tools_jsonl_load(os.path.join(AWM_ROOT, "outputs/gen_tasks.jsonl")):
        s = normalize_scenario_name(item["scenario"])
        tasks[s] = item.get("tasks", [])

    schemas = {}
    for item in tools_jsonl_load(os.path.join(AWM_ROOT, "outputs/gen_db.jsonl")):
        schemas[normalize_scenario_name(item["scenario"])] = item

    sample_data = {}
    for item in tools_jsonl_load(os.path.join(AWM_ROOT, "outputs/gen_sample.jsonl")):
        sample_data[normalize_scenario_name(item["scenario"])] = item

    # Prepare template DBs
    template_dir = os.path.join(args.db_dir, "_templates")
    os.makedirs(template_dir, exist_ok=True)

    logger.info("Loaded %d verifiers, %d scenarios", len(verifiers), len(tasks))

    # Prepare all tasks
    task_list = []
    for key, verifier in verifiers.items():
        code = verifier.get("verification", {}).get("code", "")
        if not code:
            continue
        scenario, task_idx_str = key.split("::")
        task_idx = int(task_idx_str)
        task_texts = tasks.get(scenario, [])
        task_text = task_texts[task_idx] if task_idx < len(task_texts) else ""

        # Ensure template DB
        db_path = os.path.join(template_dir, f"{scenario}.db")
        if not os.path.exists(db_path):
            schema = schemas.get(scenario)
            if not schema:
                continue
            tmp_path, _, _, _ = create_sqlite_database(scenario, schema["db_schema"], template_dir)
            sd = sample_data.get(scenario)
            if sd:
                execute_sample_data(tmp_path, sd["sample_data"], scenario)
            if tmp_path != db_path:
                os.rename(tmp_path, db_path)

        db_schema_text = get_db_schema_text(db_path)

        task_list.append({
            "scenario": scenario,
            "task_idx": task_idx,
            "task_text": task_text,
            "verifier_code": code,
            "db_schema": db_schema_text,
            "db_path": db_path,
        })

    if args.limit:
        task_list = task_list[:args.limit]

    logger.info("Processing %d tasks with concurrency %d", len(task_list), args.concurrency)

    # Run extraction
    semaphore = asyncio.Semaphore(args.concurrency)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results = []
    t_start = time.time()

    async with aiohttp.ClientSession() as session:
        async def process_one(task_info):
            result = await extract_predicates_for_task(
                session=session,
                api_url=args.api_url,
                model_name=args.model_name,
                verifier_code=task_info["verifier_code"],
                task_description=task_info["task_text"],
                db_schema=task_info["db_schema"],
                scenario=task_info["scenario"],
                task_idx=task_info["task_idx"],
                semaphore=semaphore,
            )

            # Validate predicates
            if result.get("predicates"):
                result["predicates"] = validate_predicates(
                    result["predicates"], task_info["db_path"])

            results.append(result)

            if len(results) % 50 == 0:
                elapsed = time.time() - t_start
                rate = len(results) / elapsed * 60
                logger.info("Progress: %d/%d (%.1f/min)", len(results), len(task_list), rate)

            return result

        # Process in batches
        batch_size = args.concurrency * 2
        for i in range(0, len(task_list), batch_size):
            batch = task_list[i:i+batch_size]
            coros = [process_one(t) for t in batch]
            batch_results = await asyncio.gather(*coros, return_exceptions=True)
            for j, br in enumerate(batch_results):
                if isinstance(br, Exception):
                    logger.error("Task %d failed with exception: %s", i+j, br)

    # Write results
    with open(args.output, "w") as f:
        for r in results:
            # Remove raw_response to save space
            r_clean = {k: v for k, v in r.items() if k != "raw_response"}
            f.write(json.dumps(r_clean, ensure_ascii=False) + "\n")

    # Stats
    elapsed = time.time() - t_start
    success = sum(1 for r in results if r.get("status") == "success")
    with_preds = sum(1 for r in results if len(r.get("predicates", [])) > 0)
    total_preds = sum(len(r.get("predicates", [])) for r in results)
    executable = sum(
        sum(1 for p in r.get("predicates", []) if p.get("executable", False))
        for r in results
    )

    logger.info("\n" + "=" * 60)
    logger.info("EXTRACTION COMPLETE in %.1f minutes", elapsed / 60)
    logger.info("=" * 60)
    logger.info("Total tasks: %d", len(results))
    logger.info("Successful extractions: %d (%.1f%%)", success, 100 * success / max(1, len(results)))
    logger.info("Tasks with predicates: %d (%.1f%%)", with_preds, 100 * with_preds / max(1, len(results)))
    logger.info("Total predicates: %d (avg %.1f/task)", total_preds, total_preds / max(1, with_preds))
    logger.info("Executable predicates: %d (%.1f%%)", executable, 100 * executable / max(1, total_preds))
    logger.info("Output: %s", args.output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--awm_root", default="../agent-world-model")
    parser.add_argument("--api_url", default="http://127.0.0.1:8000", help="vLLM/SGLang OpenAI-compatible API base URL")
    parser.add_argument("--model_name", default="/mnt/lustre/rpi/zlu10/Stateful-RL/models/Qwen3-235B-A22B-Instruct-2507", help="Model name for API")
    parser.add_argument("--output", default="examples/awm_esa/data/predicates.jsonl")
    parser.add_argument("--db_dir", default="/dev/shm/awm_databases")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Process only first N tasks (for testing)")
    args = parser.parse_args()

    asyncio.run(run_extraction(args))


if __name__ == "__main__":
    main()
