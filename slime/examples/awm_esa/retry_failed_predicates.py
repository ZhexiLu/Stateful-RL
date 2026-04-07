"""
Retry failed predicate extractions.
Reads failed keys, re-runs extraction, merges results back.

Usage:
  cd slime/
  PYTHONPATH=../agent-world-model:. python examples/awm_esa/retry_failed_predicates.py \
    --api_url http://127.0.0.1:8000
"""
import asyncio
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


async def main():
    # Import the extraction function from the main script
    sys.path.insert(0, SCRIPT_DIR)
    from extract_predicates_llm import (
        extract_predicates_for_task, validate_predicates, get_db_schema_text,
    )

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_url", default="http://127.0.0.1:8000")
    parser.add_argument("--model_name", default="/mnt/lustre/rpi/zlu10/Stateful-RL/models/Qwen3-235B-A22B-Instruct-2507")
    parser.add_argument("--awm_root", default="../agent-world-model")
    parser.add_argument("--db_dir", default="/dev/shm/awm_databases")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max_retries", type=int, default=3)
    args = parser.parse_args()

    AWM_ROOT = args.awm_root
    if AWM_ROOT not in sys.path:
        sys.path.insert(0, AWM_ROOT)
    from awm.tools import tools_jsonl_load, normalize_scenario_name
    from awm.core.db import create_sqlite_database
    from awm.core.sample import execute_sample_data

    predicates_path = os.path.join(SCRIPT_DIR, "data", "predicates.jsonl")
    failed_keys_path = os.path.join(SCRIPT_DIR, "data", "failed_keys.json")

    # Load failed keys
    with open(failed_keys_path) as f:
        failed_keys = set(json.load(f))
    logger.info("Retrying %d failed tasks", len(failed_keys))

    # Load existing results (keep successful ones)
    existing = {}
    for line in open(predicates_path):
        r = json.loads(line)
        key = "%s::%d" % (r["scenario"], r["task_idx"])
        if key not in failed_keys:
            existing[key] = r

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

    template_dir = os.path.join(args.db_dir, "_templates")

    # Build retry list
    retry_tasks = []
    for key in failed_keys:
        verifier = verifiers.get(key)
        if not verifier:
            continue
        code = verifier.get("verification", {}).get("code", "")
        if not code:
            continue
        scenario, task_idx_str = key.split("::")
        task_idx = int(task_idx_str)
        task_texts = tasks.get(scenario, [])
        task_text = task_texts[task_idx] if task_idx < len(task_texts) else ""

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
        retry_tasks.append({
            "key": key,
            "scenario": scenario,
            "task_idx": task_idx,
            "task_text": task_text,
            "verifier_code": code,
            "db_schema": db_schema_text,
            "db_path": db_path,
        })

    logger.info("Found %d tasks to retry", len(retry_tasks))

    import aiohttp
    semaphore = asyncio.Semaphore(args.concurrency)
    new_results = {}

    async with aiohttp.ClientSession() as session:
        async def retry_one(task_info):
            for attempt in range(args.max_retries):
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
                if result.get("predicates"):
                    result["predicates"] = validate_predicates(
                        result["predicates"], task_info["db_path"])
                if result.get("status") == "success" and len(result.get("predicates", [])) > 0:
                    new_results[task_info["key"]] = result
                    return
            # All retries failed — keep the last result anyway
            new_results[task_info["key"]] = result

        coros = [retry_one(t) for t in retry_tasks]
        await asyncio.gather(*coros, return_exceptions=True)

    # Merge and rewrite
    success_retry = sum(1 for r in new_results.values()
                        if r.get("status") == "success" and len(r.get("predicates", [])) > 0)
    logger.info("Retry results: %d/%d succeeded", success_retry, len(retry_tasks))

    # Write merged output
    all_results = dict(existing)
    all_results.update(new_results)

    with open(predicates_path, "w") as f:
        for key in sorted(all_results.keys()):
            r = all_results[key]
            r_clean = {k: v for k, v in r.items() if k != "raw_response"}
            f.write(json.dumps(r_clean, ensure_ascii=False) + "\n")

    total = len(all_results)
    with_preds = sum(1 for r in all_results.values() if len(r.get("predicates", [])) > 0)
    logger.info("Final: %d total, %d with predicates (%.1f%%)", total, with_preds, 100*with_preds/total)


if __name__ == "__main__":
    asyncio.run(main())
