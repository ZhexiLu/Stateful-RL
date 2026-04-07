"""
Extract executable predicates from AWM verification code.

For each task's verifier, we decompose it into a set of SQL predicates
that can be evaluated at any intermediate step to measure progress.

Pipeline:
  1. Programmatic AST analysis of verifier source code
  2. LLM-assisted completion for complex verifiers (optional)
  3. Replay validation against known trajectories

Usage:
  cd slime/
  python examples/awm_esa/predicate_extractor.py \
    --awm_root ../agent-world-model \
    --output examples/awm_esa/data/predicates.jsonl
"""
import argparse
import ast
import json
import os
import re
import sqlite3
import sys


def extract_sql_from_code(code: str) -> list[str]:
    """Extract SQL query strings from Python verification code via regex."""
    patterns = [
        r'\.execute\(\s*["\'](.+?)["\']',          # cursor.execute("SQL")
        r'\.execute\(\s*f["\'](.+?)["\']',          # cursor.execute(f"SQL")
        r'\.execute\(\s*"""(.+?)"""',               # cursor.execute("""SQL""")
        r"\.execute\(\s*'''(.+?)'''",               # cursor.execute('''SQL''')
    ]
    sqls = []
    for pat in patterns:
        for match in re.finditer(pat, code, re.DOTALL):
            sql = match.group(1).strip()
            if sql:
                sqls.append(sql)
    return sqls


def classify_predicate(sql: str) -> dict:
    """Classify a SQL query into a predicate type."""
    sql_upper = sql.upper().strip()

    # Count queries → scalar predicate
    if "COUNT(" in sql_upper:
        return {"type": "scalar", "subtype": "count"}

    # SUM/AVG queries → scalar predicate
    if any(agg in sql_upper for agg in ["SUM(", "AVG(", "TOTAL("]):
        return {"type": "scalar", "subtype": "aggregate"}

    # Existence check
    if sql_upper.startswith("SELECT") and "WHERE" in sql_upper:
        return {"type": "binary", "subtype": "existence"}

    return {"type": "binary", "subtype": "generic"}


def extract_predicates_from_verifier(code: str, scenario: str, task_idx: int,
                                     db_path: str) -> list[dict]:
    """Extract executable predicates from a single verifier's code.

    Returns list of predicate dicts:
      - sql: SQL query string
      - type: "binary" or "scalar"
      - expected: expected value for binary predicates (from initial state)
      - target: target value for scalar predicates
      - initial: initial value for scalar predicates
    """
    sqls = extract_sql_from_code(code)
    if not sqls:
        return []

    predicates = []
    seen = set()

    # Try to evaluate each SQL against the initial DB to get baseline values
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
    except Exception:
        conn = None
        cursor = None

    for sql in sqls:
        # Skip non-SELECT queries
        if not sql.strip().upper().startswith("SELECT"):
            continue

        # Skip duplicate queries
        sql_normalized = " ".join(sql.split())
        if sql_normalized in seen:
            continue
        seen.add(sql_normalized)

        # Skip queries with Python format variables that can't be evaluated
        if "{" in sql or "?" in sql:
            continue

        pred_info = classify_predicate(sql)
        pred = {
            "sql": sql,
            "type": pred_info["type"],
        }

        # Try to evaluate against initial DB
        if cursor is not None:
            try:
                cursor.execute(sql)
                result = cursor.fetchone()
                initial_val = result[0] if result else None

                if pred_info["type"] == "scalar":
                    pred["initial"] = initial_val if initial_val is not None else 0
                    # For scalar predicates, we need the target.
                    # We cannot determine it from the initial state alone.
                    # Mark as needing LLM completion.
                    pred["target"] = None  # To be filled by LLM or manually
                else:
                    # Binary: the initial value tells us what should change
                    pred["expected"] = initial_val
            except Exception:
                pass

        predicates.append(pred)

    if conn is not None:
        conn.close()

    return predicates


def extract_predicates_with_heuristics(code: str, task: str, scenario: str,
                                       task_idx: int, db_path: str) -> list[dict]:
    """Enhanced extraction using task description heuristics.

    For modification tasks (add/create/update/delete), we can infer
    what the predicates should check based on the task description.
    """
    base_predicates = extract_predicates_from_verifier(code, scenario, task_idx, db_path)

    # Heuristic: detect comparison patterns in code
    # e.g., "if final_count > initial_count" → the predicate checks for increase
    comparison_patterns = [
        (r'final.*>.*initial', "increase"),
        (r'final.*<.*initial', "decrease"),
        (r'final.*==.*initial', "unchanged"),
        (r'final.*!=.*initial', "changed"),
    ]

    task_lower = task.lower()
    task_type = "unknown"
    if any(kw in task_lower for kw in ["add", "create", "insert", "book", "post", "register"]):
        task_type = "create"
    elif any(kw in task_lower for kw in ["update", "change", "modify", "edit", "set"]):
        task_type = "update"
    elif any(kw in task_lower for kw in ["delete", "remove", "cancel"]):
        task_type = "delete"
    elif any(kw in task_lower for kw in ["get", "find", "search", "list", "show", "check"]):
        task_type = "query"

    # For create tasks with count predicates, set target = initial + 1
    for pred in base_predicates:
        if pred["type"] == "scalar" and pred.get("target") is None:
            initial = pred.get("initial", 0)
            if task_type == "create" and initial is not None:
                pred["target"] = initial + 1
            elif task_type == "delete" and initial is not None and initial > 0:
                pred["target"] = initial - 1

    # For binary existence predicates on create tasks, flip expected
    for pred in base_predicates:
        if pred["type"] == "binary":
            expected = pred.get("expected")
            if task_type == "create" and expected is not None:
                # After creation, the row should exist (or count should increase)
                if expected == 0 or expected is None:
                    pred["expected"] = 1  # Expect existence after creation

    return base_predicates


def validate_predicates(predicates: list[dict], db_path: str) -> list[dict]:
    """Validate that all predicates are executable against the DB."""
    valid = []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
    except Exception:
        return predicates  # Can't validate, return all

    for pred in predicates:
        try:
            cursor.execute(pred["sql"])
            _ = cursor.fetchone()
            valid.append(pred)
        except Exception:
            pass  # Skip non-executable predicates

    conn.close()
    return valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--awm_root", default="../agent-world-model")
    parser.add_argument("--output", default="examples/awm_esa/data/predicates.jsonl")
    parser.add_argument("--db_dir", default="/dev/shm/awm_databases")
    args = parser.parse_args()

    if args.awm_root not in sys.path:
        sys.path.insert(0, args.awm_root)
    from awm.tools import tools_jsonl_load, normalize_scenario_name
    from awm.core.db import create_sqlite_database
    from awm.core.sample import execute_sample_data

    # Load data
    schemas = {}
    for item in tools_jsonl_load(os.path.join(args.awm_root, "outputs/gen_db.jsonl")):
        schemas[normalize_scenario_name(item["scenario"])] = item

    sample_data = {}
    for item in tools_jsonl_load(os.path.join(args.awm_root, "outputs/gen_sample.jsonl")):
        sample_data[normalize_scenario_name(item["scenario"])] = item

    verifiers = {}
    for item in tools_jsonl_load(os.path.join(args.awm_root, "outputs/gen_verifier.pure_code.jsonl")):
        s = normalize_scenario_name(item["scenario"])
        verifiers[f"{s}::{item['task_idx']}"] = item

    tasks = {}
    for item in tools_jsonl_load(os.path.join(args.awm_root, "outputs/gen_tasks.jsonl")):
        s = normalize_scenario_name(item["scenario"])
        tasks[s] = item.get("tasks", [])

    # Create template DBs for predicate evaluation
    template_dir = os.path.join(args.db_dir, "_templates")
    os.makedirs(template_dir, exist_ok=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    total = 0
    with_preds = 0

    with open(args.output, "w") as f:
        for key, verifier in verifiers.items():
            scenario, task_idx_str = key.split("::")
            task_idx = int(task_idx_str)

            code = verifier.get("verification", {}).get("code", "")
            if not code:
                continue

            # Ensure template DB exists
            db_path = os.path.join(template_dir, f"{scenario}.db")
            if not os.path.exists(db_path):
                schema = schemas.get(scenario)
                if not schema:
                    continue
                tmp_path, _, _, _ = create_sqlite_database(
                    scenario, schema["db_schema"], template_dir)
                sd = sample_data.get(scenario)
                if sd:
                    execute_sample_data(tmp_path, sd["sample_data"], scenario)
                if tmp_path != db_path:
                    os.rename(tmp_path, db_path)

            # Get task description
            task_list = tasks.get(scenario, [])
            task_desc = task_list[task_idx] if task_idx < len(task_list) else ""

            # Extract predicates
            predicates = extract_predicates_with_heuristics(
                code, task_desc, scenario, task_idx, db_path)

            # Validate
            predicates = validate_predicates(predicates, db_path)

            total += 1
            if predicates:
                with_preds += 1

            row = {
                "scenario": scenario,
                "task_idx": task_idx,
                "task": task_desc,
                "predicates": predicates,
                "num_predicates": len(predicates),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Processed {total} tasks, {with_preds} with extractable predicates "
          f"({with_preds/max(1,total)*100:.1f}%)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
