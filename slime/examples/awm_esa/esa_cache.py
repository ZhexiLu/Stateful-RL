# ESA data cache: loads schemas, verifiers, predicates, and manages template DBs.

import asyncio
import json
import os
import shutil
import threading
from dataclasses import dataclass, field

from esa_config import ESA_CONFIGS, logger

from awm.tools import tools_jsonl_load, normalize_scenario_name
from awm.core.db import create_sqlite_database
from awm.core.sample import execute_sample_data


@dataclass
class _DataCache:
    db_schemas: dict = field(default_factory=dict)
    sample_data: dict = field(default_factory=dict)
    verifiers: dict = field(default_factory=dict)
    sql_verifiers: dict = field(default_factory=dict)
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
# Template DB
# ═══════════════════════════════════════════════════════════════════════════════
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
