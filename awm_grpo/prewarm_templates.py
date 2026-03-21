from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from awm.tools import normalize_scenario_name
from env_awm import _ensure_template_db, preload_awm_data


def _iter_scenarios(path: Path):
    with path.open() as f:
        for line in f:
            record = json.loads(line)
            metadata = record.get("metadata", {})
            scenario = metadata.get("scenario")
            if scenario:
                yield normalize_scenario_name(scenario)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prebuild AWM SQLite template databases for rollout scenarios.")
    parser.add_argument("--prompt-data", type=Path, nargs="+", required=True)
    parser.add_argument("--db-schema-path", type=Path, required=True)
    parser.add_argument("--sample-path", type=Path, required=True)
    parser.add_argument("--verifier-path", type=Path, required=True)
    parser.add_argument("--envs-path", type=Path, required=True)
    parser.add_argument("--db-dir", type=Path, required=True)
    args = parser.parse_args()

    preload_awm_data(
        str(args.db_schema_path),
        str(args.sample_path),
        str(args.verifier_path),
        str(args.envs_path),
        str(args.db_dir),
    )

    scenarios = []
    seen = set()
    for path in args.prompt_data:
        for scenario in _iter_scenarios(path):
            if scenario not in seen:
                seen.add(scenario)
                scenarios.append(scenario)

    start = time.perf_counter()
    for idx, scenario in enumerate(scenarios, start=1):
        _ensure_template_db(scenario)
        if idx % 50 == 0 or idx == len(scenarios):
            elapsed = time.perf_counter() - start
            print(f"prewarmed {idx}/{len(scenarios)} scenarios in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
