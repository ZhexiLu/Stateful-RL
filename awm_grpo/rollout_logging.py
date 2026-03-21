from __future__ import annotations

import json
import os
import statistics
import threading
import time
from pathlib import Path
from typing import Any

from slime.utils.types import Sample

_WRITE_LOCK = threading.Lock()
_DEFAULT_TRACE_DIR = Path(__file__).resolve().parent / "logs" / "rollouts"
_DEFAULT_LIVE_TRACE_DIR = Path(__file__).resolve().parent / "logs" / "rollouts_live"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Sample.Status):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_to_jsonable(v) for v in value)
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


def _get_trace_dir(args: Any) -> Path:
    configured = os.environ.get("AWM_ROLLOUT_TRACE_DIR") or getattr(args, "rollout_trace_dir", None)
    return Path(configured).expanduser() if configured else _DEFAULT_TRACE_DIR


def _get_live_trace_dir(args: Any) -> Path:
    configured = os.environ.get("AWM_LIVE_ROLLOUT_TRACE_DIR") or getattr(args, "live_rollout_trace_dir", None)
    return Path(configured).expanduser() if configured else _DEFAULT_LIVE_TRACE_DIR


def _sample_record(
    sample: Sample,
    *,
    rollout_id: int | None,
    rollout_extra_metrics: dict[str, Any] | None,
    rollout_time: float | None,
    event: str,
) -> dict:
    metadata = dict(sample.metadata or {})
    trajectory = metadata.pop("trajectory", [])
    verify_result = metadata.pop("verify_result", None)
    num_iterations = metadata.pop("num_iterations", None)

    return {
        "event": event,
        "rollout_id": rollout_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid": os.getpid(),
        "sample_index": sample.index,
        "group_index": sample.group_index,
        "scenario": metadata.get("scenario"),
        "task_idx": metadata.get("task_idx"),
        "task": metadata.get("task"),
        "status": sample.status.value,
        "reward": _to_jsonable(sample.reward),
        "response_length": sample.response_length,
        "effective_response_length": sample.effective_response_length,
        "non_generation_time": sample.non_generation_time,
        "prompt": _to_jsonable(sample.prompt),
        "response": sample.response,
        "trajectory": _to_jsonable(trajectory),
        "verify_result": _to_jsonable(verify_result),
        "num_iterations": num_iterations,
        "rollout_metrics": _to_jsonable(rollout_extra_metrics or {}),
        "rollout_time": rollout_time,
        "metadata": _to_jsonable(metadata),
    }


def _build_record(rollout_id: int, sample: Sample, rollout_extra_metrics: dict[str, Any] | None, rollout_time: float) -> dict:
    return _sample_record(
        sample,
        rollout_id=rollout_id,
        rollout_extra_metrics=rollout_extra_metrics,
        rollout_time=rollout_time,
        event="rollout_complete",
    )


def log_live_sample_data(args: Any, sample: Sample, event: str = "sample_complete") -> None:
    trace_dir = _get_live_trace_dir(args)
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"live_samples_pid{os.getpid()}.jsonl"
    record = _sample_record(
        sample,
        rollout_id=None,
        rollout_extra_metrics=None,
        rollout_time=None,
        event=event,
    )

    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_rollout_data(
    rollout_id: int,
    args: Any,
    samples: list[Sample],
    rollout_extra_metrics: dict[str, Any] | None,
    rollout_time: float,
) -> bool:
    trace_dir = _get_trace_dir(args)
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"rollout_{rollout_id:06d}.jsonl"
    summary_path = trace_dir / f"rollout_{rollout_id:06d}.summary.json"

    with _WRITE_LOCK:
        records = [_build_record(rollout_id, sample, rollout_extra_metrics, rollout_time) for sample in samples]
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        timing_keys = sorted({k for record in records for k in record["metadata"].get("timing", {})})
        summary = {
            "rollout_id": rollout_id,
            "num_samples": len(records),
            "rollout_time": rollout_time,
            "mean_reward": statistics.mean(record["reward"] for record in records) if records else 0.0,
            "mean_response_length": statistics.mean(record["response_length"] for record in records) if records else 0.0,
            "mean_non_generation_time": (
                statistics.mean(record["non_generation_time"] for record in records) if records else 0.0
            ),
            "mean_tool_call_count": (
                statistics.mean(record["metadata"].get("tool_call_count", 0) for record in records) if records else 0.0
            ),
            "timing_mean": {
                key: statistics.mean(record["metadata"].get("timing", {}).get(key, 0.0) for record in records)
                for key in timing_keys
            },
            "rollout_metrics": _to_jsonable(rollout_extra_metrics or {}),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return False
