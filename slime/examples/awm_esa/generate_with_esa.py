# ESA (Execution-Status Awareness) environment for SLIME RL training.
#
# SLIME entry points:
#   --custom-generate-function-path  generate_with_esa.generate
#   --custom-rm-path                 generate_with_esa.reward_func
#   --custom-rollout-log-function-path generate_with_esa.log_rollout_data

import asyncio
import json
import os
import shutil
import time

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# ── Internal modules ────────────────────────────────────────────────────────
from esa_config import ESA_CONFIGS, logger, get_esa_system_prompt  # noqa: F401
from esa_cache import _CACHE, _load_cache
from esa_server_pool import _init_pool, _acquire_slot, _release_slot, _ensure_server_ready
from esa_predicates import (
    _execute_tool_call, _parse_control_block,
    _evaluate_predicate_signature, _compute_progress_reward,
    # Re-exports for external scripts
    _evaluate_predicates, _is_write_operation,  # noqa: F401
)
from esa_rewards import (
    _compute_turn_level_returns, _compute_outcome_reward,
    # Re-exports for external scripts
    _compute_predicate_anchored_rewards, reward_func,  # noqa: F401
)

from awm.tools import normalize_scenario_name
from awm.core.agent import parse_tool_calls, parse_call_tool_arguments


# ═══════════════════════════════════════════════════════════════════════════════
# generate() — main ESA rollout function
# ═══════════════════════════════════════════════════════════════════════════════
async def generate(args, sample: Sample, sampling_params) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for ESA."

    await asyncio.to_thread(_load_cache)
    _init_pool()

    state = GenerateState(args)
    tokenizer = state.tokenizer
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    metadata = sample.metadata or {}
    scenario = normalize_scenario_name(metadata["scenario"])
    task_idx = metadata["task_idx"]

    prompt_text = sample.prompt
    prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    cfg = ESA_CONFIGS
    pred_key = f"{scenario}::{task_idx}"
    predicate_sqls = _CACHE.predicates.get(pred_key, {}).get("predicates", [])
    has_predicates = len(predicate_sqls) > 0

    max_retries = 3
    retry_timeout = 60.0
    last_error = None
    retry_start = time.monotonic()
    rollout_success = False

    for attempt in range(max_retries):
        # Reset all mutable state for each attempt
        response = ""
        response_token_ids = []
        loss_mask = []
        rollout_log_probs = [] if cfg["return_logprob"] else None
        turn_rewards = []
        turn_token_counts = []
        turn_signatures = []
        current_turn = 0
        prev_phi = 0.0
        prev_signature = ()
        prev_was_error = False
        format_error = False
        prev_tool_call = None
        sample.status = Sample.Status.PENDING

        slot = await _acquire_slot(scenario)

        try:
            await _ensure_server_ready(slot, scenario)

            initial_db = slot.db_path + ".init"
            await asyncio.to_thread(shutil.copy2, slot.db_path, initial_db)

            # Evaluate initial Phi(s_0) and signature
            if has_predicates:
                prev_phi, prev_signature = await asyncio.to_thread(
                    _evaluate_predicate_signature, initial_db, predicate_sqls)

            mcp = slot.mcp
            final_answer = ""
            max_ctx = getattr(args, "rollout_max_context_len", None) or 32768
            max_new = sampling_params.get("max_new_tokens", 2048)

            for _turn in range(cfg["max_turns"]):
                total_tokens = len(prompt_token_ids) + len(response_token_ids)
                remaining = max_ctx - total_tokens
                if remaining <= 0:
                    sample.status = Sample.Status.TRUNCATED
                    break

                cur_sampling_params = sampling_params.copy()
                cur_sampling_params["max_new_tokens"] = min(max_new, remaining)

                payload = {"text": prompt_text + response,
                           "sampling_params": cur_sampling_params}
                if cfg["return_logprob"]:
                    payload["return_logprob"] = True

                output = await post(url, payload)

                if output["meta_info"]["finish_reason"]["type"] == "abort":
                    sample.status = Sample.Status.ABORTED
                    break

                cur_response = output["text"]

                if cfg["return_logprob"]:
                    cur_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
                    cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
                else:
                    cur_token_ids = tokenizer(cur_response, add_special_tokens=False)["input_ids"]

                # ── ESA: Parse control block ──
                control_status, valid_format = _parse_control_block(cur_response)
                current_turn += 1
                turn_token_count = len(cur_token_ids)

                if cfg["enable_turn_level"]:
                    loss_mask += [current_turn] * turn_token_count
                else:
                    loss_mask += [1] * turn_token_count

                response += cur_response
                response_token_ids += cur_token_ids
                if rollout_log_probs is not None:
                    rollout_log_probs += cur_log_probs

                if output["meta_info"]["finish_reason"]["type"] == "length":
                    turn_signatures.append(prev_signature)
                    turn_rewards.append(0.0)
                    turn_token_counts.append(turn_token_count)
                    sample.status = Sample.Status.TRUNCATED
                    break

                # ── ESA: Status tag enforcement ──
                if valid_format:
                    turn_format_bonus = cfg["format_correct_bonus"]
                elif current_turn <= 1:
                    turn_format_bonus = 0.0  # first turn exempt (no status seed)
                else:
                    # Missing <status> on turn 2+ → hard terminate, same as tool call error
                    turn_signatures.append(prev_signature)
                    turn_rewards.append(cfg["missing_status_penalty"])
                    turn_token_counts.append(turn_token_count)
                    format_error = True
                    sample.status = Sample.Status.COMPLETED
                    break

                # ── ESA: Handle STOP ──
                if control_status == "STOP":
                    final_answer = cur_response
                    turn_signatures.append(prev_signature)
                    turn_rewards.append(turn_format_bonus)
                    turn_token_counts.append(turn_token_count)
                    sample.status = Sample.Status.COMPLETED
                    break

                # ── AWM-style format validation (strict) ──
                tool_calls = parse_tool_calls(cur_response)
                if not tool_calls:
                    turn_signatures.append(prev_signature)
                    turn_rewards.append(cfg["format_error_penalty"])
                    turn_token_counts.append(turn_token_count)
                    format_error = True
                    sample.status = Sample.Status.COMPLETED
                    break

                tc = tool_calls[0]
                tc_name = tc.get("name", "")

                if tc_name not in ("list_tools", "call_tool"):
                    turn_signatures.append(prev_signature)
                    turn_rewards.append(cfg["format_error_penalty"])
                    turn_token_counts.append(turn_token_count)
                    format_error = True
                    sample.status = Sample.Status.COMPLETED
                    break

                # ── First turn must be list_tools ──
                if current_turn == 1 and tc_name != "list_tools":
                    turn_signatures.append(prev_signature)
                    turn_rewards.append(cfg["format_error_penalty"])
                    turn_token_counts.append(turn_token_count)
                    format_error = True
                    sample.status = Sample.Status.COMPLETED
                    break

                if tc_name == "call_tool":
                    try:
                        tool_name, tool_args = parse_call_tool_arguments(tc["arguments"])
                        if not tool_name:
                            raise ValueError("empty tool_name")
                    except Exception:
                        turn_signatures.append(prev_signature)
                        turn_rewards.append(cfg["format_error_penalty"])
                        turn_token_counts.append(turn_token_count)
                        format_error = True
                        sample.status = Sample.Status.COMPLETED
                        break

                # ── Detect duplicate tool call ──
                current_tool_call = json.dumps(tc, sort_keys=True)
                is_duplicate = (current_tool_call == prev_tool_call)
                prev_tool_call = current_tool_call

                # ── Execute tool call ──
                obs_text, done, is_write = await _execute_tool_call(
                    mcp, cur_response, slot.tools_text)

                is_env_error = obs_text.startswith("Error:") and any(
                    kw in obs_text for kw in ["Internal Server Error", "500", "timed out"])

                turn_signatures.append(prev_signature)

                # ── ESA: Compute per-turn reward ──
                # Status bonus + tool call bonus (both correct to reach here)
                turn_reward = turn_format_bonus + cfg["format_correct_bonus"]

                if is_env_error:
                    prev_was_error = False
                elif has_predicates:
                    r_prog, new_phi, new_sig = await asyncio.to_thread(
                        _compute_progress_reward, slot.db_path, prev_phi, predicate_sqls)
                    turn_reward += r_prog
                    prev_was_error = (r_prog < 0) or ("Error" in obs_text)
                    prev_phi = new_phi
                    prev_signature = new_sig
                else:
                    prev_was_error = "Error" in obs_text

                if is_duplicate:
                    turn_reward += cfg["duplicate_penalty"]

                if control_status == "VERIFY" and is_write:
                    turn_reward += cfg["verify_violation_penalty"]

                turn_rewards.append(turn_reward)
                turn_token_counts.append(turn_token_count)

                if done:
                    final_answer = cur_response
                    sample.status = Sample.Status.COMPLETED
                    break

                # Append observation (loss_mask=0, no gradient)
                next_obs = (
                    f"<|im_start|>user\n<tool_response>\n{obs_text}\n</tool_response><|im_end|>\n"
                    f"<|im_start|>assistant\n<think>\n<status>"
                )
                obs_token_ids = tokenizer(next_obs, add_special_tokens=False)["input_ids"]

                response += next_obs
                response_token_ids += obs_token_ids
                loss_mask += [0] * len(obs_token_ids)
                if rollout_log_probs is not None:
                    rollout_log_probs += [0.0] * len(obs_token_ids)

                if len(prompt_token_ids) + len(response_token_ids) + 1 >= max_ctx:
                    sample.status = Sample.Status.TRUNCATED
                    break

            # ── Compute rewards ──
            if sample.status == Sample.Status.PENDING:
                sample.status = Sample.Status.COMPLETED

            if format_error:
                outcome_reward = cfg["format_error_penalty"]
                reward_source = "format_error"
            else:
                outcome_reward, reward_source = await _compute_outcome_reward(
                    scenario, task_idx, initial_db, slot.db_path, final_answer, response)

            eta = cfg["eta"]
            lam = cfg["lambda_prog"]

            if turn_rewards:
                turn_returns = _compute_turn_level_returns(
                    turn_rewards, outcome_reward, eta, lam)
                composite_reward = turn_returns[0] if turn_returns else outcome_reward
            else:
                composite_reward = outcome_reward
                turn_returns = []

            sample.reward = composite_reward
            sample.metadata["turn_rewards"] = turn_rewards
            sample.metadata["turn_token_counts"] = turn_token_counts
            sample.metadata["turn_returns"] = turn_returns
            sample.metadata["turn_signatures"] = [list(s) for s in turn_signatures]
            sample.metadata["outcome_reward"] = outcome_reward
            sample.metadata["reward_source"] = reward_source

            rollout_success = True
            break  # Success, exit retry loop

        except Exception as e:
            last_error = e
            elapsed = time.monotonic() - retry_start
            logger.warning(
                "ESA rollout attempt %d/%d failed for %s[%d] (%.1fs elapsed): %s",
                attempt + 1, max_retries, scenario, task_idx, elapsed, e,
            )
            if elapsed >= retry_timeout:
                break
            if attempt < max_retries - 1:
                await asyncio.sleep(min(3, 1 + attempt))

        finally:
            _release_slot(slot)
            initial_db = slot.db_path + ".init"
            if os.path.exists(initial_db):
                try:
                    os.remove(initial_db)
                except OSError:
                    pass

    if not rollout_success:
        logger.error(
            "ESA rollout failed after %d attempts (%.1fs) for %s[%d]: %s",
            attempt + 1, time.monotonic() - retry_start,
            scenario, task_idx, last_error, exc_info=True,
        )
        sample.reward = 0.0
        sample.remove_sample = True
        sample.metadata["infra_error"] = f"{type(last_error).__name__}: {last_error}"
        sample.status = Sample.Status.FAILED

    # Fill sample
    sample.tokens = prompt_token_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_mask
    sample.prompt = prompt_text
    if rollout_log_probs is not None:
        sample.rollout_log_probs = rollout_log_probs

    return sample


# ═══════════════════════════════════════════════════════════════════════════════
# Rollout Logging (--custom-rollout-log-function-path)
# ═══════════════════════════════════════════════════════════════════════════════
_ROLLOUT_LOG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "logs",
    time.strftime("%Y%m%d_%H%M%S")
)


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """Custom rollout log function. Returns False so default WandB logging also runs."""
    log_dir = _ROLLOUT_LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    filename = os.path.join(log_dir, f"rollout_{rollout_id:04d}.jsonl")
    with open(filename, "w") as f:
        for sample in samples:
            meta = sample.metadata or {}
            turn_rewards = meta.get("turn_rewards", [])
            turn_returns = meta.get("turn_returns", [])
            turn_token_counts = meta.get("turn_token_counts", [])
            turn_signatures = meta.get("turn_signatures", [])
            outcome_reward = meta.get("outcome_reward")

            record = {
                "scenario": meta.get("scenario", ""),
                "task_idx": meta.get("task_idx", ""),
                "task": meta.get("task", ""),
                "composite_reward": sample.reward,
                "outcome_reward": outcome_reward,
                "turn_rewards": turn_rewards,
                "turn_returns": turn_returns,
                "turn_token_counts": turn_token_counts,
                "turn_signatures": turn_signatures,
                "status": sample.status.value if sample.status else None,
                "reward_source": meta.get("reward_source"),
                "infra_error": meta.get("infra_error"),
                "response_length": sample.response_length,
                "num_turns": len(turn_rewards),
                "prompt": sample.prompt or "",
                "response": sample.response or "",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    n_total = len(samples)
    rewards = [s.reward for s in samples if s.reward is not None]
    outcomes = [s.metadata.get("outcome_reward", 0) for s in samples if s.metadata]
    n_completed = sum(1 for o in outcomes if o == 1.0)
    n_format_err = sum(1 for o in outcomes if o == ESA_CONFIGS["format_error_penalty"])
    n_infra_err = sum(1 for s in samples if s.metadata and s.metadata.get("infra_error"))
    n_code_fallback = sum(
        1 for s in samples
        if s.metadata and str(s.metadata.get("reward_source", "")).startswith("code_fallback")
    )
    n_llm_judge = sum(
        1 for s in samples
        if s.metadata and s.metadata.get("reward_source") == "llm_judge"
    )
    avg_turns = sum(
        len(s.metadata.get("turn_rewards", [])) for s in samples if s.metadata
    ) / max(1, n_total)

    # Write ESA metrics into rollout_extra_metrics for WandB
    if rollout_extra_metrics is None:
        rollout_extra_metrics = {}
    rollout_extra_metrics.update({
        "esa/completion_rate": n_completed / max(1, n_total),
        "esa/format_error_rate": n_format_err / max(1, n_total),
        "esa/infra_error_rate": n_infra_err / max(1, n_total),
        "esa/code_fallback_rate": n_code_fallback / max(1, n_total),
        "esa/llm_judge_rate": n_llm_judge / max(1, n_total),
        "esa/avg_turns": avg_turns,
        "esa/avg_reward": sum(rewards) / max(1, len(rewards)),
    })

    logger.info(
        "ESA rollout %d: %d samples, reward=%.3f, completion=%d/%d, format_err=%d, "
        "infra_err=%d, code_fallback=%d, time=%.1fs -> %s",
        rollout_id, n_total,
        sum(rewards) / max(1, len(rewards)),
        n_completed, n_total, n_format_err,
        n_infra_err, n_code_fallback, rollout_time, filename,
    )
    return False  # Continue with SLIME's default WandB logging
