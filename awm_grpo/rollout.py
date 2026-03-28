"""
Custom multi-turn rollout for AWM environments in slime.

This rollout function drives the agent-environment loop:
  1. LLM generates response (via SGLang /generate)
  2. AWMEnv parses tool calls and executes via MCP
  3. Tool response is tokenized and appended to context
  4. Repeat until done or max turns reached
  5. Run AWM verification to compute reward
"""
from __future__ import annotations

import faulthandler, sys
faulthandler.enable(file=sys.stderr, all_threads=True)

import asyncio
import logging
import os
import sys
import threading
import time
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# Add awm_grpo to path
AWMGRPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if AWMGRPO_ROOT not in sys.path:
    sys.path.insert(0, AWMGRPO_ROOT)

from env_awm import AWMEnv, build_env, preload_awm_data, _DATA_CACHE
from rollout_logging import log_live_sample_data

logger = logging.getLogger(__name__)

_AWM_INITIALIZED = False
_AWM_INIT_LOCK = threading.Lock()
_AWM_SAMPLE_SEMAPHORE: asyncio.Semaphore | None = None
_AWM_SAMPLE_SEMAPHORE_LIMIT: int | None = None
_AWM_SAMPLE_SEMAPHORE_LOCK = threading.Lock()


def _ensure_awm_initialized(args):
    """Lazily initialize AWM data cache on first rollout call."""
    global _AWM_INITIALIZED
    if _AWM_INITIALIZED:
        return

    with _AWM_INIT_LOCK:
        if _AWM_INITIALIZED:
            return

        # Read AWM paths from custom config or args
        awm_root = getattr(args, "awm_root", "../agent-world-model")
        db_schema_path = getattr(args, "awm_db_schema_path",
                                 os.path.join(awm_root, "outputs/gen_db.jsonl"))
        sample_path = getattr(args, "awm_sample_path",
                              os.path.join(awm_root, "outputs/gen_sample.jsonl"))
        verifier_path = getattr(args, "awm_verifier_path",
                                os.path.join(awm_root, "outputs/gen_verifier.pure_code.jsonl"))
        envs_path = getattr(args, "awm_envs_path",
                            os.path.join(awm_root, "outputs/gen_envs.jsonl"))
        db_dir = getattr(args, "awm_db_dir",
                         os.path.join(awm_root, "outputs/databases"))

        logger.info(f"Initializing AWM data cache from {awm_root}")
        preload_awm_data(db_schema_path, sample_path, verifier_path, envs_path, db_dir)
        _AWM_INITIALIZED = True
        logger.info("AWM data cache initialized")


def _get_awm_sample_semaphore(args: Any) -> asyncio.Semaphore:
    """Bound concurrent AWM env rollouts to keep Ray actor / MCP load stable.

    The AWM semaphore is the *primary* concurrency gate for multi-turn rollouts.
    It should match the server pool capacity since each active rollout holds one
    MCP server slot for the full episode.  Setting ``awm_max_concurrent_samples``
    to 0 (default) auto-derives the limit from ``server_pool_max_servers``.
    """
    global _AWM_SAMPLE_SEMAPHORE
    global _AWM_SAMPLE_SEMAPHORE_LIMIT

    configured = getattr(args, "awm_max_concurrent_samples", None) or 0
    configured = int(configured)
    if configured <= 0:
        # Auto: match the server pool so every concurrent rollout can get a slot
        configured = max(1, int(getattr(args, "server_pool_max_servers", 8)))

    limit = max(1, configured)
    if _AWM_SAMPLE_SEMAPHORE is not None and _AWM_SAMPLE_SEMAPHORE_LIMIT == limit:
        return _AWM_SAMPLE_SEMAPHORE

    with _AWM_SAMPLE_SEMAPHORE_LOCK:
        if _AWM_SAMPLE_SEMAPHORE is None or _AWM_SAMPLE_SEMAPHORE_LIMIT != limit:
            _AWM_SAMPLE_SEMAPHORE = asyncio.Semaphore(limit)
            _AWM_SAMPLE_SEMAPHORE_LIMIT = limit
            logger.info("Set AWM sample concurrency limit to %s", limit)

    return _AWM_SAMPLE_SEMAPHORE


# Dummy messages for computing chat template trim length
DUMMY_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


def _encode_observation(
    tokenizer,
    message: dict,
    metadata: dict | None,
    apply_chat_template: bool,
    apply_chat_template_kwargs: dict | None,
) -> list[int]:
    """Encode an observation (tool response) message into token IDs.

    We tokenize the observation within the chat template context, then
    trim the prefix that corresponds to the prior conversation so only
    the new observation tokens remain.
    """
    tools = metadata.get("tools") if metadata else None
    apply_kwargs = apply_chat_template_kwargs or {}

    if apply_chat_template:
        dummy_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
            **apply_kwargs,
        )
        formatted_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES + [message],
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **apply_kwargs,
        )
        trim_length = len(tokenizer.encode(dummy_prompt, add_special_tokens=False))
        prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        prompt_ids = prompt_ids[trim_length:]
    else:
        text = message.get("content", "")
        prompt_ids = tokenizer.encode(text, add_special_tokens=False)

    return prompt_ids


async def _run_inference(url: str, tokens: list[int], sampling_params: dict) -> tuple[str, list[int], list[float], str]:
    """Call SGLang /generate endpoint and return response."""
    payload = {
        "input_ids": tokens,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    output = await post(url, payload)
    response_text = output["text"]

    if "output_token_logprobs" in output["meta_info"]:
        new_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        new_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    else:
        new_tokens, new_log_probs = [], []

    finish_type = output["meta_info"]["finish_reason"]["type"]
    return response_text, new_tokens, new_log_probs, finish_type


async def generate(args: Any, sample: Sample, sampling_params: dict) -> Sample:
    """
    Custom multi-turn rollout for AWM tool-calling environments.

    This function:
    1. Creates an AWMEnv from sample metadata
    2. Drives the multi-turn LLM ↔ env loop via SGLang
    3. Runs AWM verification to compute reward
    4. Returns the sample with tokens, log_probs, loss_mask, reward populated
    """
    env = None
    if sample.metadata is None:
        sample.metadata = {}
    timing = {
        "reset": 0.0,
        "generate": 0.0,
        "env_step": 0.0,
        "encode_obs": 0.0,
        "verify": 0.0,
    }
    tool_call_count = 0

    # Ensure AWM data is loaded
    await asyncio.to_thread(_ensure_awm_initialized, args)
    sample_semaphore = _get_awm_sample_semaphore(args)

    state = GenerateState(args)
    tokenizer = state.tokenizer
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    max_turns = getattr(args, "max_turns", 15)

    # Build AWM environment
    env = build_env(sample=sample, args=args)

    # Tokenize prompt
    prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)
    if not sample.tokens:
        sample.tokens = list(prompt_ids)

    response_tokens: list[int] = []
    sample.loss_mask = []
    sample.rollout_log_probs = []
    sample.response_length = 0

    # Compute token budget
    budget = None
    if getattr(args, "rollout_max_context_len", None) is not None:
        budget = args.rollout_max_context_len - len(sample.tokens)
    elif sampling_params.get("max_new_tokens") is not None:
        budget = sampling_params["max_new_tokens"]

    sampling_params = sampling_params.copy()

    try:
        # Acquire the AWM semaphore to gate env-heavy work (MCP servers).
        # NOTE: slime's own semaphore (in generate_and_rm) gates how many
        # rollouts enter this function concurrently.  The AWM semaphore is an
        # additional cap tied to the MCP server pool size.  Keeping both in
        # sync (via config) avoids deadlocks while preventing OOM from too
        # many concurrent MCP processes.
        async with sample_semaphore:
            # Start environment
            start = time.perf_counter()
            await env.reset()
            timing["reset"] = time.perf_counter() - start

            if budget is not None and budget <= 0:
                sample.status = Sample.Status.TRUNCATED
                return sample

            for turn_idx in range(max_turns):
                # Update max_new_tokens based on remaining budget
                cur_params = sampling_params.copy()
                if budget is not None:
                    cur_params["max_new_tokens"] = max(1, budget)

                # LLM generates response
                start = time.perf_counter()
                response_text, new_tokens, new_log_probs, finish_type = await _run_inference(
                    url, sample.tokens, cur_params
                )
                timing["generate"] += time.perf_counter() - start

                # Append model-generated tokens (loss_mask = 1)
                sample.tokens.extend(new_tokens)
                response_tokens.extend(new_tokens)
                sample.loss_mask.extend([1] * len(new_tokens))
                sample.rollout_log_probs.extend(new_log_probs)
                sample.response_length = len(response_tokens)

                if budget is not None:
                    budget -= len(new_tokens)

                # Check finish reason
                if finish_type == "length":
                    sample.status = Sample.Status.TRUNCATED
                    break
                if finish_type == "abort":
                    sample.status = Sample.Status.ABORTED
                    break

                if budget is not None and budget <= 0:
                    sample.status = Sample.Status.TRUNCATED
                    break

                # Environment step: parse tool calls, execute, get observation
                start = time.perf_counter()
                obs, done, info = await env.step(response_text)
                timing["env_step"] += time.perf_counter() - start
                if info.get("tool_name") is not None:
                    tool_call_count += 1

                if done:
                    # If no tool call was detected, the final answer is in response_text
                    sample.status = Sample.Status.COMPLETED
                    break

                # Encode observation as tokens
                obs_message = {
                    "role": "tool" if obs.get("role") == "tool" else "user",
                    "content": obs.get("obs_str", ""),
                }
                if "tool_call_id" in obs:
                    obs_message["tool_call_id"] = obs["tool_call_id"]

                start = time.perf_counter()
                obs_ids = _encode_observation(
                    tokenizer, obs_message,
                    sample.metadata,
                    getattr(args, "apply_chat_template", False),
                    getattr(args, "apply_chat_template_kwargs", None),
                )
                timing["encode_obs"] += time.perf_counter() - start

                # Strip BOS if present
                bos_id = tokenizer.bos_token_id
                if bos_id is not None and obs_ids and obs_ids[0] == bos_id:
                    obs_ids = obs_ids[1:]

                # Append observation tokens (loss_mask = 0, log_prob = 0)
                sample.tokens.extend(obs_ids)
                response_tokens.extend(obs_ids)
                sample.loss_mask.extend([0] * len(obs_ids))
                sample.rollout_log_probs.extend([0.0] * len(obs_ids))
                sample.response_length = len(response_tokens)

                if budget is not None:
                    budget -= len(obs_ids)
                    if budget <= 0:
                        sample.status = Sample.Status.TRUNCATED
                        break

                # Check if we've reached max turns
                if turn_idx + 1 >= max_turns:
                    sample.status = Sample.Status.COMPLETED
                    break

            # Compute reward via AWM verification
            start = time.perf_counter()
            reward = await asyncio.to_thread(env.compute_reward)
            timing["verify"] = time.perf_counter() - start
            sample.reward = reward

        # ── outside semaphore: bookkeeping only ──
        sample.non_generation_time = (
            timing["reset"] + timing["env_step"] + timing["encode_obs"] + timing["verify"]
        )

        # Store trajectory in metadata for analysis
        sample.metadata["trajectory"] = env.trajectory
        sample.metadata["verify_result"] = env._verify_result
        sample.metadata["num_iterations"] = env._iteration
        sample.metadata["tool_call_count"] = tool_call_count
        sample.metadata["timing"] = {k: round(v, 6) for k, v in timing.items()}

        # Decode full response
        sample.response = tokenizer.decode(response_tokens, skip_special_tokens=False)

        if sample.status is None:
            sample.status = Sample.Status.COMPLETED

        await asyncio.to_thread(log_live_sample_data, args, sample, "sample_complete")

        return sample

    except Exception as e:
        scenario = getattr(env, "scenario", "unknown")
        task_idx = getattr(env, "task_idx", "unknown")
        logger.error(f"Rollout error for {scenario}[{task_idx}]: {e}", exc_info=False)
        sample.reward = 0.0
        sample.status = Sample.Status.FAILED
        sample.non_generation_time = (
            timing["reset"] + timing["env_step"] + timing["encode_obs"] + timing["verify"]
        )
        sample.metadata["trajectory"] = getattr(env, "trajectory", [])
        sample.metadata["verify_result"] = getattr(env, "_verify_result", None)
        sample.metadata["num_iterations"] = getattr(env, "_iteration", 0)
        sample.metadata["tool_call_count"] = tool_call_count
        sample.metadata["timing"] = {k: round(v, 6) for k, v in timing.items()}
        sample.response = tokenizer.decode(response_tokens, skip_special_tokens=False) if response_tokens else ""
        # Don't let logging failure kill the rollout
        try:
            await asyncio.to_thread(log_live_sample_data, args, sample, "sample_failed")
        except Exception:
            pass
        return sample

    finally:
        try:
            if env is not None:
                await env.close()
        except Exception:
            pass
