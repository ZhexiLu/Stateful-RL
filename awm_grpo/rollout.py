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

import logging
import os
import sys
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# Add awm_grpo to path
AWMGRPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if AWMGRPO_ROOT not in sys.path:
    sys.path.insert(0, AWMGRPO_ROOT)

from env_awm import AWMEnv, build_env, preload_awm_data, _DATA_CACHE

logger = logging.getLogger(__name__)

_AWM_INITIALIZED = False


def _ensure_awm_initialized(args):
    """Lazily initialize AWM data cache on first rollout call."""
    global _AWM_INITIALIZED
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
    # Ensure AWM data is loaded
    _ensure_awm_initialized(args)

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
        # Start environment
        env.reset()

        if budget is not None and budget <= 0:
            sample.status = Sample.Status.TRUNCATED
            return sample

        for turn_idx in range(max_turns):
            # Update max_new_tokens based on remaining budget
            cur_params = sampling_params.copy()
            if budget is not None:
                cur_params["max_new_tokens"] = budget

            # LLM generates response
            response_text, new_tokens, new_log_probs, finish_type = await _run_inference(
                url, sample.tokens, cur_params
            )

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
            obs, done, info = env.step(response_text)

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

            obs_ids = _encode_observation(
                tokenizer, obs_message,
                sample.metadata,
                getattr(args, "apply_chat_template", False),
                getattr(args, "apply_chat_template_kwargs", None),
            )

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
        reward = env.compute_reward()
        sample.reward = reward

        # Store trajectory in metadata for analysis
        sample.metadata["trajectory"] = env.trajectory
        sample.metadata["verify_result"] = env._verify_result
        sample.metadata["num_iterations"] = env._iteration

        # Decode full response
        sample.response = tokenizer.decode(response_tokens, skip_special_tokens=False)

        if sample.status is None:
            sample.status = Sample.Status.COMPLETED

        return sample

    except Exception as e:
        logger.error(f"Rollout error for {env.scenario}[{env.task_idx}]: {e}")
        sample.reward = 0.0
        sample.status = Sample.Status.FAILED
        sample.response = tokenizer.decode(response_tokens, skip_special_tokens=False) if response_tokens else ""
        return sample

    finally:
        try:
            env.close()
        except Exception:
            pass
