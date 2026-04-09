# ESA reward computation: outcome reward (LLM judge), turn-level returns,
# predicate-anchored advantage, and the SLIME reward_func entry point.

import asyncio
import json
from collections import defaultdict

import aiohttp

from esa_config import ESA_CONFIGS, logger
from esa_cache import _CACHE

from awm.core.verifier import execute_verification_code, VerificationMode


# ═══════════════════════════════════════════════════════════════════════════════
# Turn-Level Return-to-Go
# ═══════════════════════════════════════════════════════════════════════════════
def _compute_turn_level_returns(
    turn_rewards: list[float],
    outcome_reward: float,
    eta: float,
    lambda_prog: float,
) -> list[float]:
    """Compute discounted return-to-go per turn.

    G_{t} = sum_{u=t}^{T} eta^{u-t} * r^(u)
    where r^(t) = lambda * R_prog^(t), r^(T) += R_out
    """
    T = len(turn_rewards)
    if T == 0:
        return []

    step_rewards = [lambda_prog * r for r in turn_rewards]
    step_rewards[-1] += outcome_reward

    returns = [0.0] * T
    returns[T - 1] = step_rewards[T - 1]
    for t in range(T - 2, -1, -1):
        returns[t] = step_rewards[t] + eta * returns[t + 1]

    return returns


# ═══════════════════════════════════════════════════════════════════════════════
# SQL Verifier + LLM Judge (Code-Augmented Outcome Reward)
# ═══════════════════════════════════════════════════════════════════════════════
def _run_sql_verifier(scenario, task_idx, initial_db, final_db):
    """Run SQL mode verifier to collect evidence for LLM judge."""
    key = f"{scenario}::{task_idx}"
    sv = _CACHE.sql_verifiers.get(key)
    if not sv or not sv["code"]:
        return {"status": "no_verifier"}

    try:
        result = execute_verification_code(
            python_code=sv["code"],
            function_name=sv["function_name"],
            initial_db_path=initial_db,
            mode=VerificationMode.sql,
            final_db_path=final_db,
        )
        return {
            "status": "executed",
            "results": result.get("result", {}),
            "success_criteria": sv["success_criteria"],
            "failure_criteria": sv["failure_criteria"],
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "success_criteria": sv.get("success_criteria", ""),
            "failure_criteria": sv.get("failure_criteria", ""),
        }


_JUDGE_PROMPT = """You are an impartial evaluator for tool-use agent task results. Based on the agent trajectory AND database verification evidence, classify the outcome.

Categories:
- Completed: task done, database confirms it.
- Environment Error: agent blocked by API bugs (500, 422 for valid params, route conflicts).
- Agent Error: agent made mistakes (wrong params, hallucination, incomplete).

Output valid JSON only:
{"reasoning": "<brief>", "classification": "<Completed | Environment Error | Agent Error>", "confidence": <0-1>}"""


def _run_code_verifier(scenario, task_idx, initial_db, final_db, final_answer):
    """Run pure-code verifier as fallback. Returns (reward, source_str)."""
    key = f"{scenario}::{task_idx}"
    verifier_item = _CACHE.verifiers.get(key)
    if not verifier_item:
        return 0.0, "code_no_verifier"

    code = verifier_item.get("verification", {}).get("code", "")
    if not code:
        return 0.0, "code_no_verifier"

    func_name = "verify_task_completion"
    for line in code.split("\n"):
        line_s = line.strip()
        if line_s.startswith("def verify_") and "(" in line_s:
            func_name = line_s.split("(")[0].replace("def ", "").strip()
            break

    mode = VerificationMode.code if "final_answer" in code else VerificationMode.sql

    try:
        result = execute_verification_code(
            python_code=code, function_name=func_name,
            initial_db_path=initial_db, mode=mode,
            final_db_path=final_db, final_answer=final_answer,
        )
        if result.get("execution_status") == "success":
            inner = result.get("result", {})
            if isinstance(inner, dict) and str(inner.get("result", "")).lower() == "complete":
                return 1.0, "code_complete"
        return 0.0, "code_incomplete"
    except Exception as e:
        logger.warning("Code verifier error for %s::%d: %s", scenario, task_idx, e)
        return 0.0, "code_error"


_judge_model_name = None


async def _get_judge_model_name(cfg):
    """Cache the judge model name to avoid repeated /v1/models calls."""
    global _judge_model_name
    if _judge_model_name is None:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{cfg['judge_url']}/v1/models") as resp:
                models = await resp.json()
        _judge_model_name = models["data"][0]["id"]
    return _judge_model_name


async def _call_llm_judge(cfg, payload):
    """Single LLM judge API call. Returns classification string or raises."""
    payload["model"] = await _get_judge_model_name(cfg)
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{cfg['judge_url']}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            result = await resp.json()

    raw = result["choices"][0]["message"]["content"]

    # Parse JSON from response (skip thinking text)
    json_text = raw.strip()
    last_brace = json_text.rfind("{")
    if last_brace >= 0:
        depth = 0
        for idx in range(last_brace, len(json_text)):
            if json_text[idx] == "{": depth += 1
            elif json_text[idx] == "}":
                depth -= 1
                if depth == 0:
                    json_text = json_text[last_brace:idx+1]
                    break

    parsed = json.loads(json_text)
    return parsed.get("classification", "Agent Error")


async def _compute_outcome_reward(scenario, task_idx, initial_db, final_db, final_answer, trajectory):
    """Outcome reward: LLM judge with 3 retries, fallback to code verifier.

    Returns (reward, reward_source) where reward_source is one of:
        "llm_judge", "code_fallback", "code_no_verifier", "code_error"
    """
    cfg = ESA_CONFIGS

    # Step 1: SQL verifier evidence (always needed for LLM judge)
    verification = await asyncio.to_thread(
        _run_sql_verifier, scenario, task_idx, initial_db, final_db)

    if verification["status"] == "executed":
        results_str = json.dumps(verification.get("results", {}), indent=2, default=str)
        if len(results_str) > 2000:
            results_str = results_str[:2000] + "\n... [truncated]"
        evidence = (
            f"Database evidence:\n{results_str}\n\n"
            f"Success criteria: {verification.get('success_criteria', 'N/A')}\n"
            f"Failure criteria: {verification.get('failure_criteria', 'N/A')}"
        )
    else:
        evidence = f"Verification failed: {verification.get('error', 'no verifier')}"

    traj = trajectory
    if len(traj) > 4000:
        traj = traj[:1500] + "\n\n... [truncated] ...\n\n" + traj[-2500:]

    user_msg = (
        f"Task: {_CACHE.sql_verifiers.get(f'{scenario}::{task_idx}', {}).get('reasoning', '')}\n\n"
        f"Trajectory:\n{traj}\n\n{evidence}"
    )

    payload = {
        "model": "",
        "messages": [
            {"role": "system", "content": _JUDGE_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "max_tokens": 2048,
        "chat_template_kwargs": {"enable_thinking": True},
    }

    # Step 2: LLM judge (1 attempt, fallback to code on failure)
    reward_map = {"Completed": 1.0, "Environment Error": 0.0, "Agent Error": 0.0}

    judge_error = None
    try:
        classification = await _call_llm_judge(cfg, payload)
        return reward_map.get(classification, 0.0), "llm_judge"
    except Exception as e:
        judge_error = e
        logger.warning(
            "LLM judge failed for %s::%d: %s", scenario, task_idx, e,
        )

    # Step 3: Fallback to code verifier
    logger.warning(
        "LLM judge failed for %s::%d, falling back to code verifier. Last error: %s",
        scenario, task_idx, judge_error,
    )
    reward, source = await asyncio.to_thread(
        _run_code_verifier, scenario, task_idx, initial_db, final_db, final_answer)
    return reward, f"code_fallback_{source}"


# ═══════════════════════════════════════════════════════════════════════════════
# Predicate-Anchored Advantage Estimation
# ═══════════════════════════════════════════════════════════════════════════════
def _compute_predicate_anchored_rewards(samples: list) -> list[float]:
    """Compute rewards with predicate-anchored step-level advantage.

    Groups (rollout, turn) pairs by predicate signature, computes
    group-relative advantage, then: adjusted = outcome + omega * mean(step_adv)
    """
    cfg = ESA_CONFIGS
    omega = cfg.get("omega_step", 0.3)
    eta = cfg["eta"]
    lam = cfg["lambda_prog"]

    all_entries = []
    for i, sample in enumerate(samples):
        meta = sample.metadata or {}
        turn_rewards = meta.get("turn_rewards", [])
        turn_sigs = meta.get("turn_signatures", [])
        outcome = meta.get("outcome_reward", 0.0)

        if not turn_rewards:
            continue

        T = len(turn_rewards)
        step_r = [lam * r for r in turn_rewards]
        step_r[-1] += outcome

        rtg = [0.0] * T
        rtg[T - 1] = step_r[T - 1]
        for t in range(T - 2, -1, -1):
            rtg[t] = step_r[t] + eta * rtg[t + 1]

        for t in range(T):
            sig = tuple(turn_sigs[t]) if t < len(turn_sigs) else ()
            all_entries.append((i, t, sig, rtg[t]))

    if not all_entries:
        return [s.reward if s.reward is not None else 0.0 for s in samples]

    # Group by signature
    sig_groups = defaultdict(list)
    for rollout_idx, turn_idx, sig, g_val in all_entries:
        sig_groups[sig].append((rollout_idx, turn_idx, g_val))

    # Compute group-relative advantage
    step_advantages = {}
    for sig, entries in sig_groups.items():
        if len(entries) < 2:
            for ri, ti, g in entries:
                step_advantages[(ri, ti)] = 0.0
            continue

        g_values = [g for _, _, g in entries]
        mu = sum(g_values) / len(g_values)
        var = sum((g - mu) ** 2 for g in g_values) / len(g_values)
        std = var ** 0.5 + 1e-8

        for ri, ti, g in entries:
            step_advantages[(ri, ti)] = (g - mu) / std

    # Aggregate per-rollout
    rollout_step_advs = defaultdict(list)
    for (ri, ti), adv in step_advantages.items():
        rollout_step_advs[ri].append(adv)

    adjusted_rewards = []
    for i, sample in enumerate(samples):
        outcome = (sample.metadata or {}).get("outcome_reward", 0.0)
        if outcome is None:
            outcome = 0.0
        step_advs = rollout_step_advs.get(i, [])
        mean_step_adv = sum(step_advs) / len(step_advs) if step_advs else 0.0
        adjusted_rewards.append(outcome + omega * mean_step_adv)

    return adjusted_rewards


async def reward_func(args, samples_or_sample, **kwargs):
    """SLIME reward function (--custom-rm-path).

    Group mode (--group-rm): predicate-anchored advantage with infra-failure handling.
    Single mode: return pre-computed reward.
    """
    if isinstance(samples_or_sample, list):
        samples = samples_or_sample
        rewards = _compute_predicate_anchored_rewards(samples)

        # Neutralize infra-failed samples so they don't distort GRPO normalization
        valid_mask = [s.metadata.get("infra_error") is None for s in samples]
        valid_rewards = [r for r, v in zip(rewards, valid_mask) if v]
        if valid_rewards and not all(valid_mask):
            group_mean = sum(valid_rewards) / len(valid_rewards)
            rewards = [r if v else group_mean for r, v in zip(rewards, valid_mask)]

        return rewards
    else:
        sample = samples_or_sample
        if sample.reward is not None:
            return sample.reward
        return 0.0
