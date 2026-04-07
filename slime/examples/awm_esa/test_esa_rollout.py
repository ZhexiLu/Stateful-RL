"""
Quick test: run a few ESA rollouts with Qwen3-4B via vLLM to inspect model behavior.

Matches training params exactly:
  - enable_thinking=False (no <think> block from chat template)
  - temperature=1.0
  - max_tokens=2048
  - ESA system prompt

Usage (on GPU node with vLLM already running on port 8001, or start a temp one):
  # Option A: use the judge vLLM (Qwen3.5-35B-A3B) — different model but quick test
  python examples/awm_esa/test_esa_rollout.py --port 8001

  # Option B: start Qwen3-4B on a free GPU temporarily
  CUDA_VISIBLE_DEVICES=6 python -m vllm.entrypoints.openai.api_server \
    --model ../models/Qwen3-4B --port 8002 --trust-remote-code &
  python examples/awm_esa/test_esa_rollout.py --port 8002
"""
import argparse
import json
import os
import sys
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from generate_with_esa import get_esa_system_prompt

# Sample tasks from training data
def load_sample_tasks(n=3):
    data_path = os.path.join(SCRIPT_DIR, "data", "train.jsonl")
    tasks = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            row = json.loads(line)
            # row["text"] is [{"role": "system", ...}, {"role": "user", ...}]
            user_msg = row["text"][1]["content"]
            meta = row.get("metadata", {})
            tasks.append({"task": user_msg, "scenario": meta.get("scenario", ""), "task_idx": meta.get("task_idx", 0)})
    return tasks


def run_one_rollout(api_url, model_name, task_text, max_turns=5, enable_thinking=False):
    """Run a multi-turn rollout matching training behavior exactly.

    Uses raw text completions API (not chat) with <status> prefix seeding
    on turn 2+, exactly like generate_with_esa.py does during training.
    """
    import re
    from transformers import AutoTokenizer

    # Build initial prompt via chat template (same as training)
    if not hasattr(run_one_rollout, "_tokenizer"):
        model_path = requests.get(f"{api_url}/v1/models", timeout=5).json()["data"][0]["id"]
        run_one_rollout._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = run_one_rollout._tokenizer

    system_prompt = get_esa_system_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_text},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    response = ""
    turns = []

    for turn_idx in range(max_turns):
        payload = {
            "model": model_name,
            "prompt": prompt_text + response,
            "temperature": 1.0,
            "max_tokens": 2048,
        }

        resp = requests.post(f"{api_url}/v1/completions", json=payload, timeout=120)
        result = resp.json()

        if "error" in result:
            print(f"  API error: {result['error']}")
            break

        cur_response = result["choices"][0]["text"]
        finish_reason = result["choices"][0]["finish_reason"]

        # Check for <status> tag
        status_match = re.search(r"<status>\s*(CONTINUE|VERIFY|REPLAN|STOP)\s*</status>", cur_response, re.IGNORECASE)
        # Also check seeded status (turn 2+ starts with "CONTINUE</status>" etc.)
        seeded_match = re.match(r"\s*(CONTINUE|VERIFY|REPLAN|STOP)\s*</status>", cur_response, re.IGNORECASE)
        has_status = bool(status_match) or bool(seeded_match)
        status_val = ""
        if status_match:
            status_val = status_match.group(1).upper()
        elif seeded_match:
            status_val = seeded_match.group(1).upper()

        # Check for tool call
        has_tool_call = "<tool_call>" in cur_response

        turns.append({
            "turn": turn_idx + 1,
            "has_status": has_status,
            "status": status_val,
            "has_tool_call": has_tool_call,
            "finish_reason": finish_reason,
            "length": len(cur_response),
            "content": cur_response,
        })

        # Print turn summary
        status_str = f"[{status_val}]" if has_status else "[NO STATUS]"
        tool_str = "tool_call" if has_tool_call else "no_tool"
        fin_str = f"(finish: {finish_reason})"
        print(f"  Turn {turn_idx+1}: status={status_str:12s} action={tool_str:10s} tokens={len(cur_response):4d}  {fin_str}")

        # Stop conditions
        if status_val == "STOP":
            response += cur_response
            break
        if not has_tool_call:
            response += cur_response
            break
        if finish_reason == "length":
            response += cur_response
            break

        # Append response + observation with <status> seed (matches training exactly)
        response += cur_response
        next_obs = (
            "<|im_start|>user\n<tool_response>\n"
            "Tool executed successfully. Result: {\"status\": \"ok\", \"data\": \"sample response\"}\n"
            "</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n<status>"
        )
        response += next_obs

    return turns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8002, help="vLLM port")
    parser.add_argument("--num-tasks", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking mode")
    args = parser.parse_args()

    api_url = f"http://127.0.0.1:{args.port}"

    # Detect model
    resp = requests.get(f"{api_url}/v1/models", timeout=5)
    model_name = resp.json()["data"][0]["id"]
    print(f"Model: {model_name}")
    print(f"Params: temperature=1.0, max_tokens=2048, enable_thinking={args.enable_thinking}")
    print()

    tasks = load_sample_tasks(args.num_tasks)
    all_task_turns = []

    for i, task_info in enumerate(tasks):
        print(f"{'='*80}")
        print(f"Task {i}: [{task_info['scenario']}::{task_info['task_idx']}]")
        print(f"  {task_info['task'][:100]}...")
        print(f"{'─'*80}")

        turns = run_one_rollout(api_url, model_name, task_info["task"], args.max_turns, args.enable_thinking)

        all_task_turns.extend(turns)

        # Print full content of every turn
        for t in turns:
            print(f"\n  ── Turn {t['turn']} full content ──")
            print(t["content"])

        print()

    # Summary (from already-collected results, no re-run)
    all_collected_turns = all_task_turns
    total = len(all_collected_turns)
    has_status_count = sum(1 for t in all_collected_turns if t["has_status"])
    print(f"\n{'='*80}")
    print(f"SUMMARY: {has_status_count}/{total} turns have <status> tag")


if __name__ == "__main__":
    main()
