#!/bin/bash
#
# Run prompt comparison evaluation on 8×H200
#
# Usage (on the GPU node):
#   bash examples/awm_esa/run_eval.sh
#
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="${SCRIPT_DIR}/../.."
AWM_DIR="${SLIME_DIR}/../agent-world-model"

export PYTHONPATH="${AWM_DIR}:${SCRIPT_DIR}:${SLIME_DIR}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1
# Suppress noisy OpenTelemetry context warnings from MCP SDK
export OTEL_SDK_DISABLED=true
export PYTHONWARNINGS="ignore::DeprecationWarning"

# Clean up stale processes (ignore errors if nothing to kill)
pkill -9 -f "eval_server_" 2>/dev/null || true
pkill -9 -f "sglang.launch_server" 2>/dev/null || true
for port in $(seq 9200 9231); do
    fuser -k ${port}/tcp 2>/dev/null || true
done
sleep 2

set -ex

# Prepare DB directory in shared memory
mkdir -p /dev/shm/awm_eval_databases

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
MODEL_PATH="${SLIME_DIR}/../models/Qwen3-4B"

echo "Running eval: ${NUM_GPUS} GPUs, 200 tasks × 8 repeats × 2 prompts = 3200 rollouts"

python3 "${SCRIPT_DIR}/eval_prompt_comparison.py" \
    --model_path "${MODEL_PATH}" \
    --num_tasks 200 \
    --repeats 8 \
    --num_gpus "${NUM_GPUS}" \
    --num_workers 16 \
    --vllm_port 8000

echo "Done. Results in ${SCRIPT_DIR}/data/"
