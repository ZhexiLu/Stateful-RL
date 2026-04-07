#!/bin/bash
#
# ESA (Execution-Status Awareness) RL training with Qwen3-4B on slime.
#
# Three-step workflow:
#   1. Extract predicates:  python examples/awm_esa/predicate_extractor.py
#   2. Prepare data:        python examples/awm_esa/data_prep.py
#   3. Run training:        bash examples/awm_esa/run_qwen3_4B.sh
#
# Usage:
#   cd slime/
#   bash examples/awm_esa/run_qwen3_4B.sh

# Clean up stale processes (silent)
pkill -9 sglang 2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 -f "server_.*\.py" 2>/dev/null || true
for port in $(seq 9100 9131); do fuser -k ${port}/tcp 2>/dev/null || true; done
sleep 3

export PYTHONUNBUFFERED=1
export CC=gcc
export OTEL_SDK_DISABLED=true
export PYTHONWARNINGS="ignore::DeprecationWarning"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="${SCRIPT_DIR}/../.."
AWM_DIR="${SLIME_DIR}/../agent-world-model"
MEGATRON_DIR="${SLIME_DIR}/../Megatron-LM"
LOG_DIR="${SCRIPT_DIR}/logs"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/run_qwen3_4B_${RUN_TS}.log}"
VLLM_LOG="${VLLM_LOG:-${LOG_DIR}/vllm_judge_${RUN_TS}.log}"

mkdir -p "${LOG_DIR}"
ln -sfn "$(basename "${RUN_LOG}")" "${LOG_DIR}/latest_run_qwen3_4B.log"
ln -sfn "$(basename "${VLLM_LOG}")" "${LOG_DIR}/latest_vllm_judge.log"

if [ -z "${RUN_QWEN3_4B_LOGGING_INITIALIZED:-}" ]; then
    export RUN_QWEN3_4B_LOGGING_INITIALIZED=1
    exec > >(tee -a "${RUN_LOG}") 2>&1
fi

echo "Run log: ${RUN_LOG}"
echo "Judge log: ${VLLM_LOG}"

source "${SLIME_DIR}/scripts/models/qwen3-4B.sh"

# ── GPU layout: 6 GPUs for training (0-5), 2 GPUs for LLM judge (6-7) ────────
TOTAL_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NUM_GPUS=6  # Reserve GPU 6,7 for LLM judge
JUDGE_PORT=8001
JUDGE_MODEL="${SLIME_DIR}/../models/Qwen3.5-35B-A3B"
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_MEM_GB=$((GPU_MEM_MB / 1024))

if [ "${GPU_MEM_GB}" -ge 120 ]; then
    MEM_FRAC=0.50
elif [ "${GPU_MEM_GB}" -ge 70 ]; then
    MEM_FRAC=0.45
else
    MEM_FRAC=0.35
fi
MAX_RESP=2048; MAX_TOK=2048
echo "GPU: ${NUM_GPUS}x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) (${GPU_MEM_GB}GB)"

# ── Step 1: Extract predicates (if not already done) ────────────────────────
PRED_FILE="${SCRIPT_DIR}/data/predicates.jsonl"
if [ ! -f "${PRED_FILE}" ]; then
    echo "Extracting predicates..."
    python3 "${SCRIPT_DIR}/predicate_extractor.py" \
        --awm_root "${AWM_DIR}" \
        --output "${PRED_FILE}"
fi

# ── Step 2: Prepare data (if not already done) ──────────────────────────────
TRAIN_DATA="${SCRIPT_DIR}/data/train.jsonl"
if [ ! -f "${TRAIN_DATA}" ]; then
    echo "Preparing training data..."
    python3 "${SCRIPT_DIR}/data_prep.py" \
        --awm_root "${AWM_DIR}" \
        --output_dir "${SCRIPT_DIR}/data"
fi

# ── Paths ────────────────────────────────────────────────────────────────────
HF_CKPT="${HF_CKPT:-${SLIME_DIR}/../models/Qwen3-4B}"
REF_CKPT="${REF_CKPT:-${SLIME_DIR}/../models/Qwen3-4B_torch_dist}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${REF_CKPT}"
   --load "${SLIME_DIR}/../checkpoints/awm_esa/"
   --save "${SLIME_DIR}/../checkpoints/awm_esa/"
   --save-interval 1
)

ROLLOUT_ARGS=(
   --prompt-data "${TRAIN_DATA}"
   --input-key text
   --metadata-key metadata
   --apply-chat-template
   --num-rollout 3  # TODO: change back to 96 for full training
   --rollout-batch-size 64
   --n-samples-per-prompt 16
   --rollout-max-response-len "${MAX_RESP}"
   --rollout-max-context-len 32768
   --rollout-temperature 1
   --global-batch-size 1023  # must be divisible by data_parallel_size=3 (6 GPUs, CP=2)
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOK}"
)

# Standard GRPO — compatible with ESA's composite scalar reward
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 7e-7
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project awm-esa
   --wandb-group qwen3-4B-esa
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static "${MEM_FRAC}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_esa.generate
   --custom-rm-path generate_with_esa.reward_func
   --custom-rollout-log-function-path generate_with_esa.log_rollout_data
   --group-rm
)

# ── Prepare DB directory in /dev/shm (RAM) ──────────────────────────────────
mkdir -p "/dev/shm/awm_databases"

# ── Launch LLM Judge on GPU 6,7 ─────────────────────────────────────────────
_check_judge() { python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${JUDGE_PORT}/v1/models', timeout=2)" 2>/dev/null; }

if _check_judge; then
    echo "Judge server already running on port ${JUDGE_PORT}"
else
    echo "Starting LLM judge on GPU 6,7..."
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    sleep 2
    eval "$(micromamba shell hook -s bash)"
    micromamba activate vllm
    export LD_LIBRARY_PATH=/mnt/home/zlu10/micromamba/envs/vllm/lib:$LD_LIBRARY_PATH
    CUDA_VISIBLE_DEVICES=6,7 nohup python -m vllm.entrypoints.openai.api_server \
        --model "${JUDGE_MODEL}" \
        --tensor-parallel-size 2 \
        --port "${JUDGE_PORT}" \
        --trust-remote-code \
        > "${VLLM_LOG}" 2>&1 &
    while ! _check_judge; do sleep 5; done
    echo "Judge server ready on port ${JUDGE_PORT}"
    micromamba activate slime
fi

set -ex

# ── Launch Ray (only sees GPU 0-5) ──────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export RAY_grpc_client_keepalive_time_ms=30000
export RAY_grpc_client_keepalive_timeout_ms=60000
export RAY_grpc_server_keepalive_time_ms=30000
export RAY_grpc_server_keepalive_timeout_ms=60000
export RAY_health_check_period_ms=30000
export RAY_health_check_timeout_ms=60000
export RAY_num_heartbeats_timeout=20

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_DIR}:${SCRIPT_DIR}:${AWM_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"CC\": \"gcc\",
    \"OTEL_SDK_DISABLED\": \"true\",
    \"PYTHONWARNINGS\": \"ignore::DeprecationWarning\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${NUM_GPUS}" \
   --rollout-num-gpus "${NUM_GPUS}" \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   --apply-chat-template-kwargs '{"enable_thinking": false}'
