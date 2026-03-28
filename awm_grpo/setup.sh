#!/bin/bash
#
# AWM-GRPO One-Click Setup Script (uv-based)
#
# Tested on:
#   - OS: Ubuntu 22.04 / 24.04 (or equivalent Linux with CUDA 12.4+)
#   - GPU: H200 (4× recommended)
#   - CUDA: 12.4+
#   - Python: 3.12
#
# Usage:
#   cd Stateful-RL/awm_grpo
#   bash setup.sh                          # Full setup (create venv + install + download + convert)
#   bash setup.sh --skip-model-download    # Skip model download (if you already have Qwen3-4B)
#   bash setup.sh --skip-data-download     # Skip AWM data download
#   bash setup.sh --model-path /my/Qwen3-4B  # Use custom model path
#   bash setup.sh --num-gpus 2             # Override GPU count
#   bash setup.sh --venv /my/venv          # Custom venv path (default: ../.venv_awm_grpo)
#
# Two-phase setup for HPC (login node has network, compute node has GPU):
#   bash setup.sh --phase1                 # Login node: venv + deps + data prep (no GPU needed)
#   sbatch setup_slurm.sh                  # GPU node:   model conversion + GPU verification
#
# After setup completes, run training with:
#   source ../.venv_awm_grpo/bin/activate
#   bash run_awm_grpo.sh                   # Full training (200 rollouts, ~24h on 4×H200)
#   NUM_ROLLOUT=3 bash run_awm_grpo.sh     # Quick smoke test (~20 min)
#

set -e

# ============================================================================
# Configuration (override via env vars or flags)
# ============================================================================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SLIME_DIR="${PROJECT_ROOT}/slime"
AWM_DIR="${PROJECT_ROOT}/agent-world-model"
MEGATRON_DIR="${PROJECT_ROOT}/Megatron-LM"

VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv_awm_grpo}"
MODEL_NAME="${MODEL_NAME:-Qwen3-4B}"
MODEL_HF_REPO="${MODEL_HF_REPO:-Qwen/${MODEL_NAME}}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models/${MODEL_NAME}}"
MODEL_DIST_DIR="${MODEL_DIST_DIR:-${PROJECT_ROOT}/models/${MODEL_NAME}_torch_dist}"
AWM_DATA_REPO="${AWM_DATA_REPO:-Snowflake/AgentWorldModel-1K}"
AWM_OUTPUTS_DIR="${AWM_DIR}/outputs"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

SKIP_MODEL_DOWNLOAD=0
SKIP_DATA_DOWNLOAD=0
SKIP_DEPS=0
SKIP_CONVERT=0
PHASE=""

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --phase1)              PHASE="1"; shift ;;
        --phase2)              PHASE="2"; shift ;;
        --skip-model-download) SKIP_MODEL_DOWNLOAD=1; shift ;;
        --skip-data-download)  SKIP_DATA_DOWNLOAD=1; shift ;;
        --skip-deps)           SKIP_DEPS=1; shift ;;
        --skip-convert)        SKIP_CONVERT=1; shift ;;
        --model-path)          MODEL_DIR="$2"; SKIP_MODEL_DOWNLOAD=1; shift 2 ;;
        --model-dist-path)     MODEL_DIST_DIR="$2"; SKIP_CONVERT=1; shift 2 ;;
        --num-gpus)            NUM_GPUS="$2"; shift 2 ;;
        --venv)                VENV_DIR="$2"; shift 2 ;;
        --help|-h)
            head -35 "$0" | grep "^#" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Phase mode: auto-set skip flags
if [ "${PHASE}" = "1" ]; then
    # Login node: deps + data, skip GPU-dependent steps
    SKIP_CONVERT=1
    echo "=== Phase 1: Login node setup (no GPU required) ==="
elif [ "${PHASE}" = "2" ]; then
    # GPU node: model conversion only, skip downloads and deps
    SKIP_MODEL_DOWNLOAD=1
    SKIP_DATA_DOWNLOAD=1
    SKIP_DEPS=1
    echo "=== Phase 2: GPU node setup (model conversion + verification) ==="
fi

# ============================================================================
# Helpers
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

step_num=0
step() {
    step_num=$((step_num + 1))
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  Step ${step_num}: $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}!${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; exit 1; }

# ============================================================================
# Pre-flight checks
# ============================================================================
step "Pre-flight checks"

# Install uv if not present
if command -v uv &>/dev/null; then
    ok "uv $(uv --version | awk '{print $2}')"
else
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    ok "uv installed: $(uv --version)"
fi

command -v git &>/dev/null || fail "git not found"

# Check GPU (optional in phase 1)
if [ "${NUM_GPUS}" -lt 1 ]; then
    if [ "${PHASE}" = "1" ]; then
        warn "No GPUs detected (OK for phase 1, GPU needed in phase 2)"
        GPU_NAME="none"
        CUDA_VER="unknown"
    else
        fail "No GPUs detected. This training requires at least 1 GPU."
    fi
else
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    ok "${NUM_GPUS}× ${GPU_NAME}"
    CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+' || echo "unknown")
    ok "CUDA ${CUDA_VER}"
fi

# Check directories — auto-clone Megatron-LM if missing
[ -d "${SLIME_DIR}" ]   || fail "slime directory not found at ${SLIME_DIR}"
[ -d "${AWM_DIR}" ]     || fail "agent-world-model directory not found at ${AWM_DIR}"
if [ ! -d "${MEGATRON_DIR}" ]; then
    echo "  Megatron-LM not found, cloning from NVIDIA/Megatron-LM..."
    git clone --depth 1 https://github.com/NVIDIA/Megatron-LM.git "${MEGATRON_DIR}"
    # Pin to tested commit for reproducibility
    cd "${MEGATRON_DIR}"
    git fetch --depth 1 origin 3714d81d418c9f1bca4594fc35f9e8289f652862
    git checkout 3714d81d418c9f1bca4594fc35f9e8289f652862
    cd "${PROJECT_ROOT}"
    ok "Megatron-LM cloned (commit 3714d81)"
else
    ok "Megatron-LM found"
fi
ok "Project structure verified"

# ============================================================================
# Step: Create venv and install all dependencies via uv
# ============================================================================
if [ "${SKIP_DEPS}" = "0" ]; then
    step "Create venv and install dependencies (via uv)"

    if [ -d "${VENV_DIR}" ] && "${VENV_DIR}/bin/python3" -c "import torch, sglang, ray, slime, awm, mbridge, transformer_engine.pytorch" 2>/dev/null; then
        ok "Venv already complete at ${VENV_DIR}"
        source "${VENV_DIR}/bin/activate"
    else
        # Create fresh venv
        echo "  Creating venv at ${VENV_DIR} ..."
        uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}" --clear
        source "${VENV_DIR}/bin/activate"
        ok "Venv created (Python $(python3 --version | awk '{print $2}'))"

        # ── Phase 1: PyTorch (CUDA wheels from PyTorch index) ──────────────
        echo "  [1/6] Installing PyTorch + CUDA..."
        uv pip install \
            "torch==2.9.1" \
            "torchvision==0.24.1" \
            "torchaudio==2.9.1" \
            --index-url https://download.pytorch.org/whl/cu126
        ok "PyTorch installed"

        # ── Phase 2: flash-attn (needs torch + psutil + wheel at build time)
        echo "  [2/6] Installing flash-attn (may take a few minutes to build)..."
        uv pip install packaging wheel psutil pybind11
        uv pip install "flash-attn==2.8.3" --no-build-isolation
        ok "flash-attn installed"

        # ── Phase 3: SGLang + Ray ────────────────────────────────────────
        echo "  [3/6] Installing SGLang + Ray..."
        uv pip install \
            "sglang[all]==0.5.9" \
            "sglang-router==0.3.2" \
            "ray[default]==2.54.0"
        ok "SGLang + Ray installed"

        # ── Phase 4: Megatron-LM + mbridge (from source) ────────────────
        echo "  [4/6] Installing Megatron-LM + mbridge..."
        uv pip install -e "${MEGATRON_DIR}" --no-deps
        # megatron-bridge has conflicting transformer-engine version constraints
        # with megatron-core, so install without dependency resolution
        uv pip install "mbridge==0.15.1" --no-deps
        uv pip install "megatron-bridge>=0.3.0" --no-deps
        ok "Megatron + mbridge installed"

        # ── Phase 5: slime + agent-world-model (editable) ───────────────
        echo "  [5/6] Installing slime + agent-world-model..."
        uv pip install -e "${SLIME_DIR}" --no-deps
        uv pip install -e "${AWM_DIR}"
        ok "slime + AWM installed"

        # ── Phase 6: Remaining dependencies ──────────────────────────────
        echo "  [6/6] Installing remaining dependencies..."
        # Megatron requires numpy 1.x; transformer-engine needed for RoPE fusion
        uv pip install "numpy==1.26.4"
        # transformer-engine: megatron-bridge requires TE >=2.10.0,<2.13.0.
        # Install TE dependencies first, then TE packages with --no-deps to
        # avoid dependency resolution conflicts. If PyPI prebuilt .so has ABI
        # issues, fall back to building .so from source.
        if python3 -c "import transformer_engine.pytorch" 2>/dev/null; then
            ok "transformer-engine already available"
        else
            # ── Install TE runtime dependencies first ─────────────────────
            uv pip install onnxscript onnx_ir onnx nvdlfw-inspect pydantic

            # ── Try PyPI prebuilt install (fast, no nvcc needed) ──────────
            echo "  Installing transformer-engine from PyPI (no-deps)..."
            rm -rf "${VENV_DIR}/lib/python${PYTHON_VERSION}/site-packages/transformer_engine"* 2>/dev/null || true
            uv pip install \
                "transformer-engine==2.12.0" \
                "transformer-engine-cu12==2.12.0" \
                "transformer-engine-torch==2.12.0" \
                --no-deps --no-build-isolation 2>/dev/null || true

            if python3 -c "import transformer_engine.pytorch" 2>/dev/null; then
                ok "transformer-engine installed from PyPI"
            else
                # ── PyPI .so incompatible, build from source ──────────────
                warn "PyPI transformer-engine has ABI issues with current PyTorch, building from source (~5-20 min)..."
                if ! command -v nvcc &>/dev/null; then
                    fail "nvcc not found. Install CUDA toolkit or set PATH to include nvcc."
                fi
                TE_SITE="${VENV_DIR}/lib/python${PYTHON_VERSION}/site-packages/transformer_engine"
                # Build .so from source against current PyTorch
                NVTE_FRAMEWORK=pytorch uv pip install \
                    "transformer-engine[pytorch] @ git+https://github.com/NVIDIA/TransformerEngine.git@v2.12" \
                    --no-build-isolation --no-deps
                # Source build only produces .so; reinstall PyPI Python packages on top
                cp "${TE_SITE}/libtransformer_engine.so" /tmp/_te_core.so
                cp "${TE_SITE}/transformer_engine_torch.cpython"*.so /tmp/_te_torch.so
                rm -rf "${VENV_DIR}/lib/python${PYTHON_VERSION}/site-packages/transformer_engine"*
                uv pip install \
                    "transformer-engine==2.12.0" \
                    "transformer-engine-cu12==2.12.0" \
                    "transformer-engine-torch==2.12.0" \
                    --no-deps --no-build-isolation
                # Replace PyPI .so with source-built ones and remove incompatible wheel_lib
                rm -rf "${TE_SITE}/wheel_lib"
                cp /tmp/_te_core.so "${TE_SITE}/libtransformer_engine.so"
                cp /tmp/_te_torch.so "${TE_SITE}/transformer_engine_torch.cpython-312-x86_64-linux-gnu.so"
                rm -f /tmp/_te_core.so /tmp/_te_torch.so
                ok "transformer-engine built from source"
            fi
        fi
        uv pip install \
            "accelerate>=1.10.0" \
            "wandb>=0.25.0" \
            "tensorboard>=2.20.0" \
            "torch_memory_saver>=0.0.5" \
            "numba>=0.64.0" \
            "omegaconf>=2.3.0" \
            "memray>=1.19.0" \
            "ring_flash_attn>=0.1.0" \
            "qwen_vl_utils>=0.0.14" \
            "pylatexenc>=2.10" \
            "aiohttp-cors>=0.8.0" \
            "einops>=0.8.0"
        ok "All dependencies installed"
    fi

    # ── Verify critical imports ──────────────────────────────────────────
    echo "  Verifying imports..."
    VERIFY_RESULT=$(python3 -c "
import sys
failures = []
checks = [
    ('torch',           'import torch; assert torch.cuda.is_available()' if '${PHASE}' != '1' else 'import torch'),
    ('flash_attn',      'import flash_attn'),
    ('sglang',          'import sglang'),
    ('ray',             'import ray'),
    ('transformers',    'import transformers'),
    ('mbridge',         'import mbridge'),
    ('slime',           'import slime'),
    ('awm',             'import awm'),
    ('megatron.core',   'import megatron.core'),
    ('fastapi',         'import fastapi'),
    ('mcp',             'import mcp'),
    ('sqlalchemy',      'import sqlalchemy'),
]
for name, code in checks:
    try:
        exec(code)
    except Exception as e:
        failures.append(f'{name}: {e}')
if failures:
    print('FAIL')
    for f in failures:
        print(f'  {f}')
else:
    print('OK')
" 2>&1) || true

    if echo "${VERIFY_RESULT}" | grep -q "^OK$"; then
        ok "All critical imports verified"
    else
        warn "Some imports failed (non-fatal). Details:"
        echo "${VERIFY_RESULT}" | while IFS= read -r line; do echo "    ${line}"; done
        warn "You may need to rebuild some packages. Continuing setup..."
    fi
else
    step "Install dependencies (skipped)"
    source "${VENV_DIR}/bin/activate" 2>/dev/null || fail "Venv not found at ${VENV_DIR}. Run without --skip-deps first."
fi

# ============================================================================
# Step: Download AWM-1K data
# ============================================================================
if [ "${SKIP_DATA_DOWNLOAD}" = "0" ]; then
    step "Download AWM-1K dataset"

    if [ -f "${AWM_OUTPUTS_DIR}/gen_tasks.jsonl" ] && \
       [ -f "${AWM_OUTPUTS_DIR}/gen_envs.jsonl" ] && \
       [ -f "${AWM_OUTPUTS_DIR}/gen_verifier.pure_code.jsonl" ] && \
       [ -f "${AWM_OUTPUTS_DIR}/gen_db.jsonl" ]; then
        ok "AWM data already exists at ${AWM_OUTPUTS_DIR}"
    else
        echo "  Downloading from ${AWM_DATA_REPO}..."
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    '${AWM_DATA_REPO}',
    repo_type='dataset',
    local_dir='${AWM_OUTPUTS_DIR}',
)
print('Download complete')
"
        ok "AWM data downloaded"
    fi

    # Verify data files
    for f in gen_tasks.jsonl gen_envs.jsonl gen_verifier.pure_code.jsonl gen_db.jsonl gen_sample.jsonl; do
        if [ ! -f "${AWM_OUTPUTS_DIR}/${f}" ]; then
            fail "Missing ${f} in ${AWM_OUTPUTS_DIR}/"
        fi
    done
    ok "All required data files present"

    # Generate databases if missing
    if [ ! -d "${AWM_OUTPUTS_DIR}/databases" ] || [ "$(ls -1 ${AWM_OUTPUTS_DIR}/databases/*.db 2>/dev/null | wc -l)" -lt 100 ]; then
        echo "  Generating SQLite databases (this takes ~2 minutes)..."
        PYTHONPATH="${AWM_DIR}:${PYTHONPATH:-}" python3 -c "
from awm.tools import tools_jsonl_load, normalize_scenario_name
from awm.core.db import create_sqlite_database
from awm.core.sample import execute_sample_data
import os

os.makedirs('${AWM_OUTPUTS_DIR}/databases', exist_ok=True)
schemas = {normalize_scenario_name(x['scenario']): x for x in tools_jsonl_load('${AWM_OUTPUTS_DIR}/gen_db.jsonl')}
samples = {normalize_scenario_name(x['scenario']): x for x in tools_jsonl_load('${AWM_OUTPUTS_DIR}/gen_sample.jsonl')}

created = 0
for scenario, schema in schemas.items():
    db_path = f'${AWM_OUTPUTS_DIR}/databases/{scenario}.db'
    if os.path.exists(db_path):
        continue
    db_path, _, _, _ = create_sqlite_database(scenario, schema['db_schema'], '${AWM_OUTPUTS_DIR}/databases')
    if scenario in samples:
        execute_sample_data(db_path, samples[scenario]['sample_data'], scenario)
    created += 1
    if created % 100 == 0:
        print(f'  Created {created} databases...')
print(f'  Done. Created {created} new databases.')
"
        ok "Databases generated"
    else
        DB_COUNT=$(ls -1 ${AWM_OUTPUTS_DIR}/databases/*.db 2>/dev/null | wc -l)
        ok "Databases already exist (${DB_COUNT} files)"
    fi
else
    step "Download AWM-1K dataset (skipped)"
fi

# ============================================================================
# Step: Download model
# ============================================================================
if [ "${SKIP_MODEL_DOWNLOAD}" = "0" ]; then
    step "Download ${MODEL_NAME} model"

    if [ -d "${MODEL_DIR}" ] && [ "$(ls -1 ${MODEL_DIR}/*.safetensors 2>/dev/null | wc -l)" -gt 0 ]; then
        ok "Model already exists at ${MODEL_DIR}"
    else
        echo "  Downloading ${MODEL_HF_REPO}..."
        mkdir -p "$(dirname ${MODEL_DIR})"
        python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download('${MODEL_HF_REPO}', local_dir='${MODEL_DIR}')
print(f'Downloaded to: {path}')
"
        ok "Model downloaded to ${MODEL_DIR}"
    fi
else
    step "Download model (skipped)"
    if [ ! -d "${MODEL_DIR}" ]; then
        fail "Model directory not found: ${MODEL_DIR}"
    fi
    ok "Using model at ${MODEL_DIR}"
fi

# ============================================================================
# Step: Convert model to Megatron torch_dist format
# ============================================================================
if [ "${SKIP_CONVERT}" = "0" ]; then
    step "Convert model to Megatron format"

    if [ -d "${MODEL_DIST_DIR}" ] && [ -f "${MODEL_DIST_DIR}/latest_checkpointed_iteration.txt" ]; then
        ok "Converted checkpoint already exists at ${MODEL_DIST_DIR}"
    else
        echo "  Converting HF checkpoint → torch_dist (this takes ~2 minutes)..."

        # Load model architecture config
        source "${SLIME_DIR}/scripts/models/qwen3-4B.sh"

        PYTHONPATH="${MEGATRON_DIR}:${SLIME_DIR}:${PYTHONPATH:-}" \
        CUDA_DEVICE_MAX_CONNECTIONS=1 \
        python3 "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
            "${MODEL_ARGS[@]}" \
            --no-gradient-accumulation-fusion \
            --hf-checkpoint "${MODEL_DIR}" \
            --save "${MODEL_DIST_DIR}"

        ok "Model converted to ${MODEL_DIST_DIR}"
    fi
else
    step "Convert model (skipped)"
    if [ ! -d "${MODEL_DIST_DIR}" ]; then
        fail "Converted model not found: ${MODEL_DIST_DIR}"
    fi
    ok "Using converted model at ${MODEL_DIST_DIR}"
fi

# ============================================================================
# Step: Prepare training data
# ============================================================================
step "Prepare training data"

DATA_DIR="${SCRIPT_DIR}/data"

if [ -f "${DATA_DIR}/train.jsonl" ] && [ -f "${DATA_DIR}/val.jsonl" ]; then
    TRAIN_COUNT=$(wc -l < "${DATA_DIR}/train.jsonl")
    VAL_COUNT=$(wc -l < "${DATA_DIR}/val.jsonl")
    ok "Training data already exists (${TRAIN_COUNT} train, ${VAL_COUNT} val)"
else
    echo "  Generating train/val/test splits..."
    PYTHONPATH="${AWM_DIR}:${SCRIPT_DIR}:${PYTHONPATH:-}" python3 "${SCRIPT_DIR}/data_prep.py" \
        --tasks_path "${AWM_OUTPUTS_DIR}/gen_tasks.jsonl" \
        --envs_path "${AWM_OUTPUTS_DIR}/gen_envs.jsonl" \
        --verifier_path "${AWM_OUTPUTS_DIR}/gen_verifier.pure_code.jsonl" \
        --output_dir "${DATA_DIR}"
    ok "Training data generated"
fi

# ============================================================================
# Step: Write environment config
# ============================================================================
step "Write environment config"

CONFIG_FILE="${SCRIPT_DIR}/.env"
cat > "${CONFIG_FILE}" << EOF
# Auto-generated by setup.sh on $(date -Iseconds)
# Source this file or let run_awm_grpo.sh pick it up automatically.
export HF_CKPT="${MODEL_DIR}"
export REF_CKPT="${MODEL_DIST_DIR}"
export SAVE_DIR="${SCRIPT_DIR}/checkpoints/awm_grpo"
export NUM_GPUS=${NUM_GPUS}
EOF
ok "Config written to ${CONFIG_FILE}"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Environment:"
echo "    Venv:             ${VENV_DIR}"
echo "    Model (HF):       ${MODEL_DIR}"
echo "    Model (Megatron): ${MODEL_DIST_DIR}"
echo "    AWM Data:         ${AWM_OUTPUTS_DIR}"
echo "    Training Data:    ${DATA_DIR}"
echo "    Config:           ${CONFIG_FILE}"
echo "    GPUs:             ${NUM_GPUS}× ${GPU_NAME}"
echo ""
if [ "${PHASE}" = "1" ]; then
    echo "  Next step: submit GPU job for phase 2 (model conversion)"
    echo ""
    echo -e "    ${YELLOW}# Submit phase 2 to SLURM${NC}"
    echo "    cd ${SCRIPT_DIR}"
    echo "    sbatch setup_slurm.sh"
    echo ""
else
    echo "  Next steps:"
    echo ""
    echo -e "    ${YELLOW}# Activate the venv${NC}"
    echo "    source ${VENV_DIR}/bin/activate"
    echo ""
    echo -e "    ${YELLOW}# Quick smoke test (~20 minutes)${NC}"
    echo "    cd ${SCRIPT_DIR}"
    echo "    NUM_ROLLOUT=3 bash run_awm_grpo.sh"
    echo ""
    echo -e "    ${YELLOW}# Full training (~24 hours on 4× H200)${NC}"
    echo "    cd ${SCRIPT_DIR}"
    echo "    bash run_awm_grpo.sh"
    echo ""
fi
