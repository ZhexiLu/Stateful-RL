# AWM-GRPO: Multi-Turn Tool-Use RL Training on AWM Environments

Vanilla GRPO training for tool-calling agents on [AgentWorldModel-1K](https://huggingface.co/datasets/Snowflake/AgentWorldModel-1K) environments, powered by [slime](https://github.com/THUDM/slime) (Megatron + SGLang + Ray).

**TL;DR**: Trains a language model (Qwen3-4B) to use tools in 1,000 SQL-backed environments via multi-turn reinforcement learning. The model learns to call MCP tools, read results, and iteratively solve tasks — rewarded only when verification code confirms the database reached the goal state.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  slime (Ray + Megatron + SGLang)                             │
│                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────┐  │
│  │ Megatron │    │   SGLang     │    │  Custom Rollout    │  │
│  │ Training │◄──►│   Engines    │◄──►│  (rollout.py)      │  │
│  │ (GRPO)   │    │  ×4 (TP=1)  │    │  multi-turn loop   │  │
│  └──────────┘    └──────────────┘    └────────┬───────────┘  │
│                                               │              │
└───────────────────────────────────────────────┼──────────────┘
                                                │
                               ┌────────────────▼────────────────┐
                               │  AWM Environment (env_awm.py)   │
                               │                                 │
                               │  ┌──────────┐  ┌────────────┐  │
                               │  │ SQLite   │  │ MCP Server │  │
                               │  │ Database │◄─│ (FastAPI)  │  │
                               │  └──────────┘  └────────────┘  │
                               │                                 │
                               │  ┌──────────────────────────┐   │
                               │  │ Verification Code        │   │
                               │  │ (reward = 0 or 1)        │   │
                               │  └──────────────────────────┘   │
                               └─────────────────────────────────┘
```

### Training Loop (per rollout)

```
1. Sample 8 tasks from train.jsonl
2. For each task, generate 4 trajectories (N_SAMPLES_PER_PROMPT=4):
   a. Model generates response (may include <think> block + <tool_call>)
   b. Tool call executed via MCP → observation returned
   c. Repeat up to 15 turns
   d. Run verification code: complete → reward=1, else → reward=0
3. GRPO update: within each 4-sample group, contrast successes vs failures
4. Sync updated weights to SGLang engines
5. Repeat for NUM_ROLLOUT iterations (default: 200)
```

## Hardware Requirements

The launch script **auto-detects GPU memory** and adjusts batch size, sequence length, and SGLang memory fraction accordingly. No manual tuning needed.

| GPU         | VRAM  | Profile  | Auto Config                                         |
|-------------|-------|----------|-----------------------------------------------------|
| H200        | 140 GB| `large`  | batch=8, max_resp=16384, mem_frac=0.55, servers=16  |
| H100 / A100 | 80 GB | `medium` | batch=4, max_resp=8192, mem_frac=0.45, servers=8    |
| A100-40GB   | 40 GB | `small`  | batch=2, max_resp=4096, mem_frac=0.35, servers=4    |

You can always override any setting via environment variables:

```bash
# Example: force large batch on H100
ROLLOUT_BATCH_SIZE=8 SGLANG_MEM_FRACTION_STATIC=0.50 bash run_awm_grpo.sh
```

> Qwen3-4B (~8 GB weights) fits on a single GPU. Using TP=1 (no tensor parallelism) maximizes data parallelism — each GPU runs its own independent SGLang inference engine.

### Memory Budget (per GPU)

| Component                          | H200 (140 GB) | H100 (80 GB) |
|------------------------------------|----------------|---------------|
| SGLang model weights (Qwen3-4B bf16)| ~8 GB         | ~8 GB         |
| SGLang KV cache                    | ~50 GB         | ~28 GB        |
| Megatron training (peak, during train phase) | ~50 GB | ~35 GB   |
| MCP server subprocesses (CPU-bound)| ~5 GB (16×)    | ~2.4 GB (8×)  |
| **Total peak**                     | **~96 GB**     | **~65 GB**    |

## Quick Start (One-Click)

The automated setup script handles everything: dependency check, data download, model download, checkpoint conversion, and training data preparation.

```bash
# 1. Start from slime Docker (recommended)
docker pull slimerl/slime:latest
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /your/workspace:/workspace \
  -it slimerl/slime:latest /bin/bash

# 2. Clone the repo
cd /workspace
git clone https://github.com/<your-org>/Stateful-RL.git
cd Stateful-RL

# 3. Run one-click setup (~15 minutes)
cd awm_grpo
bash setup.sh

# 4. Run training
source .env
NUM_ROLLOUT=3 bash run_awm_grpo.sh     # Smoke test (~20 min)
bash run_awm_grpo.sh                    # Full training (~24h on 4×H200)
```

### setup.sh Options

| Flag                    | Description                                |
|-------------------------|--------------------------------------------|
| `--skip-model-download` | Skip Qwen3-4B download (already have it)  |
| `--skip-data-download`  | Skip AWM-1K dataset download               |
| `--skip-deps`           | Skip dependency installation                |
| `--skip-convert`        | Skip HF→Megatron conversion                |
| `--model-path /path`    | Use existing HF model at this path         |
| `--model-dist-path /path` | Use existing Megatron checkpoint          |
| `--num-gpus N`          | Override GPU count (default: auto-detect)  |

## Step-by-Step Setup (Manual)

If you prefer to understand each step, or if `setup.sh` fails at some point, follow these manual steps.

### Step 0: Docker Environment

```bash
docker pull slimerl/slime:latest
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /your/workspace:/workspace \
  -it slimerl/slime:latest /bin/bash
```

The Docker image includes: PyTorch 2.9+, CUDA 12.8, flash-attn, SGLang, Ray, Megatron-Core, mbridge, transformer-engine. If not using Docker, install these manually (see [requirements.txt](requirements.txt) for versions).

### Step 1: Install Project Packages

```bash
cd /workspace/Stateful-RL

# Install slime (RL training framework)
cd slime && pip install -e . --no-deps && cd ..

# Install agent-world-model (AWM environment)
cd agent-world-model && pip install -e . && cd ..

# Install any remaining dependencies
pip install -r awm_grpo/requirements.txt
```

### Step 2: Download AWM-1K Data

The AWM-1K dataset contains 1,000 synthetic environments with SQL databases, MCP server code, user tasks, and verification code.

```bash
# Download from HuggingFace (~500 MB)
huggingface-cli download Snowflake/AgentWorldModel-1K \
  --repo-type dataset \
  --local-dir agent-world-model/outputs

# Or via Python:
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Snowflake/AgentWorldModel-1K', repo_type='dataset', local_dir='agent-world-model/outputs')
"
```

Verify the download:

```bash
ls agent-world-model/outputs/
# Must contain:
#   gen_db.jsonl                  - 1,000 database schemas
#   gen_sample.jsonl              - 1,000 sample data sets
#   gen_tasks.jsonl               - 1,000 × 10 user tasks
#   gen_envs.jsonl                - 1,000 MCP server implementations
#   gen_verifier.pure_code.jsonl  - 10,000 verification functions
#   databases/                    - 1,000+ SQLite .db files
```

If the `databases/` directory is missing or incomplete, generate it:

```bash
cd agent-world-model
PYTHONPATH=. python3 -c "
from awm.tools import tools_jsonl_load, normalize_scenario_name
from awm.core.db import create_sqlite_database
from awm.core.sample import execute_sample_data
import os

os.makedirs('outputs/databases', exist_ok=True)
schemas = {normalize_scenario_name(x['scenario']): x for x in tools_jsonl_load('outputs/gen_db.jsonl')}
samples = {normalize_scenario_name(x['scenario']): x for x in tools_jsonl_load('outputs/gen_sample.jsonl')}

for i, (scenario, schema) in enumerate(schemas.items()):
    db_path = f'outputs/databases/{scenario}.db'
    if os.path.exists(db_path):
        continue
    db_path, _, _, _ = create_sqlite_database(scenario, schema['db_schema'], 'outputs/databases')
    if scenario in samples:
        execute_sample_data(db_path, samples[scenario]['sample_data'], scenario)
    if (i + 1) % 100 == 0:
        print(f'Created {i + 1}/{len(schemas)} databases')
print('Done')
"
cd ..
```

### Step 3: Download the Base Model

```bash
# Qwen3-4B (~8 GB, downloads in ~1 minute on fast connection)
huggingface-cli download Qwen/Qwen3-4B --local-dir models/Qwen3-4B

# Or via Python:
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-4B', local_dir='models/Qwen3-4B')
"
```

### Step 4: Convert Model to Megatron Format

Slime uses Megatron-LM for training, which requires a `torch_dist` checkpoint format.

```bash
cd slime

# Load Qwen3-4B architecture config
source scripts/models/qwen3-4B.sh

# Run conversion (single GPU, ~2 minutes)
PYTHONPATH=../Megatron-LM:. \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
python3 tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --no-gradient-accumulation-fusion \
    --hf-checkpoint ../models/Qwen3-4B \
    --save ../models/Qwen3-4B_torch_dist

cd ..
```

> **Common error**: `gradient_accumulation_fusion` — add `--no-gradient-accumulation-fusion` (required when APEX fused kernels are not available, which is the default in most Docker images).

Verify the conversion:

```bash
cat models/Qwen3-4B_torch_dist/latest_checkpointed_iteration.txt
# Should print: 0
ls models/Qwen3-4B_torch_dist/iter_0000000/mp_rank_00/
# Should contain: model_optim_rng.pt (or similar)
```

### Step 5: Prepare Training Data

```bash
cd awm_grpo

PYTHONPATH=../agent-world-model:. python3 data_prep.py \
  --tasks_path ../agent-world-model/outputs/gen_tasks.jsonl \
  --envs_path ../agent-world-model/outputs/gen_envs.jsonl \
  --verifier_path ../agent-world-model/outputs/gen_verifier.pure_code.jsonl \
  --output_dir ./data
```

This creates scenario-level splits (no train/test leakage):

| File            | Tasks  | Scenarios | Purpose          |
|-----------------|--------|-----------|------------------|
| `data/train.jsonl` | ~7,600 | ~400     | Training         |
| `data/val.jsonl`   | ~1,200 | ~60      | Validation       |
| `data/test.jsonl`  | ~1,200 | ~66      | Held-out testing |

### Step 6: Set Paths and Run

```bash
# Set model paths
export HF_CKPT=../models/Qwen3-4B
export REF_CKPT=../models/Qwen3-4B_torch_dist
export SAVE_DIR=./checkpoints/awm_grpo

# Quick smoke test (3 rollouts, ~20 minutes)
NUM_ROLLOUT=3 bash run_awm_grpo.sh

# Full training (200 rollouts, ~24 hours on 4× H200)
bash run_awm_grpo.sh
```

Monitor via Ray dashboard: http://localhost:8265

## What Happens When You Run Training

```
bash run_awm_grpo.sh
│
├── Kill stale sglang/ray processes
├── Prewarm template databases (fast DB reset during rollout)
├── Start Ray cluster (head node, all GPUs)
│
└── ray job submit → slime/train.py
    │
    ├── Launch 4 SGLang engines (1 per GPU, TP=1)
    ├── Launch 4 Megatron training actors (1 per GPU)
    │
    └── for rollout_id in range(200):
        │
        ├── [Rollout Phase] ~5-10 min
        │   ├── Sample 8 prompts from train.jsonl
        │   ├── Generate 4 trajectories per prompt (32 total)
        │   │   └── Each trajectory: up to 15 turns of:
        │   │       LLM generate → parse tool_call → MCP execute → observation
        │   ├── Run verification code → reward (0 or 1)
        │   └── Dynamic sampling: filter groups where all 4 have same reward
        │
        ├── [Training Phase] ~15-60s
        │   ├── Compute GRPO advantages (group-relative)
        │   ├── Policy gradient update (loss_mask: only model tokens)
        │   └── KL penalty against reference model
        │
        ├── [Weight Sync] ~0.6s
        │   └── Push updated weights to all 4 SGLang engines
        │
        └── [Checkpoint] every 10 rollouts
            └── Save model + optimizer state
```

## File Structure

| File | Description |
|------|-------------|
| **`setup.sh`** | One-click setup: deps, data download, model download, conversion, data prep. |
| **`run_awm_grpo.sh`** | Main launch script. Configures all hyperparameters, starts Ray, submits job. |
| `awm_config.yaml` | AWM-specific config: max_turns, server pool size, data paths. |
| `env_awm.py` | AWM environment: MCP server pool, tool execution, DB reset, verification. |
| `rollout.py` | Custom multi-turn rollout for slime: drives the LLM ↔ environment loop. |
| `reward.py` | Reward wrapper: returns pre-computed verification reward to slime. |
| `data_prep.py` | Converts AWM tasks to slime JSONL format (scenario-level splits). |
| `prewarm_templates.py` | Pre-builds template databases for fast reset during rollout. |
| `rollout_logging.py` | Per-sample trajectory logging for debugging and analysis. |
| `requirements.txt` | Python dependencies with install notes. |

## Key Configuration

### `run_awm_grpo.sh` — Training Hyperparameters

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_GPUS` | 4 | Total GPUs (auto-detected) |
| `NUM_ROLLOUT` | 200 | Total rollout iterations |
| `TP_SIZE` | 1 | Tensor parallel size. Keep 1 for 4B model. |
| `ROLLOUT_BATCH_SIZE` | 8/4/2 (auto) | Prompts sampled per rollout |
| `N_SAMPLES_PER_PROMPT` | 4 | Trajectories per prompt (GRPO group size) |
| `GLOBAL_BATCH_SIZE` | auto | Must equal ROLLOUT_BATCH_SIZE × N_SAMPLES_PER_PROMPT |
| `ROLLOUT_MAX_RESPONSE_LEN` | 16384/8192/4096 (auto) | Max tokens per trajectory. Qwen3 `<think>` needs ≥8192. |
| `SGLANG_MEM_FRACTION_STATIC` | 0.55/0.45/0.35 (auto) | GPU memory fraction for SGLang. Lower = more KV cache for long sequences. |
| `SGLANG_SERVER_CONCURRENCY` | 8/4/2 (auto) | Concurrent requests per SGLang engine |
| `SAVE_INTERVAL` | 10 | Save checkpoint every N rollouts |
| `EVAL_INTERVAL` | 10 | Run validation every N rollouts |

### `awm_config.yaml` — Environment Config

| Key | Default | Description |
|-----|---------|-------------|
| `max_turns` | 15 | Max tool-calling turns per episode |
| `server_pool_max_servers` | 16/8/4 (auto) | Max concurrent MCP server subprocesses. **This is the real concurrency bottleneck.** Lower if OOM, raise if GPUs idle. |
| `server_startup_timeout` | 30.0 | Seconds to wait for MCP server readiness |
| `server_startup_max_retries` | 2 | Retry count for failed server starts |
| `tool_timeout` | 30.0 | Timeout per tool call |

### Tuning Concurrency

The key balance is between `server_pool_max_servers` (AWM concurrency) and available CPU/memory:

```
server_pool_max_servers=16  →  16 MCP servers running concurrently
                            →  ~16 × 0.3 GB = ~5 GB CPU memory
                            →  Good for 4× H200 (140 GB each)

server_pool_max_servers=8   →  Safer for 2× H100 (80 GB each)
server_pool_max_servers=24  →  Faster rollout if you have spare memory
```

## Rollout Flow (Per Sample)

```
1. Reset
   ├── Acquire MCP server slot from pool (or wait if all busy)
   ├── Copy template .db → episode-specific .db
   └── Save initial DB snapshot (for verification later)

2. Multi-turn Loop (up to max_turns=15)
   ├── SGLang generates response tokens
   │   ├── May include <think>...</think> reasoning (Qwen3)
   │   ├── Ends with <tool_call>{"name": "...", "arguments": {...}}</tool_call>
   │   └── loss_mask=1, log_probs recorded
   │
   ├── Parse tool call from response
   │   ├── list_tools → return cached tool listing
   │   └── call_tool → execute via MCP HTTP call
   │
   ├── Tokenize tool response as observation
   │   └── loss_mask=0, log_prob=0 (no gradient through tool output)
   │
   └── Check terminal conditions
       ├── No tool call in response → episode ends
       └── Max turns reached → episode ends

3. Reward
   ├── Load verification code for this task
   ├── Run SQL queries comparing initial_db vs final_db
   └── Result: "complete" → reward=1.0, else → reward=0.0

4. Cleanup
   ├── Close MCP connection
   ├── Release server slot back to pool (server stays alive for reuse)
   └── Delete episode-specific .db
```

### Loss Masking

Only model-generated tokens receive gradient. Tool responses are masked out:

```
Tokens:    [system_prompt] [user_task] [<think>reasoning</think><tool_call>...] [observation] [<think>...</think><tool_call>...] [observation] ...
loss_mask: [0 0 0 ... 0]  [0 0 ... 0] [1 1 1 1 1 1 ... 1 1 1 1 1 1 1 1 1 1]  [0 0 ... 0]  [1 1 1 1 1 ... 1 1 1 1 1 1 1 1]  [0 0 ... 0]
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                        Model generates (receives gradient)                   Model generates (receives gradient)
```

## Performance Tuning

### Symptoms and Fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ActorUnavailableError` / OOM crash | Too many MCP servers eating memory | Lower `server_pool_max_servers` (16→8) |
| 99% samples `truncated` | Token budget too small for Qwen3 `<think>` | Increase `ROLLOUT_MAX_RESPONSE_LEN` to 16384+ |
| SGLang shows `#running-req: 1` constantly | AWM concurrency too low | Increase `server_pool_max_servers` and `SGLANG_SERVER_CONCURRENCY` |
| Slow rollout (~30 min per batch) | MCP server startup dominates | Set `PREWARM_TEMPLATES=1` (default), increase `server_pool_max_servers` |
| 0% reward for many rollouts | Model never calls tools correctly | Check system prompt; verify `ROLLOUT_MAX_RESPONSE_LEN ≥ 16384` |
| KV cache OOM in SGLang | Too many long sequences | Lower `SGLANG_MEM_FRACTION_STATIC` (0.55→0.45) |
| Training phase very slow | Batch too large for GPU memory | Lower `MAX_TOKENS_PER_GPU` |

### Expected Metrics (Qwen3-4B, 4× H200)

| Metric | First Rollout | After 50 Rollouts | After 200 Rollouts |
|--------|--------------|-------------------|-------------------|
| Rollout time | ~8 min | ~8 min | ~8 min |
| Training time | ~60s | ~30s | ~15s |
| Raw reward | 0.30-0.35 | 0.45-0.55 | 0.55-0.70 |
| Truncation rate | <5% | <3% | <3% |
| Tool calls/sample | ~4 | ~5 | ~6 |

## Logs and Debugging

### Log Locations

```
awm_grpo/
├── logs/
│   ├── awm_grpo_v2_YYYYMMDD_HHMMSS.log   # Full training log (stdout/stderr)
│   ├── rollouts/                           # Per-rollout summaries
│   │   ├── rollout_000000.jsonl            # All samples from rollout 0
│   │   └── rollout_000000.summary.json     # Aggregate stats
│   └── rollouts_live/                      # Real-time per-sample logging
│       └── live_samples_pid<PID>.jsonl     # Each line = one completed sample
└── checkpoints/
    └── awm_grpo/
        └── iter_XXXXXXX/                   # Megatron checkpoint
```

### Analyze Rollout Quality

```bash
# Quick stats from live log
python3 -c "
import json, statistics
samples = [json.loads(l) for l in open('logs/rollouts_live/live_samples_pid<PID>.jsonl')]
rewards = [s['reward'] for s in samples]
print(f'Total samples: {len(samples)}')
print(f'Reward=1: {sum(1 for r in rewards if r > 0)} ({sum(1 for r in rewards if r > 0)/len(rewards)*100:.1f}%)')
print(f'Truncated: {sum(1 for s in samples if s[\"status\"]==\"truncated\")} ({sum(1 for s in samples if s[\"status\"]==\"truncated\")/len(samples)*100:.1f}%)')
print(f'Avg tool calls: {statistics.mean(s[\"metadata\"].get(\"tool_call_count\", 0) for s in samples):.1f}')
print(f'Avg iterations: {statistics.mean(s.get(\"num_iterations\", 0) for s in samples):.1f}')
"
```

### Inspect a Single Trajectory

```bash
# View the trajectory of the first successful sample
python3 -c "
import json
samples = [json.loads(l) for l in open('logs/rollouts_live/live_samples_pid<PID>.jsonl')]
success = [s for s in samples if s['reward'] == 1.0]
if success:
    s = success[0]
    print(f'Scenario: {s[\"scenario\"]}')
    print(f'Task: {s.get(\"task\", s[\"metadata\"].get(\"task\", \"?\"))}')
    print(f'Tool calls: {s[\"metadata\"].get(\"tool_call_count\", 0)}')
    print(f'Reward: {s[\"reward\"]}')
    print()
    for msg in s.get('trajectory', []):
        role = msg['role']
        content = msg['content'][:200]
        print(f'[{role}] {content}')
        print()
"
```

## Troubleshooting

### "MCP server for X failed to start on port Y"

The MCP server code for that scenario has a bug or missing dependency.

```bash
# Check the server error log
ls agent-world-model/outputs/server_logs/
cat agent-world-model/outputs/server_logs/<scenario>_<port>.log
```

The training will retry with `server_startup_max_retries=2`. If a scenario consistently fails, it won't block other scenarios.

### "Permission denied" for server logs or databases

```bash
chmod -R u+w agent-world-model/outputs/
```

### Training hangs at "Rollout generation: 0%"

This is normal for the first batch. MCP servers are starting up (each takes 10-30s). With `server_pool_max_servers=16`, expect 2-3 minutes for the first rollout. Subsequent rollouts reuse warm servers and are much faster.

### "CUDA out of memory" during training phase

The training phase needs to hold both Megatron model and the rollout data in GPU memory.

```bash
# Option 1: Reduce sequence length budget
ROLLOUT_MAX_RESPONSE_LEN=12288 bash run_awm_grpo.sh

# Option 2: Reduce batch size
ROLLOUT_BATCH_SIZE=4 N_SAMPLES_PER_PROMPT=4 GLOBAL_BATCH_SIZE=16 bash run_awm_grpo.sh

# Option 3: Lower SGLang memory to leave more for training
SGLANG_MEM_FRACTION_STATIC=0.45 bash run_awm_grpo.sh
```

### "apply_rope_fusion is not available. Please install TE >= 1.4"

transformer-engine (TE) is needed by Megatron. The PyPI prebuilt wheels often have ABI mismatches with PyTorch, so `setup.sh` builds it from source. This requires `nvcc`:

```bash
# Check if nvcc is available
nvcc --version

# If not found, install CUDA toolkit:
# Ubuntu/Debian:
sudo apt install nvidia-cuda-toolkit
# Or via conda:
conda install cuda-toolkit -c nvidia
# Or set PATH if CUDA is installed but not in PATH:
export PATH=/usr/local/cuda/bin:$PATH

# Then re-run setup.sh (it will detect nvcc and build TE)
bash setup.sh --skip-deps    # skip other deps, just re-check
# Or install TE manually:
pip install "transformer-engine[pytorch]" --no-build-isolation
```

### Checkpoint conversion fails

```bash
# Error: "gradient_accumulation_fusion"
# Fix: already included in our script, but if running manually:
python3 tools/convert_hf_to_torch_dist.py ... --no-gradient-accumulation-fusion

# Error: "World size must be less than or equal to number of layers"
# Fix: use fewer GPUs for conversion (single GPU is fine)
CUDA_VISIBLE_DEVICES=0 python3 tools/convert_hf_to_torch_dist.py ...
```

### Converting checkpoint back to HuggingFace format

After training, convert the Megatron checkpoint back to HF format for inference:

```bash
cd slime
PYTHONPATH=../Megatron-LM:. python3 tools/convert_torch_dist_to_hf.py \
  --input-dir ../awm_grpo/checkpoints/awm_grpo/iter_XXXXXXX/ \
  --output-dir ../models/Qwen3-4B-AWM-GRPO \
  --origin-hf-dir ../models/Qwen3-4B
```

## Tests

```bash
cd awm_grpo

# Test environment lifecycle (DB + MCP + tool calls + verification)
PYTHONPATH=../agent-world-model:. python3 test_env.py

# Test rollout logic with scripted agent (no GPU needed)
PYTHONPATH=../agent-world-model:. python3 test_rollout_mock.py

# Test server reuse optimization
PYTHONPATH=../agent-world-model:. python3 test_server_reuse.py
```

## Verified Environment

This setup has been tested and confirmed working on:

| Component | Version |
|-----------|---------|
| OS | Ubuntu 24.04 |
| Python | 3.12.3 |
| PyTorch | 2.9.1+cu128 |
| CUDA Toolkit | 12.8 |
| NVIDIA Driver | 570.195.03 |
| GPU | 4× NVIDIA H200 (140 GB) |
| flash-attn | 2.8.3 |
| SGLang | 0.5.9 |
| Ray | 2.54.0 |
| transformers | 4.57.1 |
| megatron-core | 0.16.0rc0 |
| mbridge | 0.15.1 |
| slime | 0.2.3 |

## Citation

```bibtex
@article{wang2026agentworldmodelinfinity,
      title={Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning},
      author={Zhaoyang Wang and Canwen Xu and Boyi Liu and Yite Wang and Siwei Han and Zhewei Yao and Huaxiu Yao and Yuxiong He},
      year={2026},
      eprint={2602.10090},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.10090},
}
```
