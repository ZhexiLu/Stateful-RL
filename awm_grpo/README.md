# AWM-GRPO: Multi-Turn Tool-Use RL Training on AWM Environments

Vanilla GRPO training for tool-calling agents on [AgentWorldModel-1K](https://huggingface.co/datasets/Snowflake/AgentWorldModel-1K) environments, powered by [slime](https://github.com/THUDM/slime) (Megatron + SGLang + Ray).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  slime (Ray + Megatron + SGLang)                        │
│                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │ Megatron │    │   SGLang     │    │  Rollout      │  │
│  │ Training │◄──►│   Engines    │◄──►│  (rollout.py) │  │
│  │ (GRPO)   │    │  /generate   │    │               │  │
│  └──────────┘    └──────────────┘    └───────┬───────┘  │
│                                              │          │
└──────────────────────────────────────────────┼──────────┘
                                               │
                              ┌────────────────▼────────────────┐
                              │  AWM Environment (env_awm.py)   │
                              │                                 │
                              │  ┌───────┐  ┌──────────────┐   │
                              │  │SQLite │  │ MCP Server   │   │
                              │  │  DB   │◄─│ (FastAPI)    │   │
                              │  └───────┘  └──────────────┘   │
                              │                                 │
                              │  ┌──────────────────────────┐   │
                              │  │ Verification Code        │   │
                              │  │ (reward = 0 or 1)        │   │
                              │  └──────────────────────────┘   │
                              └─────────────────────────────────┘
```

## Hardware Requirements

| Config         | GPUs          | Notes                                     |
|----------------|---------------|-------------------------------------------|
| Recommended    | 4× H200/H100  | TP=1, 4 SGLang engines, ~85 GB/GPU        |
| Minimum        | 2× H100       | TP=1, 2 SGLang engines, slower rollout     |

> Qwen3-4B (~8 GB weights) fits on a single GPU. Using TP=1 maximizes data parallelism and inference throughput.

## Quick Start

The full setup takes **~15 minutes** on a fresh machine (assuming Docker with GPUs).

### Step 0: Docker (Recommended)

Use the slime Docker image which has Megatron, SGLang, Ray, and PyTorch pre-installed:

```bash
docker pull slimerl/slime:latest
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /your/workspace:/workspace \
  -it slimerl/slime:latest /bin/bash
```

### Step 1: Clone the Repo

```bash
cd /workspace
git clone https://github.com/<your-org>/Stateful-RL.git
cd Stateful-RL
```

### Step 2: Install Dependencies

```bash
# Install slime (if not already in Docker)
cd slime && pip install -e . --no-deps && cd ..

# Install AWM environment
cd agent-world-model && pip install -e . && cd ..

# Install awm_grpo dependencies
pip install -r awm_grpo/requirements.txt
```

### Step 3: Download AWM-1K Data

Download the 1,000-environment dataset from HuggingFace:

```bash
# Method 1: huggingface-cli (recommended)
huggingface-cli download Snowflake/AgentWorldModel-1K \
  --repo-type dataset \
  --local-dir agent-world-model/outputs

# Method 2: Python
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Snowflake/AgentWorldModel-1K',
    repo_type='dataset',
    local_dir='agent-world-model/outputs',
)
"
```

After download, verify the data:

```bash
ls agent-world-model/outputs/
# Expected files:
#   gen_db.jsonl              (1000 lines - DB schemas)
#   gen_sample.jsonl          (1000 lines - sample data)
#   gen_tasks.jsonl           (1000 lines - user tasks, 10 per scenario)
#   gen_envs.jsonl            (1000 lines - MCP server code)
#   gen_verifier.pure_code.jsonl (10010 lines - verification code)
#   databases/                (1000+ .db files)
```

If the `databases/` directory is missing, generate it:

```bash
cd agent-world-model
python -c "
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

### Step 4: Download and Convert Model

```bash
# Download Qwen3-4B
huggingface-cli download Qwen/Qwen3-4B --local-dir /workspace/models/Qwen3-4B

# Convert to Megatron torch_dist format (single GPU, ~2 minutes)
cd slime
source scripts/models/qwen3-4B.sh

PYTHONPATH=$(pwd)/../Megatron-LM:$(pwd) \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
python tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --no-gradient-accumulation-fusion \
    --hf-checkpoint /workspace/models/Qwen3-4B \
    --save /workspace/models/Qwen3-4B_torch_dist

cd ..
```

> **Note**: The `--no-gradient-accumulation-fusion` flag is required if APEX is not installed (which is the case in most Docker images).

### Step 5: Prepare Training Data

```bash
cd awm_grpo

PYTHONPATH=../agent-world-model:. python data_prep.py \
  --tasks_path ../agent-world-model/outputs/gen_tasks.jsonl \
  --envs_path ../agent-world-model/outputs/gen_envs.jsonl \
  --verifier_path ../agent-world-model/outputs/gen_verifier.pure_code.jsonl \
  --output_dir ./data
```

This creates:
- `data/train.jsonl` — ~7,600 tasks from ~400 scenarios
- `data/val.jsonl` — ~1,200 tasks from ~60 scenarios
- `data/test.jsonl` — ~1,200 tasks from ~66 scenarios

### Step 6: Configure Paths

Edit `run_awm_grpo.sh` and update these paths (or export them as env vars):

```bash
export HF_CKPT=/workspace/models/Qwen3-4B
export REF_CKPT=/workspace/models/Qwen3-4B_torch_dist
# SAVE_DIR defaults to ./checkpoints/awm_grpo
```

### Step 7: Run Training

```bash
# Full training (200 rollouts, ~24 hours on 4× H200)
bash run_awm_grpo.sh

# Quick test (3 rollouts, ~20 minutes)
NUM_ROLLOUT=3 bash run_awm_grpo.sh
```

Monitor via Ray dashboard: http://localhost:8265

## File Structure

| File | Description |
|------|-------------|
| `run_awm_grpo.sh` | Main launch script. Configures all hyperparameters, starts Ray, submits training job. |
| `awm_config.yaml` | AWM-specific config: max_turns, server pool size, data paths. |
| `env_awm.py` | AWM environment wrapper: MCP server pool, tool execution, DB reset, verification. |
| `rollout.py` | Custom multi-turn rollout for slime: drives the LLM ↔ env loop via SGLang. |
| `reward.py` | Custom reward function: returns pre-computed verification reward to slime. |
| `data_prep.py` | Converts AWM tasks to slime training JSONL format (scenario-level splits). |
| `prewarm_templates.py` | Pre-builds template databases for fast reset during rollout. |
| `rollout_logging.py` | Per-sample and per-rollout logging for trajectory analysis. |

## Key Configuration

### `run_awm_grpo.sh`

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_GPUS` | 4 | Total GPUs |
| `TP_SIZE` | 1 | Tensor parallel size (1 = max data parallelism) |
| `ROLLOUT_BATCH_SIZE` | 8 | Number of prompt groups per rollout |
| `N_SAMPLES_PER_PROMPT` | 4 | Samples per prompt (for GRPO advantage) |
| `ROLLOUT_MAX_RESPONSE_LEN` | 16384 | Max response tokens (must be large for Qwen3 `<think>` mode) |
| `GLOBAL_BATCH_SIZE` | 32 | Training batch size |
| `SGLANG_SERVER_CONCURRENCY` | 8 | SGLang concurrent requests per engine |
| `SGLANG_MEM_FRACTION_STATIC` | 0.55 | GPU memory fraction for SGLang weights |

### `awm_config.yaml`

| Key | Default | Description |
|-----|---------|-------------|
| `max_turns` | 15 | Max tool-calling turns per episode |
| `server_pool_max_servers` | 16 | Max concurrent MCP server subprocesses |
| `server_startup_timeout` | 30.0 | Seconds to wait for MCP server readiness |
| `server_startup_max_retries` | 2 | Retry count for failed server starts |

## Rollout Flow (Per Sample)

1. **Reset**: Acquire MCP server slot from pool, reset SQLite DB from template
2. **Multi-turn loop** (up to `max_turns`):
   - SGLang generates LLM response tokens (`loss_mask=1`, log_probs recorded)
   - Parse `<tool_call>` from response
   - Execute via MCP: `list_tools` or `call_tool`
   - Tokenize tool response (`loss_mask=0`, `log_prob=0`)
   - Append to context for next turn
3. **Reward**: Run AWM verification code comparing initial vs final DB state
   - `"complete"` → reward = 1.0
   - Otherwise → reward = 0.0
4. **Cleanup**: Close MCP connection, release server slot back to pool

## Performance Tuning

### Symptoms and Fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ActorUnavailableError` crash | Too many MCP servers → OOM | Lower `server_pool_max_servers` |
| 99% samples `truncated` | Token budget too small for `<think>` | Increase `ROLLOUT_MAX_RESPONSE_LEN` to 16384+ |
| SGLang `#running-req: 1` | AWM concurrency too low | Increase `server_pool_max_servers` and `SGLANG_SERVER_CONCURRENCY` |
| Slow rollout (~30min per batch) | MCP server startup dominates | Enable `PREWARM_TEMPLATES=1`, increase `server_pool_max_servers` |
| 0% reward | Model never calls tools | Check system prompt, increase `ROLLOUT_MAX_RESPONSE_LEN` |

### Memory Budget (per GPU, H200 140 GB)

| Component | Memory |
|-----------|--------|
| SGLang weights (Qwen3-4B bf16) | ~8 GB |
| SGLang KV cache | ~30-50 GB (depends on `mem_fraction_static`) |
| Megatron training (offloaded during rollout) | ~26 GB |
| MCP server subprocesses (CPU-bound) | ~0.3 GB each |
| **Total** | **~85 GB** |

## Tests

```bash
cd awm_grpo

# Test environment lifecycle (DB + MCP + tool calls + verification)
PYTHONPATH=../agent-world-model:. python test_env.py

# Test rollout logic with scripted agent (no GPU needed)
PYTHONPATH=../agent-world-model:. python test_rollout_mock.py

# Test server reuse optimization
PYTHONPATH=../agent-world-model:. python test_server_reuse.py
```

## Logs

Rollout logs are written to `logs/`:

```
logs/
├── rollouts/              # Per-rollout summaries (after training)
│   ├── rollout_000000.jsonl
│   └── rollout_000000.summary.json
└── rollouts_live/         # Per-sample live logging (during rollout)
    └── live_samples_pid<PID>.jsonl
```

Analyze rollout quality:

```bash
python -c "
import json, statistics
samples = [json.loads(l) for l in open('logs/rollouts_live/live_samples_pid<PID>.jsonl')]
print(f'Total: {len(samples)}')
print(f'Completed: {sum(1 for s in samples if s[\"status\"]==\"completed\")}')
print(f'Truncated: {sum(1 for s in samples if s[\"status\"]==\"truncated\")}')
rewards = [s['reward'] for s in samples]
print(f'Reward: mean={statistics.mean(rewards):.3f}, nonzero={sum(1 for r in rewards if r>0)}/{len(rewards)}')
print(f'Tool calls: mean={statistics.mean(s[\"metadata\"].get(\"tool_call_count\",0) for s in samples):.1f}')
print(f'Iterations: mean={statistics.mean(s.get(\"num_iterations\",0) for s in samples):.1f}')
"
```

## Troubleshooting

### "MCP server for X failed to start on port Y"

The server code has a syntax error or missing dependency. Check the server log:

```bash
cat agent-world-model/outputs/server_logs/<scenario>_<port>.log
```

If `server_logs/` has permission issues:

```bash
chmod -R a+w agent-world-model/outputs/server_logs/
```

### "Permission denied" for server logs

The AWM outputs directory may be read-only. Fix:

```bash
chmod -R u+w agent-world-model/outputs/
```

### Training hangs at "Rollout generation: 0%"

MCP servers are starting up (each takes 10-30s). With `server_pool_max_servers=16`, the first batch may take 2-3 minutes. This is normal. Subsequent batches reuse warm servers.

### Checkpoint conversion fails with "gradient_accumulation_fusion"

Add `--no-gradient-accumulation-fusion` to the conversion command. This is required when APEX is not installed.

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
