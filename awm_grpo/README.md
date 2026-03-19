# AWM Vanilla GRPO

Vanilla GRPO training framework for tool agents on AWM (Agent World Model) environments, built on [slime](../slime/).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  slime (Ray + Megatron + SGLang)                        │
│                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │ Megatron │    │   SGLang     │    │  Rollout      │  │
│  │ Training │◄──►│   Engine     │◄──►│  (rollout.py) │  │
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
                              │  │ (reward computation)     │   │
                              │  └──────────────────────────┘   │
                              └─────────────────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `env_awm.py` | AWM environment wrapper: MCP server lifecycle, tool execution, DB management, verification |
| `rollout.py` | Custom multi-turn rollout function for slime: LLM ↔ env loop via SGLang |
| `reward.py` | Custom reward function: returns pre-computed verification reward |
| `data_prep.py` | Converts AWM tasks (gen_tasks.jsonl) to slime training JSONL |
| `awm_config.yaml` | Custom config: max_turns, AWM data paths |
| `run_awm_grpo.sh` | Launch script for 2×H100 training |

## Setup

### 1. Prerequisites

- AWM outputs generated (`agent-world-model/outputs/`)
- slime installed with Megatron-LM backend
- Qwen3-4B checkpoint converted to slime format

### 2. Prepare training data

```bash
cd /data/luz17/FailureCall/awm_grpo

python data_prep.py \
  --tasks_path ../agent-world-model/outputs/gen_tasks.jsonl \
  --envs_path ../agent-world-model/outputs/gen_envs.jsonl \
  --verifier_path ../agent-world-model/outputs/gen_verifier.pure_code.jsonl \
  --output_dir ./data
```

This creates `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl` with scenario-level splits.

### 3. Configure paths

Edit `run_awm_grpo.sh` and set:
- `HF_CKPT` - HuggingFace checkpoint path
- `SLIME_CKPT` - Converted slime/Megatron checkpoint
- `REF_CKPT` - Reference model checkpoint (for KL)
- `MEGATRON_DIR` - Megatron-LM installation

### 4. Run training

```bash
bash run_awm_grpo.sh
```

## Rollout Flow (per sample)

1. **Init**: Reset SQLite DB to initial state, start MCP server on unique port
2. **Loop** (up to `max_turns`):
   - SGLang generates LLM response tokens (loss_mask=1, log_probs recorded)
   - Parse `<tool_call>` from response
   - Execute via MCP: `list_tools` or `call_tool`
   - Tokenize tool response (loss_mask=0, log_prob=0)
   - Append to context
3. **Reward**: Run AWM verification code against initial vs final DB
   - Complete → 1.0
   - Incomplete → 0.0
4. **Cleanup**: Stop MCP server, release port

## Reward Design (Vanilla GRPO Baseline)

This is the **B1 baseline** from the research plan:

- **Outcome reward**: `R_outcome = 1.0` (Completed) or `0.0` (else)
- **Format penalty**: Invalid tool call format → early termination (handled by env)
- **No step-level reward**: Pure trajectory-level GRPO

## Key Design Decisions

- **One MCP server per sample**: Each rollout sample gets its own server on a unique port from a thread-safe pool. Simple but ensures isolation.
- **Reward computed in rollout**: AWM verification runs inside `env.compute_reward()` at the end of each episode, so the reward is available before returning to slime's training loop.
- **Loss mask**: Model-generated tokens have `loss_mask=1`, tool responses have `loss_mask=0`. Only model tokens receive gradient.
- **Chat template**: AWM's system prompt is baked into the training data. `--apply-chat-template` formats it via the tokenizer's chat template.

## Extending to State-Based Progress RL

To add state-based progress reward (the paper's main method):

1. Add predicate extraction to `env_awm.py` (load from pre-computed predicates)
2. Compute `Phi(s_t)` after each state-mutating tool call
3. Compute `R_prog(t) = Phi(s_{t+1}) - Phi(s_t)` per step
4. Modify reward to use turn-level return-to-go instead of trajectory-level
5. Implement turn-level group-relative advantage in the training loss
