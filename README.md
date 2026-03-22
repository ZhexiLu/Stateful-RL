# Stateful-RL

Reinforcement learning framework for training tool-using agents in **stateful environments** with implicit recovery learning.

Built on [AgentWorldModel-1K](https://huggingface.co/datasets/Snowflake/AgentWorldModel-1K) (1,000 synthetic SQL-backed tool-use environments) and [slime](https://github.com/THUDM/slime) (distributed RL training with Megatron + SGLang).

## Quick Start

**Requirements:** Linux with 2+ NVIDIA GPUs (tested on 4× H200/H100), CUDA 12.x, and `nvcc` available.

```bash
# Clone the repo
git clone --recursive git@github.com:ZhexiLu/Stateful-RL.git
cd Stateful-RL

# One-click setup: creates venv, installs all dependencies, downloads model & data
cd awm_grpo
bash setup.sh

# Activate the environment
source ../.venv_awm_grpo/bin/activate

# Quick smoke test (~20 minutes)
NUM_ROLLOUT=3 bash run_awm_grpo.sh

# Full training (~24 hours on 4× H200)
bash run_awm_grpo.sh
```

That's it. `setup.sh` handles everything automatically:
- Creates a Python 3.12 venv with all dependencies (PyTorch, SGLang, Megatron-LM, etc.)
- Downloads the Qwen3-4B model from HuggingFace
- Converts the model to Megatron torch_dist format
- Downloads the AWM-1K dataset and generates SQLite databases
- Prepares train/val/test splits

Run `bash setup.sh --help` for advanced options (custom model path, GPU count, etc.).

## Project Structure

```
Stateful-RL/
├── agent-world-model/     # AWM: synthetic environment generation framework
│   ├── awm/               #   Core: DB, MCP server, verifier, tool execution
│   └── outputs/           #   Generated data (downloaded by setup.sh)
├── slime/                 # slime: RL training framework (Megatron + SGLang + Ray)
├── Megatron-LM/           # Megatron-LM (training backend, cloned by setup.sh)
└── awm_grpo/              # AWM-GRPO: multi-turn GRPO training on AWM
    ├── setup.sh           #   One-click setup script
    ├── run_awm_grpo.sh    #   Main launch script
    ├── env_awm.py         #   AWM environment wrapper
    ├── rollout.py         #   Custom multi-turn rollout
    ├── reward.py          #   Reward function (verification-based)
    └── data_prep.py       #   Data preparation
```

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
