# Stateful-RL

Reinforcement learning framework for training tool-using agents in **stateful environments** with implicit recovery learning.

Built on [AgentWorldModel-1K](https://huggingface.co/datasets/Snowflake/AgentWorldModel-1K) (1,000 synthetic SQL-backed tool-use environments) and [slime](https://github.com/THUDM/slime) (distributed RL training with Megatron + SGLang).

## Project Structure

```
Stateful-RL/
├── agent-world-model/     # AWM: synthetic environment generation framework
│   ├── awm/               #   Core: DB, MCP server, verifier, tool execution
│   └── outputs/            #   Generated data (download from HuggingFace)
├── slime/                  # slime: RL training framework (Megatron + SGLang + Ray)
├── Megatron-LM/            # Megatron-LM (training backend)
└── awm_grpo/               # ★ AWM-GRPO: multi-turn GRPO training on AWM
    ├── run_awm_grpo.sh     #   Main launch script
    ├── env_awm.py          #   AWM environment wrapper
    ├── rollout.py          #   Custom multi-turn rollout
    ├── reward.py           #   Reward function (verification-based)
    ├── data_prep.py        #   Data preparation
    └── README.md           #   Detailed setup & usage guide
```

## Quick Start

See [awm_grpo/README.md](awm_grpo/README.md) for full setup instructions.

**TL;DR** (assuming Docker with 4× H200 GPUs):

```bash
# 1. Pull slime Docker image
docker pull slimerl/slime:latest
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -v $(pwd):/workspace -it slimerl/slime:latest /bin/bash

# 2. Install
cd /workspace/Stateful-RL
cd slime && pip install -e . --no-deps && cd ..
cd agent-world-model && pip install -e . && cd ..
pip install -r awm_grpo/requirements.txt

# 3. Download AWM data
huggingface-cli download Snowflake/AgentWorldModel-1K \
  --repo-type dataset --local-dir agent-world-model/outputs

# 4. Download & convert model
huggingface-cli download Qwen/Qwen3-4B --local-dir /workspace/models/Qwen3-4B
cd slime && source scripts/models/qwen3-4B.sh
PYTHONPATH=$(pwd)/../Megatron-LM:$(pwd) CUDA_DEVICE_MAX_CONNECTIONS=1 \
python tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
  --no-gradient-accumulation-fusion \
  --hf-checkpoint /workspace/models/Qwen3-4B \
  --save /workspace/models/Qwen3-4B_torch_dist && cd ..

# 5. Prepare data & train
cd awm_grpo
PYTHONPATH=../agent-world-model:. python data_prep.py
HF_CKPT=/workspace/models/Qwen3-4B REF_CKPT=/workspace/models/Qwen3-4B_torch_dist \
  NUM_ROLLOUT=3 bash run_awm_grpo.sh
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
