---
license: "cc-by-4.0"
language:
- en
tags:
- agent
- tool-use
- reinforcement-learning
- mcp
- synthetic
pretty_name: "agent-world-model"
viewer: false
---

<h1 align="center">AgentWorldModel-1K</h1>

<h3 align="center">Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning</h3>

<p align="center">
  <a href="https://github.com/Raibows">Zhaoyang Wang<sup>1</sup></a>,
  <a href="https://www.canwenxu.net/">Canwen Xu<sup>2</sup></a>,
  <a href="https://www.snowflake.com/en/blog/authors/boyi-liu/">Boyi Liu<sup>2</sup></a>,
  <a href="https://yitewang.github.io/">Yite Wang<sup>2</sup></a>,
  <a href="https://lillianwei-h.github.io/">Siwei Han<sup>1</sup></a>,<br/>
  <a href="https://yaozhewei.github.io/">Zhewei Yao<sup>2</sup></a>,
  <a href="https://www.huaxiuyao.io/">Huaxiu Yao<sup>1</sup></a>,
  <a href="https://www.snowflake.com/en/blog/authors/yuxiong-he/">Yuxiong He<sup>2</sup></a>
</p>
<p align="center">
  <sup>1</sup>UNC-Chapel Hill &nbsp; <sup>2</sup>Snowflake AI Research &nbsp;
</p>



# Overview

**AgentWorldModel-1K** contains 1,000 fully synthetic, executable, SQL database-backed tool-use environments exposed via a unified MCP (Model Context Protocol) interface, designed for large-scale multi-turn agentic reinforcement learning.

Each environment is synthesized through the **Agent World Model (AWM)** pipeline:

1. **Scenario** — A high-level description (e.g., "an online shopping platform")
2. **Tasks** — 10 user tasks per scenario that serve as functional requirements
3. **Database** — SQLite database schema and sample data as the state backend
4. **Interface** — Python interface layer (FastAPI + MCP) as the action/observation space
5. **Verification** — Verification code that inspects database state changes for reward signals

For the full synthesis pipeline, please visit [https://github.com/Snowflake-Labs/agent-world-model](https://github.com/Snowflake-Labs/agent-world-model).

# Resources
Related resources are also available, please check:

| Resource | Link |
|----------|------|
| 📄 Paper | [📄 arxiv.org/abs/2602.10090](https://arxiv.org/abs/2602.10090) |
| 💻 Code | [💻 Snowflake-Labs/agent-world-model](https://github.com/Snowflake-Labs/agent-world-model) |
| 📦 AgentWorldModel-1K | [🤗 Snowflake/AgentWorldModel-1K](https://huggingface.co/datasets/Snowflake/AgentWorldModel-1K) |
| 🤖 Arctic-AWM-4B | [🤗 Snowflake/Arctic-AWM-4B](https://huggingface.co/Snowflake/Arctic-AWM-4B) |
| 🤖 Arctic-AWM-8B | [🤗 Snowflake/Arctic-AWM-8B](https://huggingface.co/Snowflake/Arctic-AWM-8B) |
| 🤖 Arctic-AWM-14B | [🤗 Snowflake/Arctic-AWM-14B](https://huggingface.co/Snowflake/Arctic-AWM-14B) |


# Dataset Files

| File | #Entries | Description |
|------|----------|-------------|
| `gen_scenario.jsonl` | 1,000 | Synthesized scenario descriptions |
| `gen_tasks.jsonl` | 1,000 | 10 user tasks per scenario |
| `gen_db.jsonl` | 1,000 | Database schema definitions for each scenario |
| `gen_sample.jsonl` | 1,000 | Sample data to populate the initial database state |
| `gen_spec.jsonl` | 1,000 | API specifications for each scenario's interface |
| `gen_envs.jsonl` | 1,000 | MCP environment code (FastAPI + MCP server) |
| `gen_verifier.jsonl` | 10K | Verification code for code-augmented LLM-as-a-Judge |
| `gen_verifier.pure_code.jsonl` | 10K | Verification code for purely code-based Judge |

# Citation

If you find this resource useful, please kindly cite:

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
