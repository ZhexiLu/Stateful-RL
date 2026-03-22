"""Parse config.yaml and emit shell variable assignments to stdout.

Usage (from run_awm_grpo.sh):
    eval "$(python3 parse_config.py config.yaml)"

All "auto" values are left unset so the shell script can fill them
from GPU auto-detection.
"""

import sys
import yaml

MAPPING = {
    # paths
    ("paths", "hf_checkpoint"):       "HF_CKPT",
    ("paths", "ref_checkpoint"):      "REF_CKPT",
    ("paths", "save_dir"):            "SAVE_DIR",
    ("paths", "resume_checkpoint"):   "RESUME_CKPT",
    # hardware
    ("hardware", "num_gpus"):         "NUM_GPUS",
    ("hardware", "tp_size"):          "TP_SIZE",
    # training
    ("training", "num_rollout"):             "NUM_ROLLOUT",
    ("training", "rollout_batch_size"):      "ROLLOUT_BATCH_SIZE",
    ("training", "n_samples_per_prompt"):    "N_SAMPLES_PER_PROMPT",
    ("training", "rollout_temperature"):     "ROLLOUT_TEMPERATURE",
    ("training", "save_interval"):           "SAVE_INTERVAL",
    ("training", "eval_interval"):           "EVAL_INTERVAL",
    # optimizer
    ("optimizer", "lr"):              "LR",
    ("optimizer", "weight_decay"):    "WEIGHT_DECAY",
    ("optimizer", "adam_beta1"):      "ADAM_BETA1",
    ("optimizer", "adam_beta2"):      "ADAM_BETA2",
    # grpo
    ("grpo", "kl_loss_coef"):         "KL_LOSS_COEF",
    ("grpo", "eps_clip"):             "EPS_CLIP",
    ("grpo", "eps_clip_high"):        "EPS_CLIP_HIGH",
    # inference
    ("inference", "max_response_length"):  "ROLLOUT_MAX_RESPONSE_LEN",
    ("inference", "gpu_memory_utilization"): "SGLANG_MEM_FRACTION_STATIC",
    ("inference", "server_concurrency"):   "SGLANG_SERVER_CONCURRENCY",
    # environment
    ("environment", "max_turns"):                "AWM_MAX_TURNS",
    ("environment", "server_pool_max_servers"):  "SERVER_POOL_MAX",
    ("environment", "server_startup_timeout"):   "AWM_SERVER_STARTUP_TIMEOUT",
    ("environment", "server_startup_max_retries"): "AWM_SERVER_STARTUP_MAX_RETRIES",
    ("environment", "tool_timeout"):             "AWM_TOOL_TIMEOUT",
    # logging
    ("logging", "wandb_enabled"):     "WANDB_ENABLED",
    ("logging", "wandb_project"):     "WANDB_PROJECT",
    ("logging", "wandb_group"):       "WANDB_GROUP",
    ("logging", "wandb_key"):         "WANDB_KEY",
}


def main():
    if len(sys.argv) < 2:
        print("# No config file specified", file=sys.stderr)
        sys.exit(0)

    path = sys.argv[1]
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    for (section, key), env_var in MAPPING.items():
        val = (cfg.get(section) or {}).get(key)
        if val is None:
            continue
        # Skip "auto" values — let the shell script handle them
        if isinstance(val, str) and val.strip().lower() == "auto":
            continue
        # Booleans
        if isinstance(val, bool):
            val = "1" if val else "0"
        print(f'export {env_var}="{val}"')


if __name__ == "__main__":
    main()
