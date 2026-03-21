#!/bin/bash
# GPU utilization monitor - logs every 5 seconds
LOG_DIR="/workspace/Stateful-RL/awm_grpo/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/gpu_util_$(date +%Y%m%d_%H%M%S).csv"

echo "timestamp,gpu0_util,gpu0_mem_used,gpu1_util,gpu1_mem_used,gpu2_util,gpu2_mem_used,gpu3_util,gpu3_mem_used" > "$LOG_FILE"

while true; do
    TS=$(date +%H:%M:%S)
    LINE=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr '\n' ',' | sed 's/,$//')
    echo "${TS},${LINE}" >> "$LOG_FILE"
    sleep 5
done
