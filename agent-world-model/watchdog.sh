#!/bin/bash
# Watchdog: monitor eval_batch.py running in screen "awm".
# If results.jsonl stops growing for STALL_TIMEOUT seconds, kill & resume.
# If all 100 scenarios are done, kill vLLM and exit.

set -euo pipefail

WORK_DIR="/data/luz17/FailureCall/agent-world-model"
RESULTS_FILE="$WORK_DIR/eval_results_v2/results.jsonl"
TARGET_SCENARIOS=100
TASKS_PER_SCENARIO=10
TARGET_TASKS=$((TARGET_SCENARIOS * TASKS_PER_SCENARIO))
STALL_TIMEOUT=300          # seconds without progress → restart
CHECK_INTERVAL=30          # polling interval

EVAL_CMD=".venv/bin/python3 eval_batch.py \
  --model Qwen/Qwen3-4B \
  --vllm_url http://127.0.0.1:8000/v1 \
  --num_workers 8 \
  --max_scenarios $TARGET_SCENARIOS \
  --max_iterations 15 \
  --temperature 0.6 \
  --enable_thinking \
  --resume \
  --output_dir eval_results_v2"

get_task_count() {
    if [ -f "$RESULTS_FILE" ]; then
        wc -l < "$RESULTS_FILE"
    else
        echo 0
    fi
}

get_scenario_count() {
    if [ -f "$RESULTS_FILE" ]; then
        python3 -c "
import json
scenarios = set()
for line in open('$RESULTS_FILE'):
    try: scenarios.add(json.loads(line.strip())['scenario'])
    except: pass
print(len(scenarios))
" 2>/dev/null || echo 0
    else
        echo 0
    fi
}

kill_eval() {
    # Kill eval_batch.py and all its MCP server children
    pkill -f "eval_batch.py.*eval_results_v2" 2>/dev/null || true
    sleep 2
    pkill -9 -f "eval_batch.py.*eval_results_v2" 2>/dev/null || true
    pkill -f "temp_server_" 2>/dev/null || true
    sleep 1
    pkill -9 -f "temp_server_" 2>/dev/null || true
    # Free ports
    for port in $(seq 8001 8008); do
        lsof -ti :$port 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    done
    sleep 2
}

kill_vllm() {
    echo "[$(date)] Killing vLLM server..."
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    sleep 3
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    echo "[$(date)] vLLM killed."
}

start_eval() {
    echo "[$(date)] Starting eval_batch.py with --resume..."
    cd "$WORK_DIR"
    $EVAL_CMD &
    EVAL_PID=$!
    echo "[$(date)] eval_batch.py started (PID=$EVAL_PID)"
}

echo "==========================================="
echo " AWM Eval Watchdog"
echo " Target: $TARGET_SCENARIOS scenarios ($TARGET_TASKS tasks)"
echo " Stall timeout: ${STALL_TIMEOUT}s"
echo " Check interval: ${CHECK_INTERVAL}s"
echo "==========================================="

# Main loop
LAST_COUNT=$(get_task_count)
LAST_PROGRESS_TIME=$(date +%s)
RESTART_COUNT=0

while true; do
    sleep $CHECK_INTERVAL

    CURRENT_COUNT=$(get_task_count)
    CURRENT_SCENARIOS=$(get_scenario_count)
    NOW=$(date +%s)

    # Check if all done
    if [ "$CURRENT_SCENARIOS" -ge "$TARGET_SCENARIOS" ]; then
        echo "[$(date)] All $CURRENT_SCENARIOS scenarios complete! ($CURRENT_COUNT tasks)"
        kill_eval
        kill_vllm
        echo "[$(date)] Watchdog finished. All done!"
        exit 0
    fi

    # Check if progress was made
    if [ "$CURRENT_COUNT" -gt "$LAST_COUNT" ]; then
        LAST_COUNT=$CURRENT_COUNT
        LAST_PROGRESS_TIME=$NOW
        echo "[$(date)] Progress: $CURRENT_COUNT tasks, $CURRENT_SCENARIOS/$TARGET_SCENARIOS scenarios"
        continue
    fi

    # No progress - check if eval is even running
    if ! pgrep -f "eval_batch.py.*eval_results_v2" > /dev/null 2>&1; then
        echo "[$(date)] eval_batch.py not running! Restarting... ($CURRENT_SCENARIOS/$TARGET_SCENARIOS done)"
        RESTART_COUNT=$((RESTART_COUNT + 1))
        start_eval
        LAST_PROGRESS_TIME=$NOW
        continue
    fi

    # Eval is running but stalled
    STALL_DURATION=$((NOW - LAST_PROGRESS_TIME))
    if [ "$STALL_DURATION" -ge "$STALL_TIMEOUT" ]; then
        RESTART_COUNT=$((RESTART_COUNT + 1))
        echo "[$(date)] STALLED for ${STALL_DURATION}s! Restart #$RESTART_COUNT ($CURRENT_SCENARIOS/$TARGET_SCENARIOS done)"
        kill_eval
        sleep 3
        start_eval
        LAST_COUNT=$(get_task_count)
        LAST_PROGRESS_TIME=$(date +%s)
    fi
done
