#!/bin/bash
#SBATCH --account=rpi
#SBATCH --partition=rpi
#SBATCH --gres=gpu:nvidia_h200:8
#SBATCH --time=2-00:00:00
#SBATCH --job-name=esa-q3-8B
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=examples/awm_esa/logs/slurm_qwen3_8B_%j.log

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="${SCRIPT_DIR}/../.."

cd "${SLIME_DIR}" || exit 1

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

bash examples/awm_esa/run_qwen3_8B.sh

echo "Job $SLURM_JOB_ID finished at $(date)"
