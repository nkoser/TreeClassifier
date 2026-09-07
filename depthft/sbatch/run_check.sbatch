#!/bin/bash
#SBATCH --job-name=depthft_check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err

set -uo pipefail
SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
HF_HOME=${HF_HOME:-/scratch/shared/${USER}/hf_cache}
DATA=${DATA:-/scratch/shared/${USER}/data/fortress/depthft}
N=${N:-8}
SPLIT=${SPLIT:-train}
EXTRA=${EXTRA:-}

echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# shellcheck disable=SC2086
apptainer exec --nv --bind "$REPO":"$REPO" --bind /scratch:/scratch --pwd "$REPO" \
    --env HF_HOME="$HF_HOME" "$SIF" \
    "$ENV_DIR/bin/python" -u depthft/check.py --data "$DATA" --split "$SPLIT" --n "$N" \
    --out "$REPO/results_depthft/check" --device auto $EXTRA
status=$?
echo "=== Exit-Status: $status ==="
exit $status
