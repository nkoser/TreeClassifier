#!/bin/bash
#SBATCH --job-name=depthft_prepare
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err

set -uo pipefail
SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
HF_HOME=${HF_HOME:-/scratch/shared/${USER}/hf_cache}
DATA=${DATA:-/scratch/shared/${USER}/data/fortress/depthft}
RUN=${RUN:-/scratch/shared/${USER}/runs/depthft}
EXTRA=${EXTRA:-}

echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
SITES=${SITES:-}
# shellcheck disable=SC2086
apptainer exec --bind "$REPO":"$REPO" --bind /scratch:/scratch --pwd "$REPO" \
    --env HF_HOME="$HF_HOME" "$SIF" \
    "$ENV_DIR/bin/python" -u depthft/prepare.py --out "$DATA" \
    ${SITES:+--sites $SITES} $EXTRA
status=$?
echo "=== Exit-Status: $status ==="
exit $status
