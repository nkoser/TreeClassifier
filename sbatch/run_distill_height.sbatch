#!/bin/bash
#SBATCH --job-name=treeclf_distill
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
# Stufe 1: gemessene Parallaxe in ein Einzelbild-Netz destillieren.
#
# Submit:
#   sbatch sbatch/run_distill_height.sbatch
#   MODE=predict sbatch sbatch/run_distill_height.sbatch
#   EPOCHS=80 VAL_FOLDERS="dense pines" sbatch sbatch/run_distill_height.sbatch

set -uo pipefail

SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
HF_HOME=${HF_HOME:-/scratch/shared/${USER}/hf_cache}

MODE=${MODE:-train}
EPOCHS=${EPOCHS:-40}
VAL_FOLDERS=${VAL_FOLDERS:-"dense mixed"}
EXTRA=${EXTRA:-}

inc () {
    apptainer exec --nv \
        --bind "$REPO":"$REPO" --bind /scratch:/scratch --bind /cold:/cold \
        --pwd "$REPO" --env HF_HOME="$HF_HOME",HF_HUB_OFFLINE=1 "$SIF" "$@"
}

echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true

# shellcheck disable=SC2086
inc "$ENV_DIR/bin/python" -u distill_height.py \
    --mode "$MODE" --epochs "$EPOCHS" --val-folders $VAL_FOLDERS --device auto $EXTRA

status=$?
echo "=== Exit-Status: $status ==="
exit $status
