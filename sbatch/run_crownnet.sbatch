#!/bin/bash
#SBATCH --job-name=treeclf_crownnet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
# Stufe 2: Kroneninstanznetz auf BAMFORESTS.
#
# Submit:
#   MODE=prepare sbatch sbatch/run_crownnet.sbatch
#   sbatch sbatch/run_crownnet.sbatch
#   MODE=predict sbatch sbatch/run_crownnet.sbatch

set -uo pipefail

SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
HF_HOME=${HF_HOME:-/scratch/shared/${USER}/hf_cache}

MODE=${MODE:-train}
EPOCHS=${EPOCHS:-30}
EXTRA=${EXTRA:-}

inc () {
    apptainer exec --nv \
        --bind "$REPO":"$REPO" --bind /scratch:/scratch --bind /cold:/cold \
        --pwd "$REPO" --env HF_HOME="$HF_HOME",HF_HUB_OFFLINE=1 "$SIF" "$@"
}

echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true

# shellcheck disable=SC2086
inc "$ENV_DIR/bin/python" -u crownnet.py --mode "$MODE" --epochs "$EPOCHS" --device auto $EXTRA

status=$?
echo "=== Exit-Status: $status ==="
exit $status
