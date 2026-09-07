#!/bin/bash
#SBATCH --job-name=quebec_prepare
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
set -uo pipefail
SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
EXTRA=${EXTRA:-}
echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
# shellcheck disable=SC2086
apptainer exec --bind "$REPO":"$REPO" --bind /scratch:/scratch --pwd "$REPO" "$SIF" \
    "$ENV_DIR/bin/python" -u crownseg/quebec.py $EXTRA
status=$?
echo "=== Exit-Status: $status ==="
exit $status
