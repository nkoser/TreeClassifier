#!/bin/bash
#SBATCH --job-name=nurtiefe_frames
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
set -uo pipefail
SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
EXTRA=${EXTRA:-}
echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
# shellcheck disable=SC2086
apptainer exec --nv --bind "$REPO":"$REPO" --bind /scratch:/scratch --bind /cold:/cold --pwd "$REPO" \
    --env HF_HOME=/scratch/shared/${USER}/hf_cache,HF_HUB_OFFLINE=0 "$SIF" \
    "$ENV_DIR/bin/python" -u crownseg/frames_tiefe.py --device auto $EXTRA
status=$?
echo "=== Exit-Status: $status ==="
exit $status
