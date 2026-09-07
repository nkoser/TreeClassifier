#!/bin/bash
#SBATCH --job-name=sam3_ceiling
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
set -uo pipefail
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
EXTRA=${EXTRA:-}
echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
# shellcheck disable=SC2086
apptainer exec --nv --bind "$REPO":"$REPO" --bind /scratch:/scratch --pwd "$REPO" \
    --env HF_HOME=/scratch/shared/${USER}/hf_cache \
    /scratch/shared/${USER}/containers/vesselgpt_7.sif \
    /scratch/shared/${USER}/envs/treeclf/bin/python -u crownseg/sam3_ceiling.py --device auto $EXTRA
echo "=== Exit-Status: $? ==="
