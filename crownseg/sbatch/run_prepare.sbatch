#!/bin/bash
#SBATCH --job-name=crownseg_prepare
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
# BAMFORESTS-TIFFs nach JPEG umschreiben und die Kronenpolygone je Kachel ablegen.
# Einmalig, ~2456 Kacheln, Ergebnis unter $DATA/bamforests/crownseg.
#
# Submit:
#   sbatch crownseg/sbatch/run_prepare.sbatch

set -uo pipefail

SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
SPLIT=${SPLIT:-all}

echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="

apptainer exec \
    --bind "$REPO":"$REPO" --bind /scratch:/scratch \
    --pwd "$REPO" "$SIF" \
    "$ENV_DIR/bin/python" -u crownseg/bamforests.py --split "$SPLIT" --workers 16

status=$?
echo "=== Exit-Status: $status ==="
exit $status
