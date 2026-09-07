#!/bin/bash
#SBATCH --job-name=crownseg_maskrcnn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
# Mask R-CNN auf BAMFORESTS: Training, Auswertung, Anwendung.
#
# Submit:
#   sbatch crownseg/sbatch/run_maskrcnn.sbatch                    # Training
#   MODE=eval sbatch crownseg/sbatch/run_maskrcnn.sbatch          # test1 + test2
#   MODE=inspect sbatch crownseg/sbatch/run_maskrcnn.sbatch       # Bilder zum Anschauen
#   MODE=predict sbatch crownseg/sbatch/run_maskrcnn.sbatch       # eigene Drohnenframes
#   EXTRA="--epochs 40 --crop 768" sbatch crownseg/sbatch/run_maskrcnn.sbatch

set -uo pipefail

SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
# torchvision holt die COCO-Gewichte hierher, nicht ins Home.
TORCH_HOME=${TORCH_HOME:-/scratch/shared/${USER}/hf_cache}

MODE=${MODE:-train}
EXTRA=${EXTRA:-}

echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# shellcheck disable=SC2086
apptainer exec --nv \
    --bind "$REPO":"$REPO" --bind /scratch:/scratch --bind /cold:/cold \
    --pwd "$REPO" --env TORCH_HOME="$TORCH_HOME" "$SIF" \
    "$ENV_DIR/bin/python" -u crownseg/maskrcnn.py --mode "$MODE" --device auto $EXTRA

status=$?
echo "=== Exit-Status: $status ==="
exit $status
