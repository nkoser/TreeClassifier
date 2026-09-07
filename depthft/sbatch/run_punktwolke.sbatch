#!/bin/bash
#SBATCH --job-name=depthft_wolke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err

set -uo pipefail
SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
HF_HOME=${HF_HOME:-/scratch/shared/${USER}/hf_cache}
RUN=${RUN:-/scratch/shared/${USER}/runs/depthft}
OUT=${OUT:-$REPO/results_depthft_wolken}
HFOV=${HFOV:-48.0}
SCHRITT=${SCHRITT:-2}
FORMAT=${FORMAT:-"ply las"}
EXTRA=${EXTRA:-}

echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
# shellcheck disable=SC2086
apptainer exec --nv --bind "$REPO":"$REPO" --bind /scratch:/scratch --bind /cold:/cold --pwd "$REPO" \
    --env HF_HOME="$HF_HOME" "$SIF" \
    "$ENV_DIR/bin/python" -u depthft/punktwolke.py --ft "$RUN/bestes" --out "$OUT" \
    --hfov-deg "$HFOV" --schritt "$SCHRITT" --format $FORMAT --device auto $EXTRA
status=$?
echo "=== Exit-Status: $status ==="
exit $status
