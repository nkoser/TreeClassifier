#!/bin/bash
#SBATCH --job-name=depthft_3d
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err

set -uo pipefail
SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
HF_HOME=${HF_HOME:-/scratch/shared/${USER}/hf_cache}
RUN=${RUN:-/scratch/shared/${USER}/runs/depthft}
OUT=${OUT:-$REPO/results_depthft_3d}
HFOV=${HFOV:-48.0}
EXTRA=${EXTRA:-}

echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
# shellcheck disable=SC2086
apptainer exec --bind "$REPO":"$REPO" --bind /scratch:/scratch --bind /cold:/cold --pwd "$REPO" \
    --env HF_HOME="$HF_HOME" "$SIF" \
    "$ENV_DIR/bin/python" -u depthft/wolke3d.py --ft "$RUN/bestes" --out "$OUT" \
    --hfov-deg "$HFOV" $EXTRA
status=$?
echo "=== Exit-Status: $status ==="
exit $status
