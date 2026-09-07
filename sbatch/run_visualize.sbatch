#!/bin/bash
#SBATCH --job-name=treeclf_views
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
# Diagnoseansichten aus den gespeicherten Labelkarten. Braucht keine Modelle,
# laeuft in Sekunden und laesst sich beliebig oft neu rendern.
#
# Submit:
#   sbatch sbatch/run_visualize.sbatch
#   FRAMES="dense/frame_000073.jpg" VIEWS="instanzen luecken" sbatch sbatch/run_visualize.sbatch

set -uo pipefail

SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
HF_HOME=${HF_HOME:-/scratch/shared/${USER}/hf_cache}

SEGMENTS=${SEGMENTS:-$REPO/results_hybrid}
OUT=${OUT:-$REPO/results_views}
FRAMES=${FRAMES:-}
VIEWS=${VIEWS:-}

inc () {
    apptainer exec --nv \
        --bind "$REPO":"$REPO" --bind /scratch:/scratch --bind /cold:/cold \
        --pwd "$REPO" --env HF_HOME="$HF_HOME",HF_HUB_OFFLINE=1 "$SIF" "$@"
}

echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="

# shellcheck disable=SC2086
inc "$ENV_DIR/bin/python" -u visualize_crowns.py \
    --segments "$SEGMENTS" --out "$OUT" \
    ${FRAMES:+--frames $FRAMES} ${VIEWS:+--views $VIEWS} --surface "${SURFACE:-depth}"

status=$?
echo "=== Exit-Status: $status ==="
exit $status
