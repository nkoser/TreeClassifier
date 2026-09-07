#!/bin/bash
#SBATCH --job-name=treeclf_segment
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
# Kronenabgrenzung ueber monokulare Tiefe + Watershed.
#
# Submit:
#   sbatch sbatch/run_segment_trees.sbatch
#   CROWN_PX=110 FRAMES="80m/frame_000297.jpg" sbatch sbatch/run_segment_trees.sbatch
#   EXTRA="--save-chm" sbatch sbatch/run_segment_trees.sbatch

set -uo pipefail

SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
HF_HOME=${HF_HOME:-/scratch/shared/${USER}/hf_cache}

INPUT=${INPUT:-/cold/Mahfuz/chosen_frames}
OUT=${OUT:-$REPO/results_segment}
CROWN_PX=${CROWN_PX:-100}
FRAMES=${FRAMES:-}
EXTRA=${EXTRA:-}

inc () {
    apptainer exec --nv \
        --bind "$REPO":"$REPO" --bind /scratch:/scratch --bind /cold:/cold \
        --pwd "$REPO" \
        --env HF_HOME="$HF_HOME",TORCH_HOME="$HF_HOME/torch",HF_HUB_OFFLINE=1 \
        "$SIF" "$@"
}

echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# shellcheck disable=SC2086
inc "$ENV_DIR/bin/python" -u segment_trees.py \
    --input "$INPUT" --out "$OUT" --crown-px "$CROWN_PX" --device auto \
    ${FRAMES:+--frames $FRAMES} \
    $EXTRA

status=$?
echo "=== Exit-Status: $status ==="
exit $status
