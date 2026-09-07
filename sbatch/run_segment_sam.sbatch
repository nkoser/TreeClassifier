#!/bin/bash
#SBATCH --job-name=treeclf_sam
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
# Kronenabgrenzung mit SAM (automatische Maskengenerierung).
#
# Submit:
#   FRAMES="100/frame_000537.jpg" sbatch sbatch/run_segment_sam.sbatch
#   CROWN_PX=120 EXTRA="--crop-layers 0" sbatch sbatch/run_segment_sam.sbatch

set -uo pipefail

SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
HF_HOME=${HF_HOME:-/scratch/shared/${USER}/hf_cache}

INPUT=${INPUT:-/cold/Mahfuz/chosen_frames}
OUT=${OUT:-$REPO/results_sam}
CROWN_PX=${CROWN_PX:-100}
FRAMES=${FRAMES:-}
EXTRA=${EXTRA:-}

inc () {
    apptainer exec --nv \
        --bind "$REPO":"$REPO" --bind /scratch:/scratch --bind /cold:/cold \
        --pwd "$REPO" \
        --env HF_HOME="$HF_HOME",TORCH_HOME="$HF_HOME/torch" \
        "$SIF" "$@"
}

echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# shellcheck disable=SC2086
inc "$ENV_DIR/bin/python" -u segment_sam.py \
    --input "$INPUT" --out "$OUT" --crown-px "$CROWN_PX" --device auto \
    ${FRAMES:+--frames $FRAMES} \
    $EXTRA

status=$?
echo "=== Exit-Status: $status ==="
exit $status
