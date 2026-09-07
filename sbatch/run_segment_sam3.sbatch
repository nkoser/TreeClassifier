#!/bin/bash
#SBATCH --job-name=treeclf_sam3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
# Kronenabgrenzung mit SAM 3 (textgepromptet). Braucht Zugang zu facebook/sam3:
# Token nach $HF_HOME/token schreiben, siehe README.
#
# Submit:
#   sbatch sbatch/run_segment_sam3.sbatch
#   PROMPT="tree crown" THRESHOLD=0.25 sbatch sbatch/run_segment_sam3.sbatch
#   FRAMES="dense/frame_000073.jpg" TILES=3 sbatch sbatch/run_segment_sam3.sbatch

set -uo pipefail

SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
HF_HOME=${HF_HOME:-/scratch/shared/${USER}/hf_cache}

INPUT=${INPUT:-/cold/Mahfuz/chosen_frames}
OUT=${OUT:-$REPO/results_sam3}
PROMPT=${PROMPT:-tree}
THRESHOLD=${THRESHOLD:-0.3}
TILES=${TILES:-2}
TILES_MULTI=${TILES_MULTI:-}
FRAMES=${FRAMES:-}
EXTRA=${EXTRA:-}

if [ ! -s "$HF_HOME/token" ]; then
    echo "FEHLER: kein Token unter $HF_HOME/token -- facebook/sam3 ist gated."
    echo "  printf 'hf_...' > $HF_HOME/token && chmod 600 $HF_HOME/token"
    exit 1
fi

inc () {
    apptainer exec --nv \
        --bind "$REPO":"$REPO" --bind /scratch:/scratch --bind /cold:/cold \
        --pwd "$REPO" \
        --env HF_HOME="$HF_HOME",TORCH_HOME="$HF_HOME/torch" \
        "$SIF" "$@"
}

echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "Prompt: '$PROMPT' | Schwelle: $THRESHOLD | Kacheln: ${TILES}x${TILES}"

# shellcheck disable=SC2086
inc "$ENV_DIR/bin/python" -u segment_sam3.py \
    --input "$INPUT" --out "$OUT" --device auto \
    --prompt "$PROMPT" --threshold "$THRESHOLD" --tiles "$TILES" \
    ${TILES_MULTI:+--tiles-multi $TILES_MULTI} \
    ${FRAMES:+--frames $FRAMES} \
    $EXTRA

status=$?
echo "=== Exit-Status: $status ==="
exit $status
