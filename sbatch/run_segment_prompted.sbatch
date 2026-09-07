#!/bin/bash
#SBATCH --job-name=treeclf_prompted
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
# Tiefe liefert die Wipfel als Prompt-Punkte, SAM zieht die Kronengrenzen.
#
# Submit:
#   sbatch sbatch/run_segment_prompted.sbatch
#   PROMINENCE=0.08 FRAMES="80m/frame_000297.jpg" sbatch sbatch/run_segment_prompted.sbatch

set -uo pipefail

SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
HF_HOME=${HF_HOME:-/scratch/shared/${USER}/hf_cache}

INPUT=${INPUT:-/cold/Mahfuz/chosen_frames}
OUT=${OUT:-$REPO/results_prompted}
PROMINENCE=${PROMINENCE:-0.04}
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
echo "Wipfel-Prominenz: $PROMINENCE"

# shellcheck disable=SC2086
inc "$ENV_DIR/bin/python" -u segment_prompted.py \
    --input "$INPUT" --out "$OUT" --device auto \
    --peak-prominence "$PROMINENCE" --select "${SELECT:-basin}" \
    ${FRAMES:+--frames $FRAMES} \
    $EXTRA

status=$?
echo "=== Exit-Status: $status ==="
exit $status
