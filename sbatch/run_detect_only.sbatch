#!/bin/bash
#SBATCH --job-name=treeclf_detect
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
# Nur die DeepForest-Detektionen, ohne Klassifikation. Zeigt, welche Instanzen
# ueberhaupt gefunden werden und welche die Filter ueberstehen.
#
# Submit:
#   sbatch sbatch/run_detect_only.sbatch
#   MIN_SCORE=0.2 SHOW_SCORES=1 sbatch sbatch/run_detect_only.sbatch

set -uo pipefail

SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
HF_HOME=${HF_HOME:-/scratch/shared/${USER}/hf_cache}

INPUT=${INPUT:-/cold/Mahfuz/chosen_frames}
OUT=${OUT:-$REPO/results_detect}
MIN_SCORE=${MIN_SCORE:-0.35}
MIN_BOX=${MIN_BOX:-25}
MAX_TREES=${MAX_TREES:-150}
FRAMES_PER_FOLDER=${FRAMES_PER_FOLDER:-0}
SHOW_SCORES=${SHOW_SCORES:-}

inc () {
    apptainer exec --nv \
        --bind "$REPO":"$REPO" \
        --bind /scratch:/scratch \
        --bind /cold:/cold \
        --pwd "$REPO" \
        --env HF_HOME="$HF_HOME",TORCH_HOME="$HF_HOME/torch" \
        "$SIF" "$@"
}

echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

inc "$ENV_DIR/bin/python" -u detect_only.py \
    --input "$INPUT" \
    --out "$OUT" \
    --min-score "$MIN_SCORE" \
    --min-box-px "$MIN_BOX" \
    --max-trees-per-frame "$MAX_TREES" \
    --frames-per-folder "$FRAMES_PER_FOLDER" \
    ${SHOW_SCORES:+--show-scores}

status=$?
echo "=== Exit-Status: $status ==="
exit $status
