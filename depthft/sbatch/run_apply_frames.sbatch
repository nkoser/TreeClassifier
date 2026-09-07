#!/bin/bash
#SBATCH --job-name=depthft_frames
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err

set -uo pipefail
SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
HF_HOME=${HF_HOME:-/scratch/shared/${USER}/hf_cache}
DATA=${DATA:-/scratch/shared/${USER}/data/fortress/depthft}
RUN=${RUN:-/scratch/shared/${USER}/runs/depthft}
EXTRA=${EXTRA:-}

echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
INPUT=${INPUT:-/cold/Mahfuz/chosen_frames}
OUT=${OUT:-$REPO/results_depthft_frames}
ALTITUDES=${ALTITUDES:-}
# shellcheck disable=SC2086
apptainer exec --nv --bind "$REPO":"$REPO" --bind /scratch:/scratch --bind /cold:/cold --pwd "$REPO" \
    --env HF_HOME="$HF_HOME" "$SIF" \
    "$ENV_DIR/bin/python" -u depthft/apply_frames.py --input "$INPUT" --ft "$RUN/bestes" \
    --out "$OUT" --device auto ${ALTITUDES:+--altitudes $ALTITUDES} $EXTRA
status=$?
echo "=== Exit-Status: $status ==="
exit $status
