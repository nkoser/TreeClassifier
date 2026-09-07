#!/bin/bash
#SBATCH --job-name=treeclf_ablation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
# Ablation ueber SAM-Varianten (v1 base/large/huge, v2, v2.1) bei identischen
# Filterregeln. SAM 3 ist gated -- mit HF_TOKEN nachziehbar.
#
# Submit:
#   sbatch sbatch/run_ablate_sam.sbatch
#   MODELS="sam_vit_huge sam2.1_large" sbatch sbatch/run_ablate_sam.sbatch
#   HF_TOKEN=hf_... MODELS="sam3" sbatch sbatch/run_ablate_sam.sbatch

set -uo pipefail

SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
HF_HOME=${HF_HOME:-/scratch/shared/${USER}/hf_cache}
HF_TOKEN=${HF_TOKEN:-}

OUT=${OUT:-$REPO/results_ablation}
MODELS=${MODELS:-}
FRAMES=${FRAMES:-}

inc () {
    apptainer exec --nv \
        --bind "$REPO":"$REPO" --bind /scratch:/scratch --bind /cold:/cold \
        --pwd "$REPO" \
        --env HF_HOME="$HF_HOME",TORCH_HOME="$HF_HOME/torch",HF_TOKEN="$HF_TOKEN" \
        "$SIF" "$@"
}

echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# shellcheck disable=SC2086
inc "$ENV_DIR/bin/python" -u ablate_sam.py \
    --out "$OUT" --device auto \
    ${MODELS:+--models $MODELS} \
    ${FRAMES:+--frames $FRAMES}

status=$?
echo "=== Exit-Status: $status ==="
exit $status
