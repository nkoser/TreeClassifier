#!/bin/bash
#SBATCH --job-name=treeclf_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
# Diagnose: wie stark haengt die Artvorhersage vom angenommenen Bildmassstab ab?
# Klassifiziert dieselben Detektionen bei verschiedenen Ausschnittsgroessen.
#
# Submit:
#   sbatch sbatch/run_scale_sweep.sbatch
#   FRAMES="/cold/Mahfuz/chosen_frames/dense/frame_000073.jpg" sbatch sbatch/run_scale_sweep.sbatch

set -uo pipefail

SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
CKPT=${CKPT:-/scratch/shared/${USER}/data/treeclf/checkpoints/dinovtreeb_quebectrees.pth}
HF_HOME=${HF_HOME:-/scratch/shared/${USER}/hf_cache}

FRAMES=${FRAMES:-"/cold/Mahfuz/chosen_frames/pines/frame_000006.jpg /cold/Mahfuz/chosen_frames/80m/frame_000297.jpg"}
CROP_SIZES=${CROP_SIZES:-"100 150 250 400 600"}
N_TREES=${N_TREES:-8}
OUT=${OUT:-$REPO/results/scale_sweep.csv}

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

# shellcheck disable=SC2086
inc "$ENV_DIR/bin/python" -u scale_sweep.py \
    --frames $FRAMES \
    --crop-sizes $CROP_SIZES \
    --n-trees "$N_TREES" \
    --ckpt "$CKPT" \
    --out "$OUT" \
    --batch-size 32 \
    --device auto

status=$?
echo "=== Exit-Status: $status ==="
exit $status
