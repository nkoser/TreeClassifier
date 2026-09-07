#!/bin/bash
#SBATCH --job-name=treeclf_arten
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
# Artklassifikation auf segmentierten Kronen (statt DeepForest-Boxen).
# Crop-Groesse kommt aus dem gemessenen Kronendurchmesser.
#
# Submit:
#   sbatch sbatch/run_classify_crowns.sbatch
#   SEGMENTS=results_merged/s0.10_c16 OUT=results_arten_merged sbatch sbatch/run_classify_crowns.sbatch
#   CROP_FACTOR=2.0 sbatch sbatch/run_classify_crowns.sbatch

set -uo pipefail

SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
HF_HOME=${HF_HOME:-/scratch/shared/${USER}/hf_cache}

SEGMENTS=${SEGMENTS:-$REPO/results_sam3/fix2}
OUT=${OUT:-$REPO/results_arten}
CROP_FACTOR=${CROP_FACTOR:-2.5}
EXTRA=${EXTRA:-}

inc () {
    apptainer exec --nv \
        --bind "$REPO":"$REPO" --bind /scratch:/scratch --bind /cold:/cold \
        --pwd "$REPO" --env HF_HOME="$HF_HOME",HF_HUB_OFFLINE=1 "$SIF" "$@"
}

echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
echo "Segmente: $SEGMENTS | Crop-Faktor: $CROP_FACTOR"

# shellcheck disable=SC2086
inc "$ENV_DIR/bin/python" -u classify_crowns.py \
    --segments "$SEGMENTS" --out "$OUT" --crop-factor "$CROP_FACTOR" --device auto $EXTRA

status=$?
echo "=== Exit-Status: $status ==="
exit $status
