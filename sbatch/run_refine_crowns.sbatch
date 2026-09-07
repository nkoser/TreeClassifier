#!/bin/bash
#SBATCH --job-name=treeclf_refine
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
# Instanzen aus dem Bild (SAM 3), Korrektur aus der Tiefe: teilen UND verschmelzen.
# Braucht keine GPU, laeuft aber im selben Container.
#
# Submit:
#   sbatch sbatch/run_refine_crowns.sbatch
#   SPLITPROM=0.25 sbatch sbatch/run_refine_crowns.sbatch
#   EXTRA="--no-merge" sbatch sbatch/run_refine_crowns.sbatch

set -uo pipefail

SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
HF_HOME=${HF_HOME:-/scratch/shared/${USER}/hf_cache}

SEGMENTS=${SEGMENTS:-$REPO/results_sam3/multiskala}
SPLITPROM=${SPLITPROM:-0.35}
OUT=${OUT:-$REPO/results_refined}
SPLIT=${SPLIT:-0.06}
COLOR=${COLOR:-12}
ROUNDS=${ROUNDS:-2}
EXTRA=${EXTRA:-}

inc () {
    apptainer exec --nv \
        --bind "$REPO":"$REPO" --bind /scratch:/scratch --bind /cold:/cold \
        --pwd "$REPO" --env HF_HOME="$HF_HOME",HF_HUB_OFFLINE=1 "$SIF" "$@"
}

echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
echo "Teilen ab Prominenz $SPLITPROM | Verschmelzen bei Sattel < $SPLIT und Farbe < $COLOR"

inc "$ENV_DIR/bin/python" -u refine_crowns.py \
    --segments "$SEGMENTS" --out "$OUT" \
    --split-threshold "$SPLIT" --color-threshold "$COLOR" --rounds "$ROUNDS" \
    --split-prominence "$SPLITPROM" --surface "${SURFACE:-depth}" $EXTRA

status=$?
echo "=== Exit-Status: $status ==="
exit $status
