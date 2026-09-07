#!/bin/bash
#SBATCH --job-name=crownseg_query
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
# Query-basierte Kroneninstanzen: EoMT oder Mask2Former.
#
# Submit:
#   ARCH=eomt sbatch crownseg/sbatch/run_queryseg.sbatch
#   ARCH=mask2former MODE=eval sbatch crownseg/sbatch/run_queryseg.sbatch

set -uo pipefail

SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
HF_HOME=${HF_HOME:-/scratch/shared/${USER}/hf_cache}

ARCH=${ARCH:-eomt}
MODE=${MODE:-train}
EXTRA=${EXTRA:-}

echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "Architektur: $ARCH | Modus: $MODE"

# shellcheck disable=SC2086
apptainer exec --nv \
    --bind "$REPO":"$REPO" --bind /scratch:/scratch --bind /cold:/cold \
    --pwd "$REPO" --env HF_HOME="$HF_HOME" "$SIF" \
    "$ENV_DIR/bin/python" -u crownseg/queryseg.py --arch "$ARCH" --mode "$MODE" --device auto $EXTRA

status=$?
echo "=== Exit-Status: $status ==="
exit $status
