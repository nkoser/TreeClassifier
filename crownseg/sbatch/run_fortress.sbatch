#!/bin/bash
#SBATCH --job-name=fortress_kronen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
#   SITES="CFB014 CFB019" sbatch crownseg/sbatch/run_fortress.sbatch
#   sbatch crownseg/sbatch/run_fortress.sbatch          # alle 47 Gebiete

set -uo pipefail
SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
HF_HOME=${HF_HOME:-/scratch/shared/${USER}/hf_cache}
OUT=${OUT:-/scratch/shared/${USER}/data/fortress/kronen}
SITES=${SITES:-}
EXTRA=${EXTRA:-}

echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
# shellcheck disable=SC2086
apptainer exec --nv --bind "$REPO":"$REPO" --bind /scratch:/scratch --pwd "$REPO" \
    --env HF_HOME="$HF_HOME" "$SIF" \
    "$ENV_DIR/bin/python" -u crownseg/fortress.py --out "$OUT" --device auto \
    ${SITES:+--sites $SITES} $EXTRA
status=$?
echo "=== Exit-Status: $status ==="
exit $status
