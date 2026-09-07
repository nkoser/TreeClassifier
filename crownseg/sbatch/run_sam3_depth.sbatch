#!/bin/bash
#SBATCH --job-name=crownseg_sam3depth
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
# SAM 3 mit Tiefe: Textprompt findet, Tiefe trennt und ergaenzt.
#
# Submit:
#   sbatch crownseg/sbatch/run_sam3_depth.sbatch
#   DEPTH=dav2 OUT=results_sam3depth_dav2 sbatch crownseg/sbatch/run_sam3_depth.sbatch

set -uo pipefail

SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
HF_HOME=${HF_HOME:-/scratch/shared/${USER}/hf_cache}

INPUT=${INPUT:-/scratch/shared/${USER}/data/bamforests/subset}
OUT=${OUT:-$REPO/results_sam3depth}
DEPTH=${DEPTH:-depthpro}
EXTRA=${EXTRA:-}

echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
echo "Tiefenmodell: $DEPTH | Eingabe: $INPUT"

# shellcheck disable=SC2086
apptainer exec --nv \
    --bind "$REPO":"$REPO" --bind /scratch:/scratch --bind /cold:/cold \
    --pwd "$REPO" --env HF_HOME="$HF_HOME" "$SIF" \
    "$ENV_DIR/bin/python" -u crownseg/sam3_depth.py \
    --input "$INPUT" --out "$OUT" --depth-model "$DEPTH" --device auto $EXTRA

status=$?
echo "=== Exit-Status: $status ==="
exit $status
