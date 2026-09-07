#!/bin/bash
#SBATCH --job-name=arten_fortress
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
#   LABELS=results_sam3/multiskala OUT=results_arten_fortress_sam3 sbatch crownseg/sbatch/run_apply_head.sbatch
set -uo pipefail
SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
LABELS=${LABELS:-results_frames_eomt_final}
OUT=${OUT:-results_arten_fortress}
KOPF=${KOPF:-/scratch/shared/${USER}/data/treeclf/checkpoints/kopf_fortress.pth}
SCALES=${SCALES:-dense=1.2 dense1=1.2 mixed=1.0 pines=1.2 100=1.0 80m=1.0 mixed1=1.0 urban=1.5}
echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} | $LABELS -> $OUT ==="
# shellcheck disable=SC2086
apptainer exec --nv --bind "$REPO":"$REPO" --bind /scratch:/scratch --bind /cold:/cold --pwd "$REPO" \
    --env HF_HOME=/scratch/shared/${USER}/hf_cache,HF_HUB_OFFLINE=1 "$SIF" \
    "$ENV_DIR/bin/python" -u crownseg/apply_head.py --device auto \
    --labels "$LABELS" --out "$REPO/$OUT" --kopf "$KOPF" --scales $SCALES
status=$?
echo "=== Exit-Status: $status ==="
exit $status
