#!/bin/bash
#SBATCH --job-name=crownseg_cluster
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
#   LABELS=results_sam3/multiskala sbatch crownseg/sbatch/run_cluster.sbatch

set -uo pipefail
SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
HF_HOME=${HF_HOME:-/scratch/shared/${USER}/hf_cache}
LABELS=${LABELS:-results_frames_eomt_v2}
OUT=${OUT:-$REPO/results_cluster_crownseg}
CLUSTERS=${CLUSTERS:-12}
SCALES=${SCALES:-dense=1.2 dense1=1.2 mixed=1.0 pines=1.2 100=1.0 80m=1.0 mixed1=1.0 urban=1.5}
EXTRA=${EXTRA:-}

echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
echo "Instanzen: $LABELS"
# shellcheck disable=SC2086
apptainer exec --nv --bind "$REPO":"$REPO" --bind /scratch:/scratch --bind /cold:/cold \
    --pwd "$REPO" --env HF_HOME="$HF_HOME",HF_HUB_OFFLINE=1 "$SIF" \
    "$ENV_DIR/bin/python" -u crownseg/cluster.py --labels "$LABELS" --out "$OUT" \
    --clusters "$CLUSTERS" --scales $SCALES --device auto $EXTRA
status=$?
echo "=== Exit-Status: $status ==="
exit $status
