#!/bin/bash
#SBATCH --job-name=treeclf_cluster
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
# Kronen nach DINOvTree-Merkmalen clustern, statt die 14 kanadischen Klassen
# zu benutzen. Ausgabe: Kontaktboegen pro Cluster zum Benennen.
#
# Submit:
#   sbatch sbatch/run_cluster_crowns.sbatch
#   CLUSTERS=12 FEATURES=backbone sbatch sbatch/run_cluster_crowns.sbatch
#   METHOD=hdbscan sbatch sbatch/run_cluster_crowns.sbatch

set -uo pipefail

SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
HF_HOME=${HF_HOME:-/scratch/shared/${USER}/hf_cache}

SEGMENTS=${SEGMENTS:-$REPO/results_merged/s0.10_c16}
OUT=${OUT:-$REPO/results_cluster}
FEATURES=${FEATURES:-head}
METHOD=${METHOD:-kmeans}
CLUSTERS=${CLUSTERS:-}
EXTRA=${EXTRA:-}

inc () {
    apptainer exec --nv \
        --bind "$REPO":"$REPO" --bind /scratch:/scratch --bind /cold:/cold \
        --pwd "$REPO" --env HF_HOME="$HF_HOME",HF_HUB_OFFLINE=1 "$SIF" "$@"
}

echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
echo "Segmente: $SEGMENTS | Merkmale: $FEATURES | Verfahren: $METHOD"

# shellcheck disable=SC2086
inc "$ENV_DIR/bin/python" -u cluster_crowns.py \
    --segments "$SEGMENTS" --out "$OUT" --features "$FEATURES" --method "$METHOD" \
    ${CLUSTERS:+--clusters $CLUSTERS} --device auto $EXTRA

status=$?
echo "=== Exit-Status: $status ==="
exit $status
