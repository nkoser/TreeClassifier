#!/bin/bash
#SBATCH --job-name=treeclf_stereo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
# Echte Parallaxe aus aufeinanderfolgenden Videoframes. Rein CPU-seitig (OpenCV),
# laeuft aber im selben Container.
#
# Submit:
#   sbatch sbatch/run_stereo_probe.sbatch
#   FOLDERS="80m dense1" sbatch sbatch/run_stereo_probe.sbatch

set -uo pipefail

SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
HF_HOME=${HF_HOME:-/scratch/shared/${USER}/hf_cache}

OUT=${OUT:-$REPO/results_stereo}
FOLDERS=${FOLDERS:-}
EXTRA=${EXTRA:-}

inc () {
    apptainer exec --nv \
        --bind "$REPO":"$REPO" --bind /scratch:/scratch --bind /cold:/cold \
        --pwd "$REPO" --env HF_HOME="$HF_HOME",HF_HUB_OFFLINE=1 "$SIF" "$@"
}

echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="

# shellcheck disable=SC2086
inc "$ENV_DIR/bin/python" -u stereo_probe.py --out "$OUT" ${FOLDERS:+--folders $FOLDERS} $EXTRA

status=$?
echo "=== Exit-Status: $status ==="
exit $status
