#!/bin/bash
#SBATCH --job-name=treeclf_depth
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
# Probe: taugt monokulare Tiefenschaetzung als Ersatz-CHM fuer die
# Kronenabgrenzung? Nutzt die bereits im hf_cache liegenden Modelle.
#
# Submit:
#   sbatch sbatch/run_depth_probe.sbatch
#   MODELS="depth_anything" sbatch sbatch/run_depth_probe.sbatch

set -uo pipefail

SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
HF_HOME=${HF_HOME:-/scratch/shared/${USER}/hf_cache}

INPUT=${INPUT:-/cold/Mahfuz/chosen_frames}
OUT=${OUT:-$REPO/results_depth}
FRAMES=${FRAMES:-"pines/frame_000006.jpg dense/frame_000073.jpg mixed1/frame_000594.jpg 80m/frame_000297.jpg"}
MODELS=${MODELS:-"depth_anything depthpro"}

inc () {
    apptainer exec --nv \
        --bind "$REPO":"$REPO" --bind /scratch:/scratch --bind /cold:/cold \
        --pwd "$REPO" \
        --env HF_HOME="$HF_HOME",TORCH_HOME="$HF_HOME/torch",HF_HUB_OFFLINE=1 \
        "$SIF" "$@"
}

echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# shellcheck disable=SC2086
inc "$ENV_DIR/bin/python" -u depth_probe.py \
    --input "$INPUT" --frames $FRAMES --models $MODELS --out "$OUT" --device auto

status=$?
echo "=== Exit-Status: $status ==="
exit $status
