#!/bin/bash
#SBATCH --job-name=treeclf_scaletest
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
# Testet, ob DeepForest auf herunterskalierten Frames kronengrosse Instanzen
# findet statt Kronenfragmente. Verkleinern entspricht hoeherem Flug.
#
# Submit:
#   sbatch sbatch/run_detect_scale_test.sbatch
#   SCALES="1.0 0.6 0.4 0.3" sbatch sbatch/run_detect_scale_test.sbatch

set -uo pipefail

SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
HF_HOME=${HF_HOME:-/scratch/shared/${USER}/hf_cache}

INPUT=${INPUT:-/cold/Mahfuz/chosen_frames}
OUT=${OUT:-$REPO/results_scaletest}
SCALES=${SCALES:-"1.0 0.7 0.5 0.35 0.25"}
FRAMES=${FRAMES:-"pines/frame_000006.jpg dense/frame_000073.jpg mixed1/frame_000594.jpg 80m/frame_000297.jpg"}

inc () {
    apptainer exec --nv \
        --bind "$REPO":"$REPO" \
        --bind /scratch:/scratch \
        --bind /cold:/cold \
        --pwd "$REPO" \
        --env HF_HOME="$HF_HOME",TORCH_HOME="$HF_HOME/torch" \
        "$SIF" "$@"
}

echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# shellcheck disable=SC2086
inc "$ENV_DIR/bin/python" -u detect_scale_test.py \
    --input "$INPUT" \
    --frames $FRAMES \
    --scales $SCALES \
    --out "$OUT"

status=$?
echo "=== Exit-Status: $status ==="
exit $status
