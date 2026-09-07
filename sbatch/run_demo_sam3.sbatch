#!/bin/bash
#SBATCH --job-name=treeclf_demo_sam3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
# Run the standalone demo (demo_sam3_multiscale.py) on the cluster, so the
# output can be checked before handing the script to someone else. The demo
# itself needs none of this -- it runs on any machine with a GPU and the
# packages listed in its docstring.
#
# Submit:
#   sbatch sbatch/run_demo_sam3.sbatch
#   IMAGE=/cold/Mahfuz/chosen_frames/pines/frame_000006.jpg sbatch sbatch/run_demo_sam3.sbatch
#   EXTRA="--tile-levels 2 3 --threshold 0.25" sbatch sbatch/run_demo_sam3.sbatch

set -uo pipefail

SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
HF_HOME=${HF_HOME:-/scratch/shared/${USER}/hf_cache}

IMAGE=${IMAGE:-/cold/Mahfuz/chosen_frames/dense/frame_000073.jpg}
OUT=${OUT:-$REPO/results_demo_sam3}
EXTRA=${EXTRA:-}

inc () {
    apptainer exec --nv \
        --bind "$REPO":"$REPO" \
        --bind /scratch:/scratch \
        --bind /cold:/cold \
        --pwd "$REPO" \
        --env HF_HOME="$HF_HOME",TORCH_HOME="$HF_HOME/torch" \
        "$SIF" "$@"
}

echo "=== Node: $(hostname), job ${SLURM_JOB_ID:-local} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "Image:  $IMAGE"
echo "Output: $OUT"

# shellcheck disable=SC2086
inc "$ENV_DIR/bin/python" -u demo_sam3_multiscale.py \
    --image "$IMAGE" \
    --out "$OUT" \
    $EXTRA

status=$?
echo "=== Exit status: $status ==="
exit $status
