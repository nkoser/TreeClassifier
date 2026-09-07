#!/bin/bash
#SBATCH --job-name=crownseg_videopar
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
# Gemessene Hoehenkarten aus einem Drohnenvideo (Parallaxe).
#   sbatch crownseg/sbatch/run_video_parallax.sbatch
#   COUNT=120 STRIDE=6 sbatch crownseg/sbatch/run_video_parallax.sbatch

set -uo pipefail
SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
VIDEO=${VIDEO:-/cold/Mahfuz/DJI_20230506174726_0004_Z_80m.MP4}
STRIDE=${STRIDE:-10}
COUNT=${COUNT:-60}
START=${START:-0}
EXTRA=${EXTRA:-}

echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
echo "Video: $VIDEO | jeder $STRIDE. Frame | $COUNT Stueck ab $START"
# shellcheck disable=SC2086
apptainer exec --bind "$REPO":"$REPO" --bind /scratch:/scratch --bind /cold:/cold \
    --pwd "$REPO" "$SIF" "$ENV_DIR/bin/python" -u crownseg/video_parallax.py \
    --video "$VIDEO" --stride "$STRIDE" --count "$COUNT" --start "$START" $EXTRA
status=$?
echo "=== Exit-Status: $status ==="
exit $status
