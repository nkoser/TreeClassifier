#!/bin/bash
#SBATCH --job-name=treeclf_infer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
# Artklassifikation auf Drohnen-Frames: DeepForest -> baum-zentrierter Crop
# -> DINOvTree-B (Quebec Trees). Laeuft im vesselgpt_7.sif mit dem venv aus
# sbatch/setup_env.sbatch.
#
# Submit (Defaults):
#   sbatch sbatch/run_infer_species.sbatch
# Mit bekannten Flughoehen pro Ordner:
#   ALTITUDES="pines=35 dense=60 mixed=60" sbatch sbatch/run_infer_species.sbatch
# Anderer Eingabeordner / mehr Baeume pro Frame:
#   INPUT=/cold/Mahfuz/andere OUT=results_v2 MAX_TREES=200 sbatch sbatch/run_infer_species.sbatch

set -uo pipefail

SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
CKPT=${CKPT:-/scratch/shared/${USER}/data/treeclf/checkpoints/dinovtreeb_quebectrees.pth}
HF_HOME=${HF_HOME:-/scratch/shared/${USER}/hf_cache}

INPUT=${INPUT:-/cold/Mahfuz/chosen_frames}
OUT=${OUT:-$REPO/results}
MAX_TREES=${MAX_TREES:-150}
BATCH=${BATCH:-32}
ALTITUDES=${ALTITUDES:-}
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

echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "Input:  $INPUT"
echo "Output: $OUT"
echo "Ckpt:   $CKPT"
echo "Hoehen: ${ALTITUDES:-<aus Ordnernamen / Default>}"

# shellcheck disable=SC2086
inc "$ENV_DIR/bin/python" -u infer_species.py \
    --input "$INPUT" \
    --out "$OUT" \
    --ckpt "$CKPT" \
    --max-trees-per-frame "$MAX_TREES" \
    --batch-size "$BATCH" \
    --device auto \
    ${ALTITUDES:+--altitudes $ALTITUDES} \
    $EXTRA

status=$?
echo "=== Exit-Status: $status ==="
exit $status
