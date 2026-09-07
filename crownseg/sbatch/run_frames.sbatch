#!/bin/bash
#SBATCH --job-name=crownseg_frames
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
# Ein trainiertes Modell auf die eigenen Drohnenframes anwenden und die
# Diagnoseansichten daraus rendern. Zwei Schritte in einem Job, damit
# Labelkarten und Ansichten nicht auseinanderlaufen.
#
#   MODEL=eomt        sbatch crownseg/sbatch/run_frames.sbatch
#   MODEL=mask2former sbatch crownseg/sbatch/run_frames.sbatch
#   MODEL=maskrcnn    sbatch crownseg/sbatch/run_frames.sbatch
#   MODEL=eomt DEPTH=1 NAME=eomt_tiefe sbatch crownseg/sbatch/run_frames.sbatch
#
# Der Massstab je Ordner stammt aus scale_probe.py, nicht aus der Flughoehe --
# die ist fuer die meisten Ordner unbekannt, und die Annahme 100 m war falsch.

set -uo pipefail

SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
HF_HOME=${HF_HOME:-/scratch/shared/${USER}/hf_cache}

MODEL=${MODEL:-eomt}
NAME=${NAME:-$MODEL}
FRAMES_DIR=${FRAMES_DIR:-/cold/Mahfuz/chosen_frames}
# Gemessen mit scale_probe.py (Job 736). Fuer die vier Waldordner liegt das
# Maximum sauber im Inneren des Suchbereichs; fuer 100, 80m, mixed1 und urban
# gibt es kein verwertbares Signal -- die Sonde nutzt Mask R-CNN, und das
# reagiert auf urbanem Material praktisch nicht. urban=1.5 stammt deshalb aus
# einem Sichtvergleich (0.4 bis 3.0 durchprobiert); ueber 1.5 fuellt das Modell
# ganze Suchfenster mit Masken, wo Rasen liegt. Fuer 100, 80m und mixed1 bleibt
# 1.0 -- das ist eine Annahme, keine Messung.
SCALES=${SCALES:-dense=1.2 dense1=1.2 mixed=1.0 pines=1.2 100=1.0 80m=1.0 mixed1=1.0 urban=1.5}
DEPTH=${DEPTH:-}
SEGMENTS=$REPO/results_frames_$NAME
VIEWS_OUT=$REPO/results_views_$NAME
EXTRA=${EXTRA:-}

inc () {
    apptainer exec --nv \
        --bind "$REPO":"$REPO" --bind /scratch:/scratch --bind /cold:/cold \
        --pwd "$REPO" --env HF_HOME="$HF_HOME" "$SIF" "$@"
}

echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
echo "Modell: $MODEL -> $SEGMENTS und $VIEWS_OUT"

echo "--- Schritt 1: Kronen vorhersagen ---"
# shellcheck disable=SC2086
if [ "$MODEL" = "maskrcnn" ]; then
    inc "$ENV_DIR/bin/python" -u crownseg/maskrcnn.py --mode predict --device auto \
        --frames-dir "$FRAMES_DIR" --out "$SEGMENTS" --scales $SCALES $EXTRA
else
    inc "$ENV_DIR/bin/python" -u crownseg/queryseg.py --arch "$MODEL" --mode predict --device auto \
        --frames-dir "$FRAMES_DIR" --out "$SEGMENTS" --scales $SCALES \
        ${DEPTH:+--depth} $EXTRA
fi
status=$?
[ $status -ne 0 ] && { echo "=== Vorhersage fehlgeschlagen: $status ==="; exit $status; }

echo "--- Schritt 2: Ansichten rendern ---"
inc "$ENV_DIR/bin/python" -u visualize_crowns.py \
    --input "$FRAMES_DIR" --segments "$SEGMENTS" --out "$VIEWS_OUT"

status=$?
echo "=== Exit-Status: $status ==="
exit $status
