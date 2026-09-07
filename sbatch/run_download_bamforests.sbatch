#!/bin/bash
#SBATCH --job-name=treeclf_bamforests
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
# BAMFORESTS herunterladen und entpacken (DLR, CC BY-NC-ND 3.0).
# 27160 annotierte Baumkronen aus vier Gebieten um Bamberg, COCO-Format,
# inklusive korrigiertem DSM. Quelle: https://pba-freesoftware.eoc.dlr.de/
#
# Submit:
#   sbatch sbatch/run_download_bamforests.sbatch
#   VARIANT=Bamberg_coco1024 sbatch sbatch/run_download_bamforests.sbatch

set -uo pipefail

DEST=${DEST:-/scratch/shared/${USER}/data/bamforests}
VARIANT=${VARIANT:-Bamberg_coco2048}
URL="https://pba-freesoftware.eoc.dlr.de/${VARIANT}.zip"

mkdir -p "$DEST"
cd "$DEST" || exit 1

echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
echo "Ziel: $DEST/${VARIANT}.zip"
df -h "$DEST" | tail -1

# -c erlaubt Fortsetzen, falls der Job neu gestartet werden muss.
wget -c --progress=dot:giga "$URL" || exit 1
ls -lh "${VARIANT}.zip"

echo "=== Entpacken ==="
unzip -q -o "${VARIANT}.zip" -d "$DEST" || exit 1

echo "=== Inhalt ==="
find "$DEST" -maxdepth 3 -type d | head -30
echo "--- Dateitypen ---"
find "$DEST" -type f ! -name "*.zip" | sed 's/.*\.//' | sort | uniq -c | sort -rn | head -10
echo "--- COCO-Dateien ---"
find "$DEST" -name "*.json" -o -name "*.geojson" | head -10
du -sh "$DEST"

echo "=== Exit-Status: $? ==="
