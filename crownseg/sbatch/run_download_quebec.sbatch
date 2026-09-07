#!/bin/bash
#SBATCH --job-name=quebec_download
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
# Quebec Trees Dataset (Cloutier et al. 2023, CC-BY-4.0).
#
# Nur ein Aufnahmedatum: 2021-09-02, weil die Kronensegmentierung auf genau
# dieser Befliegung gemacht wurde. Die anderen sechs Termine zeigen dieselben
# Baeume zu anderen Zeitpunkten und braeuchten wir nur fuer Phaenologie.
#
#   sbatch crownseg/sbatch/run_download_quebec.sbatch
#   DATE=2021-06-17 sbatch crownseg/sbatch/run_download_quebec.sbatch

set -uo pipefail
DATE=${DATE:-2021-09-02}
DEST=${DEST:-/scratch/shared/${USER}/data/quebec_trees}
URL="https://zenodo.org/records/8148479/files/quebec_trees_dataset_${DATE}.zip"

mkdir -p "$DEST"
echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
echo "Ziel: $DEST/quebec_trees_dataset_${DATE}.zip"
df -h "$DEST" | tail -1

# -c setzt einen abgebrochenen Download fort statt neu zu beginnen.
wget -c --progress=dot:giga -O "$DEST/quebec_trees_dataset_${DATE}.zip" "$URL"
status=$?
[ $status -ne 0 ] && { echo "=== Download fehlgeschlagen: $status ==="; exit $status; }

ls -la "$DEST"
echo "--- Inhalt ---"
unzip -l "$DEST/quebec_trees_dataset_${DATE}.zip" | head -30
echo "=== Exit-Status: 0 ==="
