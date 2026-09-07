#!/bin/bash
#SBATCH --job-name=fortress_unpack
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
# FORTRESS auspacken. Der tar enthaelt zwei grosse Zips; das Orthomosaik-Zip
# wird zuerst aus dem tar geholt und danach entpackt.

set -uo pipefail
D=${D:-/scratch/shared/${USER}/data/fortress}
TAR=${TAR:-/scratch/shared/${USER}/data/10.35097-538.tar}
cd "$D" || exit 1

echo "=== Knoten: $(hostname), Job ${SLURM_JOB_ID:-local} ==="
df -h "$D" | tail -1

for teil in orthomosaic nDSM; do
    ziel="10.35097-538/data/dataset/${teil}.zip"
    if [ ! -f "$ziel" ]; then
        echo "--- $teil.zip aus dem tar holen ---"
        tar -xf "$TAR" -C "$D" "$ziel" || exit 1
    fi
    echo "--- $teil entpacken ---"
    mkdir -p "$teil"
    unzip -o -q "$ziel" -d "$teil" || exit 1
    find "$teil" -type f | head -5
    echo "  $(find "$teil" -type f | wc -l) Dateien, $(du -sh "$teil" | cut -f1)"
    rm -f "$ziel"      # Zip nach dem Entpacken loeschen, spart 40 GB
done

df -h "$D" | tail -1
echo "=== Exit-Status: $? ==="
