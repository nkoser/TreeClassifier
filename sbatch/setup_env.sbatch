#!/bin/bash
#SBATCH --job-name=treeclf_setup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/shared/%u/runs/logs/%x_%j.out
#SBATCH --error=/scratch/shared/%u/runs/logs/%x_%j.err
#
# Einmaliges Setup, aufgesetzt auf den vorhandenen vesselgpt_7.sif.
#
# Der Container bringt Python 3.11.8 und torch 2.10.0+cu128 mit -- passend fuer
# die RTX PRO 6000 Blackwell (sm_120, Treiber 580.x). Statt eine zweite
# Torch-Installation anzulegen, wird ein venv MIT --system-site-packages auf
# /scratch erzeugt: es erbt Torch aus dem Container und ergaenzt nur, was fehlt
# (deepforest, opencv, omegaconf und deren Abhaengigkeiten).
#
# Das venv ist nur INNERHALB des Containers benutzbar -- sein Interpreter zeigt
# auf /miniconda3, das es ausserhalb nicht gibt. Alle Laeufe gehen deshalb durch
# apptainer.
#
# Submit: sbatch sbatch/setup_env.sbatch

set -uo pipefail

SIF=${SIF:-/scratch/shared/${USER}/containers/vesselgpt_7.sif}
REPO=${REPO:-/home/nik/workspace/TreeClassifier}
ENV_DIR=${ENV_DIR:-/scratch/shared/${USER}/envs/treeclf}
CKPT_DIR=${CKPT_DIR:-/scratch/shared/${USER}/data/treeclf/checkpoints}
HF_HOME=${HF_HOME:-/scratch/shared/${USER}/hf_cache}
BASE_PY=${BASE_PY:-/miniconda3/envs/vmtk_2/bin/python}
CKPT_URL="https://huggingface.co/jannikend/dinovtree/resolve/main/checkpoints/dinovtreeb_quebectrees.pth"

mkdir -p "$(dirname "$ENV_DIR")" "$CKPT_DIR" "$HF_HOME"

inc () {
    apptainer exec --nv \
        --bind "$REPO":"$REPO" \
        --bind /scratch:/scratch \
        --bind /cold:/cold \
        --pwd "$REPO" \
        --env HF_HOME="$HF_HOME",TORCH_HOME="$HF_HOME/torch" \
        "$SIF" "$@"
}

echo "=== Knoten: $(hostname) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

echo "=== Basis-Interpreter im Container ==="
inc "$BASE_PY" -c "import sys, torch; print(sys.version.split()[0], '| torch', torch.__version__)" || exit 1

if [ ! -x "$ENV_DIR/bin/python" ]; then
    echo "=== venv anlegen (erbt Container-site-packages) ==="
    inc "$BASE_PY" -m venv --system-site-packages "$ENV_DIR" || exit 1
fi

echo "=== Fehlende Pakete nachinstallieren ==="
# Torch/torchvision kommen aus dem Container und duerfen NICHT ueberschrieben
# werden -- sonst landet ein PyPI-Default-Build ohne sm_120 im venv.
inc "$ENV_DIR/bin/pip" install --no-cache-dir \
    deepforest omegaconf opencv-python-headless || exit 1

inc "$ENV_DIR/bin/pip" list 2>/dev/null | grep -iE "^(torch|torchvision|deepforest|opencv|omegaconf|numpy|pandas) "

echo "=== Checkpoint ==="
if [ ! -s "$CKPT_DIR/dinovtreeb_quebectrees.pth" ]; then
    if [ -s "$REPO/checkpoints/dinovtreeb_quebectrees.pth" ]; then
        mv "$REPO/checkpoints/dinovtreeb_quebectrees.pth" "$CKPT_DIR/"
    else
        wget -q -O "$CKPT_DIR/dinovtreeb_quebectrees.pth" "${CKPT_URL}?download=true" || exit 1
    fi
fi
ls -lh "$CKPT_DIR"

echo "=== Verifikation: echter Forward-Pass auf der GPU ==="
inc "$ENV_DIR/bin/python" - "$CKPT_DIR/dinovtreeb_quebectrees.pth" <<'PY' || exit 1
import sys

import torch

from infer_species import build_dinovtree, load_class_names, REPO_ROOT

print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("FEHLER: keine GPU sichtbar")
print("GPU:", torch.cuda.get_device_name(0), "| Capability:", torch.cuda.get_device_capability(0))

names = load_class_names(REPO_ROOT / "third_party" / "quebec_trees_categories.json", (2, 3, 10))
model = build_dinovtree(sys.argv[1], n_classes=len(names), max_height=30.0, device="cuda")
with torch.no_grad():
    logits, height = model(torch.rand(2, 3, 512, 512, device="cuda"), {})
print("Logits:", tuple(logits.shape), "| Hoehen:", tuple(height.shape))
print("Smoke-Test ok:", len(names), "Klassen")
PY

echo "=== Setup abgeschlossen ==="
