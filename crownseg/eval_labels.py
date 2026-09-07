"""Beliebige Labelkarten gegen die BAMFORESTS-Wahrheit messen.

Damit sind die Verfahren untereinander vergleichbar: SAM 3 mit Textprompt, die
Tiefen-Prompt-Kombination, der Hybrid, `crownnet.py` und Mask R-CNN liefern alle
`<stem>_labels.png`. Dieses Skript liest so einen Ordner, holt die passende
Wahrheit aus dem aufbereiteten Split und rechnet dieselben Zahlen wie
`maskrcnn.py --mode eval`.

Die Labelkarten duerfen kleiner sein als die 2048er Kachel (SAM 3 lief auf
0.7-fach verkleinerten Bildern) -- die Wahrheit wird dann mitskaliert statt die
Vorhersage hochzurechnen, damit keine Treppen in die Masken kommen.

AP ist hier ohne Aussage: eine Labelkarte hat keine Konfidenz je Instanz, die
Rangfolge ist also willkuerlich. Aussagekraeftig sind Praezision, Trefferquote,
F1 und die mittlere IoU der Treffer.

    python crownseg/eval_labels.py --labels results_sam3_bam/test1 --split test1
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bamforests as bam  # noqa: E402
import metrics as met  # noqa: E402


def instances_from_label_map(labels: np.ndarray, min_area: int) -> list[met.Instance]:
    instances = []
    for value in np.unique(labels):
        if value == 0:
            continue
        instance = met.instance_from_mask(labels == value)
        if instance is not None and instance.area >= min_area:
            instances.append(instance)
    return instances


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, required=True, help="Ordner mit <stem>_labels.png")
    parser.add_argument("--split", default="test1", choices=("train", "val", "test1", "test2"))
    parser.add_argument("--prepared", type=Path, default=bam.BAMFORESTS / "crownseg")
    parser.add_argument("--name", default=None, help="Bezeichnung des Verfahrens in der Ausgabe.")
    parser.add_argument("--min-area", type=int, default=400, help="Auf 2048er Massstab bezogen.")
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    args = parser.parse_args()

    directory = args.prepared / args.split
    index = json.loads((directory / "annotations.json").read_text())
    maps = sorted(args.labels.glob("*_labels.png"))
    if not maps:
        print(f"Keine Labelkarten in {args.labels}")
        return

    per_area = collections.defaultdict(list)
    skipped = []
    for path in maps:
        stem = path.name[: -len("_labels.png")]
        if stem not in index:
            skipped.append(stem)
            continue
        labels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        size = labels.shape[0]
        scale = size / 2048.0

        rings = [np.asarray(p, np.float32).reshape(-1, 2) * scale for p in index[stem]]
        truth_masks, _ = bam.masks_from_rings(rings, size, size, int(args.min_area * scale**2), 0.0)
        truth = [i for i in (met.instance_from_mask(m) for m in truth_masks) if i is not None]
        predicted = instances_from_label_map(labels, int(args.min_area * scale**2))
        per_area[stem.split("_")[0]].append(met.evaluate(predicted, truth, args.iou_thresh))

    name = args.name or args.labels.name
    print(f"=== {name} auf {args.split} (IoU >= {args.iou_thresh}) ===")
    if skipped:
        print(f"{len(skipped)} Karten ohne Wahrheit uebersprungen")
    for area, rows in sorted(per_area.items()):
        summary = met.accumulate(rows)
        print(f"  {area:12s} Kacheln {summary['kacheln']:3d}  GT {summary['kronen_gt']:5d}  "
              f"Vorhersagen {summary['kronen_pred']:5d}  Praez {summary['praezision']:.3f}  "
              f"Treffer {summary['trefferquote']:.3f}  F1 {summary['f1']:.3f}  "
              f"IoU {summary['mittlere_iou']:.3f}")


if __name__ == "__main__":
    main()
