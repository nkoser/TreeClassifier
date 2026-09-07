"""Measure any label maps against the BAMFORESTS ground truth.

That makes the methods comparable with each other: SAM 3 with a text prompt, the
depth-prompt combination, the hybrid, `crownnet.py` and Mask R-CNN all produce
`<stem>_labels.png`. This script reads such a folder, fetches the matching truth
from the prepared split and computes the same numbers as
`maskrcnn.py --mode eval`.

The label maps may be smaller than the 2048 tile (SAM 3 ran on images scaled
down by 0.7) -- the truth is then scaled along instead of upscaling the
prediction, so that no staircase artefacts enter the masks.

AP says nothing here: a label map has no per-instance confidence, so the ranking
is arbitrary. What is meaningful is precision, recall, F1 and the mean IoU of
the hits.

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
    parser.add_argument("--labels", type=Path, required=True, help="Folder holding <stem>_labels.png")
    parser.add_argument("--split", default="test1", choices=("train", "val", "test1", "test2"))
    parser.add_argument("--prepared", type=Path, default=bam.BAMFORESTS / "crownseg")
    parser.add_argument("--name", default=None, help="Label for the method in the output.")
    parser.add_argument("--min-area", type=int, default=400, help="Relative to the 2048 scale.")
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
