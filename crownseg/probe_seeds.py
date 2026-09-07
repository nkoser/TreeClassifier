"""Preliminary question: are the treetops of the depth map usable as extra prompts?

The completion step in `sam3_depth.py` hit nothing on test1 (0 of 291 crowns at
IoU 0.5). That does *not* imply the treetop positions are bad -- there the shape
came from the watershed, and only the shape was measured. A point prompt to
SAM 3 takes only the where from the depth and leaves the shape to the image
model.

So this script measures the ceiling before anything is built:

  missed crowns          GT crowns with no hit in the SAM 3 prediction.
  free treetops          Treetops lying in no predicted crown.
  of those, in a         How many free treetops lie in a missed crown? Only
  missed crown           those could contribute anything at all.
  reachable crowns       How many missed crowns contain at least one free
                         treetop? That is the recall a perfect prompt could
                         additionally pick up.

If the reachable rate is low, the idea is settled regardless of mask quality and
does not have to be implemented.

    python crownseg/probe_seeds.py --labels results_sam3depth_pro_split/test1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bamforests as bam  # noqa: E402
import metrics as met  # noqa: E402
from sam3_depth import treetops  # noqa: E402
from segment_trees import build_pseudo_chm  # noqa: E402


class Params:
    """The same values as in the run that produced the label maps."""

    crown_px = 275.0
    smooth_factor = 0.06
    gap_percentile = 15.0
    peak_prominence = 0.10
    detrend_factor = 3.0

    def __init__(self, prominence: float, gap: float) -> None:
        self.peak_prominence = prominence
        self.gap_percentile = gap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=Path("results_sam3depth_pro_split/test1"))
    parser.add_argument("--split", default="test1")
    parser.add_argument("--prepared", type=Path, default=bam.BAMFORESTS / "crownseg")
    parser.add_argument("--depth-cache", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/depth_cache/depthpro"))
    parser.add_argument("--min-area", type=int, default=400)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--prominence", type=float, nargs="*", default=[0.10],
                        help="Try several values to locate the bottleneck of the treetop search.")
    parser.add_argument("--gap-percentile", type=float, default=15.0)
    args = parser.parse_args()

    index = json.loads((args.prepared / args.split / "annotations.json").read_text())
    totals = {value: dict(gt=0, verpasst=0, wipfel=0, frei=0, treffend=0, erreichbar=0)
              for value in args.prominence}

    for path in sorted(args.labels.glob("*_labels.png")):
        stem = path.name[: -len("_labels.png")]
        if stem not in index:
            continue
        depth_path = args.depth_cache / f"{args.split}_{stem}.npy"
        if not depth_path.exists():
            continue

        labels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        size = labels.shape[0]
        scale = size / 2048.0
        floor = int(args.min_area * scale**2)

        rings = [np.asarray(p, np.float32).reshape(-1, 2) * scale for p in index[stem]]
        truth_masks, _ = bam.masks_from_rings(rings, size, size, floor, 0.0)
        truth = [i for i in (met.instance_from_mask(m) for m in truth_masks) if i is not None]
        predicted = [i for i in (met.instance_from_mask(labels == v)
                                 for v in np.unique(labels) if v > 0) if i is not None]
        predicted = [i for i in predicted if i.area >= floor]

        # Which GT crowns did SAM 3 miss?
        taken = set()
        for prediction in predicted:
            best, best_iou = -1, args.iou_thresh
            for j, gt in enumerate(truth):
                if j in taken:
                    continue
                value = met.iou(prediction, gt)
                if value >= best_iou:
                    best, best_iou = j, value
            if best >= 0:
                taken.add(best)
        missed = [gt for j, gt in enumerate(truth) if j not in taken]

        chm = build_pseudo_chm(np.load(depth_path), Params.crown_px, Params.detrend_factor)
        covered = labels > 0
        missed_map = np.zeros(labels.shape, np.int32)
        for number, gt in enumerate(missed, start=1):
            x0, y0, x1, y1 = gt.box
            view = missed_map[y0:y1, x0:x1]
            view[gt.mask] = number

        for value in args.prominence:
            markers, _, _ = treetops(chm, Params(value, args.gap_percentile))

            hit_crowns = set()
            free = touching = 0
            for region_id in np.unique(markers):
                if region_id == 0:
                    continue
                ys, xs = np.nonzero(markers == region_id)
                y, x = int(ys.mean()), int(xs.mean())
                if covered[y, x]:
                    continue
                free += 1
                if missed_map[y, x] > 0:
                    touching += 1
                    hit_crowns.add(int(missed_map[y, x]))

            bucket = totals[value]
            bucket["gt"] += len(truth)
            bucket["verpasst"] += len(missed)
            bucket["wipfel"] += int(markers.max())
            bucket["frei"] += free
            bucket["treffend"] += touching
            bucket["erreichbar"] += len(hit_crowns)

    print(f"=== Wipfel als zusaetzliche Prompts, Obergrenze auf {args.split} ===")
    print(f"{'Promin':>7} {'Wipfel':>7} {'frei':>6} {'in verpasster':>14} {'Quote':>7} "
          f"{'erreichbar':>11} {'Treffer max':>12} {'F1 max':>7}")
    for value in args.prominence:
        t = totals[value]
        hits = t["gt"] - t["verpasst"]
        ceiling_hits = hits + t["erreichbar"]
        # Ceiling: every hitting free treetop becomes a perfect mask, every
        # non-hitting one a false alarm.
        predicted = t["gt"] - t["verpasst"] + t["frei"]
        precision = ceiling_hits / max(1, predicted)
        recall = ceiling_hits / max(1, t["gt"])
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        print(f"{value:7.3f} {t['wipfel']:7d} {t['frei']:6d} {t['treffend']:14d} "
              f"{t['treffend'] / max(1, t['frei']):6.1%} {t['erreichbar']:11d} "
              f"{recall:12.3f} {f1:7.3f}")
    print("\nBezug: SAM 3 allein erreicht auf diesen Kacheln Trefferquote 0.296, F1 0.342.")


if __name__ == "__main__":
    main()
