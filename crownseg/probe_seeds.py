"""Vorfrage: taugen die Wipfel der Tiefenkarte als zusaetzliche Prompts?

Der Ergaenzungsschritt aus `sam3_depth.py` hat auf test1 nichts getroffen (0 von
291 Kronen bei IoU 0.5). Daraus folgt aber *nicht*, dass die Wipfelpositionen
schlecht sind -- dort kam die Form aus dem Watershed, und nur die Form wurde
gemessen. Ein Punkt-Prompt an SAM 3 nimmt aus der Tiefe nur das Wo und laesst die
Form beim Bildmodell.

Dieses Skript misst deshalb die Obergrenze, bevor irgendetwas gebaut wird:

  verpasste Kronen        GT-Kronen ohne Treffer in der SAM-3-Vorhersage.
  freie Wipfel            Wipfel, die in keiner vorhergesagten Krone liegen.
  davon in einer          Wie viele freie Wipfel liegen in einer verpassten
  verpassten Krone        Krone? Nur diese koennten ueberhaupt etwas beitragen.
  erreichbare Kronen      Wie viele verpasste Kronen enthalten mindestens einen
                          freien Wipfel? Das ist die Trefferquote, die ein
                          perfekter Prompt zusaetzlich holen koennte.

Liegt die erreichbare Quote niedrig, ist die Idee unabhaengig von der
Maskenqualitaet erledigt und muss nicht implementiert werden.

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
    """Dieselben Werte wie im Lauf, der die Labelkarten erzeugt hat."""

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
                        help="Mehrere Werte durchprobieren, um den Engpass der Wipfelsuche zu finden.")
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

        # Welche GT-Kronen hat SAM 3 verpasst?
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
        # Obergrenze: jeder treffende freie Wipfel wird eine perfekte Maske,
        # jeder nicht treffende ein Fehlalarm.
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
