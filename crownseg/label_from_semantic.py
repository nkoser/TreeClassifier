"""Label crowns from a semantic species map.

FORTRESS (Schiefer, Frey & Kattenborn 2022, CC BY 4.0) supplies pixel-level
species masks for the southern Black Forest at under 1.35 cm ground sampling --
9 species, 3 genera, deadwood and forest floor. What it does not supply is
individual trees: the labels say *which species* stands at a place, not *which
tree*. That is exactly what our crown segmentation adds. Mask plus species map
gives labelled crown crops for Central European species, without anyone marking
a single tree by hand -- the bottleneck this project has been stuck on from the
start.

The assignment is a majority decision within the mask. Two numbers are recorded
per crown, because they decide usability:

  abdeckung   Share of the mask that carries a species class at all. Low values
              mean the crown lies mostly on forest floor or outside the mapped
              area.
  reinheit    Share of the majority species in the occupied area. Low values
              mean the mask covers several species -- either a mis-segmentation
              or a crown margin.

Crowns below the thresholds are discarded. A sloppily labelled training crop is
more harmful than a missing one.

    python crownseg/label_from_semantic.py --labels ... --semantic ... --out ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classify import instances_from_labels  # noqa: E402


def assign(labels: np.ndarray, semantic: np.ndarray, background: set[int],
           min_coverage: float, min_purity: float, min_area: int) -> pd.DataFrame:
    """Assign every crown instance the majority class of its area."""
    rows = []
    for instance in instances_from_labels(labels, min_area):
        x0, y0, x1, y1 = instance["x0"], instance["y0"], instance["x1"], instance["y1"]
        mask = labels[y0:y1, x0:x1] == instance["instanz"]
        werte = semantic[y0:y1, x0:x1][mask]
        belegt = werte[~np.isin(werte, list(background))]
        if not len(belegt):
            continue

        klassen, anzahl = np.unique(belegt, return_counts=True)
        mehrheit = int(klassen[anzahl.argmax()])
        abdeckung = len(belegt) / max(1, mask.sum())
        reinheit = anzahl.max() / len(belegt)
        rows.append({**instance, "klasse": mehrheit,
                     "abdeckung": float(abdeckung), "reinheit": float(reinheit),
                     "brauchbar": bool(abdeckung >= min_coverage and reinheit >= min_purity)})
    return pd.DataFrame(rows)


def main() -> None:
    import cv2

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, required=True, help="Crown label map (uint16).")
    parser.add_argument("--semantic", type=Path, required=True, help="Species map, same geometry.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--background", type=int, nargs="*", default=[0],
                        help="Class values that are not a species (background, forest floor).")
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--min-purity", type=float, default=0.7)
    parser.add_argument("--min-area", type=int, default=200)
    args = parser.parse_args()

    labels = cv2.imread(str(args.labels), cv2.IMREAD_UNCHANGED)
    semantic = cv2.imread(str(args.semantic), cv2.IMREAD_UNCHANGED)
    if semantic.ndim == 3:
        semantic = semantic[:, :, 0]
    if semantic.shape != labels.shape:
        semantic = cv2.resize(semantic, labels.shape[::-1], interpolation=cv2.INTER_NEAREST)

    frame = assign(labels, semantic, set(args.background),
                   args.min_coverage, args.min_purity, args.min_area)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    gut = frame[frame["brauchbar"]]
    print(f"{len(frame)} Kronen | brauchbar {len(gut)} ({len(gut)/max(1,len(frame)):.0%})")
    print(f"  Abdeckung Median {frame['abdeckung'].median():.2f} | "
          f"Reinheit Median {frame['reinheit'].median():.2f}")
    print(f"\nKlassenverteilung der brauchbaren:")
    print(gut["klasse"].value_counts().to_string())


if __name__ == "__main__":
    main()
