"""Kronen aus einer semantischen Artkarte beschriften.

FORTRESS (Schiefer, Frey & Kattenborn 2022, CC BY 4.0) liefert Artmasken auf
Pixelebene fuer den Suedschwarzwald bei unter 1.35 cm Bodenaufloesung -- 9 Arten,
3 Gattungen, Totholz und Waldboden. Was es nicht liefert, sind einzelne Baeume:
die Labels sagen, *welche* Art an einer Stelle steht, nicht *welcher* Baum.

Genau das ergaenzt unsere Kronensegmentierung. Maske plus Artkarte ergibt
beschriftete Kronenausschnitte fuer mitteleuropaeische Arten, ohne dass jemand
einen einzelnen Baum von Hand markiert -- der Engpass, an dem dieses Projekt
seit Beginn haengt.

Die Zuordnung ist eine Mehrheitsentscheidung innerhalb der Maske. Zwei Zahlen
werden je Krone mitgeschrieben, weil sie ueber die Brauchbarkeit entscheiden:

  abdeckung   Anteil der Maske, der ueberhaupt eine Artklasse traegt. Niedrige
              Werte heissen, die Krone liegt groesstenteils auf Waldboden oder
              ausserhalb des kartierten Bereichs.
  reinheit    Anteil der Mehrheitsart an der belegten Flaeche. Niedrige Werte
              heissen, die Maske ueberdeckt mehrere Arten -- entweder eine
              Falschsegmentierung oder ein Kronenrand.

Kronen unter den Schwellen werden verworfen. Ein unsauber beschrifteter
Trainingsausschnitt ist schaedlicher als ein fehlender.

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
    """Jeder Kroneninstanz die Mehrheitsklasse ihrer Flaeche zuordnen."""
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
    parser.add_argument("--labels", type=Path, required=True, help="Kronen-Labelkarte (uint16).")
    parser.add_argument("--semantic", type=Path, required=True, help="Artkarte, gleiche Geometrie.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--background", type=int, nargs="*", default=[0],
                        help="Klassenwerte, die keine Art sind (Hintergrund, Waldboden).")
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
