"""Nicht-Kronen aus einer Labelkarte entfernen: Wiese, Weg, Dach.

BAMFORESTS ist reiner Wald und kennt nur die Klasse `tree`. Alles andere ist dort
unmarkierter Hintergrund, kein Gegenbeispiel -- ein darauf trainiertes Modell hat
nie gelernt, dass Rasen oder Asphalt *kein* Baum ist, weil es nie eines gesehen
hat. Auf den urbanen Frames segmentiert EoMT deshalb Parkwiesen und Wegraender.

Zwei Eigenschaften trennen eine Krone davon, und beide kommen ohne zusaetzliches
Training aus:

  textur  Ein Kronendach ist hochfrequent -- Zweige, Blattgruppen, Schattenwurf.
          Eine Wiese und eine Asphaltflaeche sind glatt. Gemessen als mittlerer
          Betrag des Laplace-Operators innerhalb der Maske, bezogen auf den
          Bildmedian, damit die Zahl nicht an der Belichtung haengt.
  relief  Eine Krone ragt ueber ihre Umgebung, eine Wiese nicht. Gemessen als
          Hoehenunterschied zwischen der Maske und einem Ring um sie herum, im
          Ersatz-CHM aus der Tiefenschaetzung. Genau die Rolle, in der ein
          Hoehenmodell etwas taugt -- als Verwerfer, nicht als Detektor.

Beides sind Filter auf fertigen Instanzen, sie koennen also nur wegnehmen. Der
Nutzen muss sich entsprechend als hoehere Praezision bei gleicher Trefferquote
zeigen, sonst ist es keiner.

    python crownseg/reject.py --labels results_frames_eomt_final --frames-dir /cold/...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics as met  # noqa: E402


def texture_map(image_bgr: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return np.abs(cv2.Laplacian(grey, cv2.CV_32F, ksize=3))


def instance_features(labels: np.ndarray, texture: np.ndarray,
                      chm: np.ndarray | None, ring_px: int) -> dict[int, dict]:
    """Textur und Relief je Instanz. Das Relief braucht einen Ring aussen herum."""
    reference = float(np.median(texture)) + 1e-6
    features = {}
    kernel = np.ones((ring_px, ring_px), np.uint8)

    for value in np.unique(labels):
        if value == 0:
            continue
        mask = labels == value
        entry = {"textur": float(texture[mask].mean() / reference)}
        if chm is not None:
            # Ring ausserhalb der Maske, aber ohne andere Kronen -- sonst misst
            # man den Hoehenunterschied zum Nachbarbaum statt zum Boden.
            grown = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
            ring = grown & ~mask & (labels == 0)
            entry["relief"] = float(chm[mask].mean() - chm[ring].mean()) if ring.sum() > 20 else 0.0
        features[int(value)] = entry
    return features


def apply(labels: np.ndarray, features: dict[int, dict],
          min_texture: float, min_relief: float | None) -> tuple[np.ndarray, int]:
    kept = labels.copy()
    dropped = 0
    for value, entry in features.items():
        fails = entry["textur"] < min_texture
        if min_relief is not None and "relief" in entry:
            fails = fails or entry["relief"] < min_relief
        if fails:
            kept[labels == value] = 0
            dropped += 1
    return kept, dropped


def load_chm(path: Path | None, shape: tuple[int, int], crown_px: float) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    raw = np.load(path).astype(np.float32) if path.suffix == ".npy" else \
        cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    surface = -raw if path.suffix == ".npy" else raw  # .npy ist Tiefe, .png bereits Hoehe
    if surface.shape != shape:
        surface = cv2.resize(surface, shape[::-1], interpolation=cv2.INTER_LINEAR)
    trend = cv2.GaussianBlur(surface, (0, 0), max(1.0, crown_px * 3.0))
    relief = surface - trend
    span = np.percentile(relief, 99) - np.percentile(relief, 1)
    return relief / max(1e-6, span)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, required=True, help="Ordner mit <ordner>/<stem>_labels.png")
    parser.add_argument("--images", type=Path, required=True, help="Wurzel der zugehoerigen Bilder.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--depth-cache", type=Path, default=None,
                        help="Ordner mit <ordner>__<stem>.npy; ohne das laeuft nur der Texturfilter.")
    parser.add_argument("--min-texture", type=float, default=0.9)
    parser.add_argument("--min-relief", type=float, default=None)
    parser.add_argument("--ring-px", type=int, default=25)
    parser.add_argument("--crown-px", type=float, default=100.0)
    parser.add_argument("--report-only", action="store_true", help="Nur Kennzahlen ausgeben, nichts schreiben.")
    args = parser.parse_args()

    maps = sorted(args.labels.rglob("*_labels.png"))
    total = kept_total = 0
    for path in maps:
        relative = path.relative_to(args.labels)
        stem = path.name[: -len("_labels.png")]
        matches = list((args.images / relative.parent).glob(f"{stem}.*"))
        image = cv2.imread(str(matches[0])) if matches else None
        if image is None:
            continue
        labels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if labels.shape[:2] != image.shape[:2]:
            labels = cv2.resize(labels, image.shape[1::-1], interpolation=cv2.INTER_NEAREST)

        chm = load_chm(args.depth_cache / f"{relative.parent.name}__{stem}.npy"
                       if args.depth_cache else None, labels.shape, args.crown_px)
        features = instance_features(labels, texture_map(image), chm, args.ring_px)
        cleaned, dropped = apply(labels, features, args.min_texture, args.min_relief)

        total += len(features)
        kept_total += len(features) - dropped
        textures = np.array([f["textur"] for f in features.values()])
        reliefs = np.array([f.get("relief", np.nan) for f in features.values()])
        quantiles = np.percentile(textures, [5, 25, 50]) if len(textures) else [0, 0, 0]
        line = (f"  {relative.parent.name}/{stem}: {len(features)} -> {len(features) - dropped} "
                f"| Textur p5/p25/med {quantiles[0]:.2f}/{quantiles[1]:.2f}/{quantiles[2]:.2f}")
        if np.isfinite(reliefs).any():
            rq = np.nanpercentile(reliefs, [5, 25, 50])
            line += f" | Relief p5/p25/med {rq[0]:+.3f}/{rq[1]:+.3f}/{rq[2]:+.3f}"
        else:
            line += " | Relief fehlt"
        print(line, flush=True)

        if not args.report_only:
            target = args.out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), cleaned)

    print(f"\n{kept_total} von {total} Instanzen behalten ({kept_total / max(1, total):.0%})")


if __name__ == "__main__":
    main()
