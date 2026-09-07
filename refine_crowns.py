"""Instanzen aus dem Bild, Korrektur aus der Tiefe -- in beide Richtungen.

SAM 3 liefert die Instanzen mit der hoechsten Abdeckung, setzt die Grenzen aber
allein nach Bildkanten. Die monokulare Tiefe kennt dafuer die Hoehenstruktur und
kann zwei Fehlerarten reparieren, die SAM aus dem Bild allein nicht sieht:

  TEILEN        Eine Instanz enthaelt zwei prominente Wipfel mit einer Kerbe
                dazwischen -> zwei Baeume wurden zusammengefasst. Getrennt wird am
                Sattel, per Watershed innerhalb der Instanz.
  VERSCHMELZEN  Zwei Nachbarinstanzen haben keinen Sattel zwischen sich und
                dieselbe Farbe -> ein Baum wurde zerschnitten.

Reihenfolge: erst teilen, dann verschmelzen. Falsch zusammengefasste Blobs werden
also zuerst aufgebrochen, danach die Bruchstuecke wieder korrekt gruppiert.

Entscheidend beim Teilen ist, dass die Prominenz **innerhalb der jeweiligen
Instanz** gemessen wird, nicht global: eine niedrige Krone hat einen kleineren
Hoehenumfang als eine hohe, und ein global gesetzter Schwellwert wuerde bei ihr
nie ausloesen.

Beispiel:
    python refine_crowns.py --segments results_sam3/multiskala --out results_refined
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.measure import label as cc_label, regionprops
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from infer_species import IMAGE_SUFFIXES, REPO_ROOT
from merge_crowns import merge_round
from segment_sam import mask_metrics, metrics_from_region
from segment_trees import build_pseudo_chm


def load_surface(folder: str, stem: str, args) -> np.ndarray | None:
    """Hoehenoberflaeche laden: gemessene Parallaxe oder geschaetzte Tiefe.

    Die Parallaxe ist bereits Hoehe ueber der angepassten Ebene und muss deshalb
    nicht invertiert werden -- anders als die Tiefe, bei der naeher an der Kamera
    hoeher bedeutet. Der grossskalige Trend wird in beiden Faellen abgezogen: die
    Homographie-Ebene trifft den Boden nur naeherungsweise.
    """
    if args.surface == "parallax":
        path = args.parallax_cache / f"{folder}__{stem}.npy"
        if not path.exists():
            return None
        raw = np.load(path).astype(np.float32)
        trend = cv2.GaussianBlur(raw, (0, 0), max(1.0, args.crown_px * args.detrend_factor))
        return raw - trend

    path = args.depth_cache / f"{folder}__{stem}.npy"
    if not path.exists():
        return None
    return build_pseudo_chm(np.load(path), args.crown_px, args.detrend_factor)


def split_instance(mask: np.ndarray, surface: np.ndarray, args) -> list[np.ndarray] | None:
    """Eine Instanz an inneren Saetteln teilen. None, wenn nichts zu teilen ist."""
    values = surface[mask]
    if values.size < 50:
        return None

    # Prominenz relativ zum Hoehenumfang DIESER Instanz.
    low, high = np.percentile(values, [5, 95])
    span = high - low
    if span <= 0:
        return None

    inner = np.where(mask, surface, surface.min())
    seeds = h_maxima(inner, span * args.split_prominence) & mask
    markers = cc_label(seeds)
    if markers.max() < 2:
        return None

    parts = watershed(-surface, markers, mask=mask)
    expected_area = np.pi * (args.crown_px / 2) ** 2
    min_part = expected_area * args.min_part_area_factor

    pieces = []
    for region in regionprops(parts):
        piece = parts == region.label
        metrics = mask_metrics(piece)
        if metrics is None or metrics["area_px"] < min_part:
            return None  # Ein zu kleines Bruchstueck -> Teilung verwerfen.
        if metrics["kompaktheit"] < args.min_compactness:
            return None
        pieces.append(piece)

    return pieces if len(pieces) >= 2 else None


def split_round(labels: np.ndarray, surface: np.ndarray, args) -> tuple[np.ndarray, int]:
    """Alle Instanzen einmal auf Teilbarkeit pruefen."""
    result = np.zeros_like(labels)
    next_label, splits = 1, 0

    for region in regionprops(labels):
        # Nur im Bounding-Box-Ausschnitt arbeiten -- sonst kostet jede Instanz
        # eine Vollbildmaske.
        window = region.slice
        mask = region.image
        pieces = split_instance(mask, surface[window], args)

        target = result[window]
        if pieces is None:
            target[mask] = next_label
            next_label += 1
            continue

        splits += 1
        for piece in pieces:
            target[piece] = next_label
            next_label += 1

    return result, splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--segments", type=Path, default=REPO_ROOT / "results_sam3" / "multiskala")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_refined")
    parser.add_argument("--depth-cache", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/depth_cache"))
    parser.add_argument("--parallax-cache", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/parallax_cache"))
    parser.add_argument("--surface", choices=("depth", "parallax"), default="depth",
                        help="depth: monokular geschaetzt. parallax: aus Framepaaren gemessen.")

    parser.add_argument("--split-prominence", type=float, default=0.35,
                        help="Wie tief der Sattel zwischen zwei Wipfeln sein muss, als Anteil des "
                             "Hoehenumfangs der jeweiligen Instanz.")
    parser.add_argument("--min-part-area-factor", type=float, default=0.20,
                        help="Beide Teile muessen mindestens so gross sein wie dieser Anteil einer "
                             "erwarteten Krone -- sonst wird die Teilung verworfen.")
    parser.add_argument("--min-compactness", type=float, default=0.25)

    parser.add_argument("--split-threshold", type=float, default=0.10, help="Schwelle fuers Verschmelzen.")
    parser.add_argument("--color-threshold", type=float, default=16.0)
    parser.add_argument("--max-area-factor", type=float, default=4.0)
    parser.add_argument("--crown-px", type=float, default=100.0)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--detrend-factor", type=float, default=3.0)
    parser.add_argument("--no-merge", action="store_true", help="Nur teilen, nicht verschmelzen.")
    parser.add_argument("--no-split", action="store_true", help="Nur verschmelzen, nicht teilen.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows, totals = [], {"vorher": 0, "geteilt": 0, "verschmolzen": 0, "nachher": 0}
    for label_path in sorted(args.segments.glob("*/*_labels.png")):
        folder = label_path.parent.name
        stem = label_path.name.replace("_labels.png", "")
        originals = [p for p in (args.input / folder).glob(f"{stem}.*") if p.suffix.lower() in IMAGE_SUFFIXES]
        if not originals:
            print(f"  uebersprungen: {folder}/{stem}")
            continue

        chm = load_surface(folder, stem, args)
        if chm is None:
            print(f"  {folder}/{stem}: keine {args.surface}-Karte, uebersprungen")
            continue

        labels = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED).astype(np.int32)
        image_bgr = cv2.imread(str(originals[0]))
        lab_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)

        surface = cv2.GaussianBlur(chm, (0, 0), max(1.0, args.crown_px * 0.06))

        before = int(labels.max())
        splits = merges = 0

        for _ in range(args.rounds):
            if not args.no_split:
                labels, n = split_round(labels, surface, args)
                splits += n
            if not args.no_merge:
                labels, m = merge_round(labels, surface, lab_image, args)
                merges += m
            if (args.no_split or n == 0) and (args.no_merge or m == 0):
                break

        out_folder = args.out / folder
        out_folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_folder / f"{stem}_labels.png"), labels.astype(np.uint16))

        after = int(labels.max())
        totals["vorher"] += before
        totals["geteilt"] += splits
        totals["verschmolzen"] += merges
        totals["nachher"] += after

        metrics = [m for m in (metrics_from_region(r) for r in regionprops(labels)) if m]
        frame = pd.DataFrame(metrics)
        if len(frame):
            frame.insert(0, "id", np.arange(1, len(frame) + 1))
            frame.insert(0, "frame", originals[0].name)
            frame.insert(0, "folder", folder)
            frame["abdeckung"] = float((labels > 0).mean())
            rows.append(frame)

        print(f"  {folder}/{stem}: {before} -> {after} | {splits} geteilt, {merges} verschmolzen | "
              f"Abdeckung {float((labels > 0).mean()):.0%}")

    if not rows:
        print("Nichts verarbeitet.")
        return

    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(args.out / "all_crowns_refined.csv", index=False)

    print(f"\nGesamt: {totals['vorher']} -> {totals['nachher']} Instanzen "
          f"({totals['geteilt']} Teilungen, {totals['verschmolzen']} Verschmelzungen)")
    print(
        combined.groupby("folder")
        .agg(kronen=("id", "count"), durchmesser_px=("durchmesser_px", "median"),
             kompaktheit=("kompaktheit", "median"), abdeckung=("abdeckung", "mean"))
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
