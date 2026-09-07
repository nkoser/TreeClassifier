"""Instances from the image, corrections from the depth -- in both directions.

SAM 3 delivers the instances with the highest coverage, but sets the boundaries
from image edges alone. The monocular depth, on the other hand, knows the height
structure and can repair two kinds of error SAM cannot see from the image alone:

  SPLIT         One instance contains two prominent treetops with a notch
                between them -> two trees were merged. They are separated at the
                saddle, by a watershed inside the instance.
  MERGE         Two neighbouring instances have no saddle between them and the
                same colour -> one tree was cut apart.

Order: split first, then merge. Wrongly merged blobs are broken up first, and
the fragments are then grouped correctly again.

What matters when splitting is that the prominence is measured **within the
respective instance**, not globally: a low crown has a smaller height range than
a tall one, and a globally set threshold would never trigger on it.

Example:
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
    """Load the height surface: measured parallax or estimated depth.

    The parallax is already a height above the fitted plane and therefore needs no
    inversion -- unlike depth, where closer to the camera means higher. The
    large-scale trend is subtracted in both cases: the homography plane only
    approximates the ground.
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
    """Split one instance at internal saddles. None if there is nothing to split."""
    values = surface[mask]
    if values.size < 50:
        return None

    # Prominence relative to the height range of THIS instance.
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
            return None  # a fragment that is too small -> discard the split
        if metrics["kompaktheit"] < args.min_compactness:
            return None
        pieces.append(piece)

    return pieces if len(pieces) >= 2 else None


def split_round(labels: np.ndarray, surface: np.ndarray, args) -> tuple[np.ndarray, int]:
    """Check every instance once for splittability."""
    result = np.zeros_like(labels)
    next_label, splits = 1, 0

    for region in regionprops(labels):
        # Work only within the bounding-box crop -- otherwise every instance costs
        # a full-frame mask.
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
                        help="depth: estimated monocularly. parallax: measured from frame pairs.")

    parser.add_argument("--split-prominence", type=float, default=0.35,
                        help="How deep the saddle between two treetops has to be, as a fraction "
                             "of the height range of the respective instance.")
    parser.add_argument("--min-part-area-factor", type=float, default=0.20,
                        help="Both parts have to be at least this fraction of an expected crown -- "
                             "otherwise the split is discarded.")
    parser.add_argument("--min-compactness", type=float, default=0.25)

    parser.add_argument("--split-threshold", type=float, default=0.10, help="Threshold for merging.")
    parser.add_argument("--color-threshold", type=float, default=16.0)
    parser.add_argument("--max-area-factor", type=float, default=4.0)
    parser.add_argument("--crown-px", type=float, default=100.0)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--detrend-factor", type=float, default=3.0)
    parser.add_argument("--no-merge", action="store_true", help="Only split, do not merge.")
    parser.add_argument("--no-split", action="store_true", help="Only merge, do not split.")
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
