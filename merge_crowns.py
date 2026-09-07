"""Merge wrongly separated crowns back together.

The segmentation often breaks a single tree into several polygons. This script
merges neighbouring instances when two independent criteria indicate that they
belong to the same tree:

  saddle prominence  Between the tops of two genuinely neighbouring trees there
                     is a notch. If the boundary instead runs across a continuous
                     dome, the saddle is flat -- a sign of a false separation.
  colour distance    Two parts of the same crown have almost the same colour. Two
                     different trees usually differ measurably. Computed in Lab
                     space, where distances match perception.

Both have to agree. That matters because the saddle prominence measures against
the estimated monocular depth -- the weakest point of the pipeline. The colour
distance is entirely independent of it and partly catches its errors.

An area ceiling prevents merges from escalating into giant blobs. Because every
merge changes treetops and saddles, the whole thing runs in several rounds.

Example:
    python merge_crowns.py --segments results_sam3/multiskala --out results_merged
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.measure import regionprops

from infer_species import IMAGE_SUFFIXES, REPO_ROOT
from segment_sam import mask_metrics, metrics_from_region
from segment_trees import build_pseudo_chm


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n + 1))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def neighbour_saddles(labels: np.ndarray, surface: np.ndarray) -> dict[tuple[int, int], float]:
    """Highest point of the shared boundary, per neighbouring pair."""
    saddles: dict[tuple[int, int], float] = {}
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = labels[max(0, -dy) : labels.shape[0] - max(0, dy), max(0, -dx) : labels.shape[1] - max(0, dx)]
        b = labels[max(0, dy) : labels.shape[0] - max(0, -dy), max(0, dx) : labels.shape[1] - max(0, -dx)]
        h = surface[max(0, -dy) : surface.shape[0] - max(0, dy), max(0, -dx) : surface.shape[1] - max(0, dx)]

        touching = (a != b) & (a > 0) & (b > 0)
        if not touching.any():
            continue
        for la, lb, height in zip(a[touching], b[touching], h[touching]):
            key = (int(min(la, lb)), int(max(la, lb)))
            if height > saddles.get(key, -np.inf):
                saddles[key] = float(height)
    return saddles


def merge_round(labels: np.ndarray, surface: np.ndarray, lab_image: np.ndarray, args) -> tuple[np.ndarray, int]:
    """One merge round. Returns (new labels, number of merges)."""
    regions = {r.label: r for r in regionprops(labels, intensity_image=surface)}
    if len(regions) < 2:
        return labels, 0

    peaks = {label: float(r.intensity_max) for label, r in regions.items()}
    areas = {label: int(r.area) for label, r in regions.items()}
    # Colour per instance from the bounding-box crop, not over the whole image.
    colors = {
        label: np.median(lab_image[r.slice][r.image], axis=0) for label, r in regions.items()
    }

    values = surface[labels > 0]
    span = max(1e-6, float(np.percentile(values, 95) - np.percentile(values, 5)))
    max_area = np.pi * (args.crown_px / 2) ** 2 * args.max_area_factor

    union = UnionFind(int(labels.max()))
    merged = 0
    # Ascending by prominence: the clearest false separations first.
    candidates = sorted(neighbour_saddles(labels, surface).items(), key=lambda kv: -kv[1])

    for (la, lb), saddle in candidates:
        if la not in peaks or lb not in peaks:
            continue
        prominence = (min(peaks[la], peaks[lb]) - saddle) / span
        if prominence >= args.split_threshold:
            continue

        delta_e = float(np.linalg.norm(colors[la] - colors[lb]))
        if delta_e >= args.color_threshold:
            continue

        ra, rb = union.find(la), union.find(lb)
        if ra == rb:
            continue
        if areas.get(ra, 0) + areas.get(rb, 0) > max_area:
            continue

        union.union(la, lb)
        root = union.find(la)
        combined = areas.pop(ra, 0) + areas.pop(rb, 0)
        areas[root] = combined
        merged += 1

    if merged == 0:
        return labels, 0

    lookup = np.zeros(labels.max() + 1, dtype=np.int32)
    for label in range(1, labels.max() + 1):
        lookup[label] = union.find(label)
    remapped = lookup[labels]

    # Renumber the labels contiguously again.
    unique = np.unique(remapped)
    unique = unique[unique > 0]
    renumber = np.zeros(remapped.max() + 1, dtype=np.int32)
    renumber[unique] = np.arange(1, len(unique) + 1)
    return renumber[remapped], merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--segments", type=Path, default=REPO_ROOT / "results_sam3" / "multiskala")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_merged")
    parser.add_argument("--depth-cache", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/depth_cache"))

    parser.add_argument("--split-threshold", type=float, default=0.06,
                        help="Saddle prominence below which two instances are merged.")
    parser.add_argument("--color-threshold", type=float, default=12.0,
                        help="Maximum Lab colour distance of two parts of the same crown.")
    parser.add_argument("--crown-px", type=float, default=100.0)
    parser.add_argument("--max-area-factor", type=float, default=4.0,
                        help="Ceiling on the merged area; prevents giant blobs.")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--detrend-factor", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for label_path in sorted(args.segments.glob("*/*_labels.png")):
        folder = label_path.parent.name
        stem = label_path.name.replace("_labels.png", "")
        originals = [p for p in (args.input / folder).glob(f"{stem}.*") if p.suffix.lower() in IMAGE_SUFFIXES]
        depth_path = args.depth_cache / f"{folder}__{stem}.npy"
        if not originals or not depth_path.exists():
            print(f"  uebersprungen: {folder}/{stem}")
            continue

        labels = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED).astype(np.int32)
        image_bgr = cv2.imread(str(originals[0]))
        lab_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)

        chm = build_pseudo_chm(np.load(depth_path), args.crown_px, args.detrend_factor)
        surface = cv2.GaussianBlur(chm, (0, 0), max(1.0, args.crown_px * 0.06))

        before = int(labels.max())
        total_merged = 0
        for _ in range(args.rounds):
            labels, merged = merge_round(labels, surface, lab_image, args)
            total_merged += merged
            if merged == 0:
                break

        out_folder = args.out / folder
        out_folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_folder / f"{stem}_labels.png"), labels.astype(np.uint16))

        after = int(labels.max())
        metrics = [m for m in (metrics_from_region(r) for r in regionprops(labels)) if m]
        frame = pd.DataFrame(metrics)
        if len(frame):
            frame.insert(0, "id", np.arange(1, len(frame) + 1))
            frame.insert(0, "frame", f"{stem}{originals[0].suffix}")
            frame.insert(0, "folder", folder)
            rows.append(frame)

        print(f"  {folder}/{stem}: {before} -> {after} Instanzen ({total_merged} verschmolzen)")

    if not rows:
        print("Nichts verschmolzen.")
        return

    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(args.out / "all_crowns_merged.csv", index=False)
    print(f"\n{len(combined)} Kronen -> {args.out}/all_crowns_merged.csv")
    print(
        combined.groupby("folder")
        .agg(kronen=("id", "count"), durchmesser_px=("durchmesser_px", "median"),
             kompaktheit=("kompaktheit", "median"))
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
