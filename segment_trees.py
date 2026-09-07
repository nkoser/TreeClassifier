"""Crown delineation via monocular depth + marker-based watershed.

The standard method of forest remote sensing for individual tree delineation
works on a CHM (canopy height model): treetops are local maxima, crown
boundaries lie in the dips between them, and a marker-based watershed draws the
lines. Our single frames have no CHM -- but a monocular depth model supplies a
surrogate surface containing the same structure.

Steps:
  1. Estimate depth, invert it (closer to the camera = higher).
  2. Detrend: subtract the large-scale component. That removes the tilt of the
     camera and the ground-plane prior of the model -- analogous to DSM minus DTM.
  3. Smooth, so that foliage texture does not create phantom treetops.
  4. Local maxima as treetop markers, minimum distance = half a crown diameter.
  5. Watershed on the inverted surface, restricted to the canopy mask.
  6. Filter the segments by area and shape.

The depth maps are cached, so that tuning the parameters afterwards works
without running model inference again.

Example:
    python segment_trees.py --crown-px 100 --frames 80m/frame_000297.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.measure import label as cc_label, regionprops
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from infer_species import IMAGE_SUFFIXES, REPO_ROOT, resolve_device

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"


# --------------------------------------------------------------------------- #
# Depth surface
# --------------------------------------------------------------------------- #


class DepthEstimator:
    """Loads the depth model once and caches the maps on disk."""

    def __init__(self, model_id: str, device, cache_dir: Path | None) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def __call__(self, image_rgb: np.ndarray, key: str) -> np.ndarray:
        cache_path = self.cache_dir / f"{key}.npy" if self.cache_dir else None
        if cache_path and cache_path.exists():
            return np.load(cache_path)

        inputs = self.processor(images=image_rgb, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        depth = self.processor.post_process_depth_estimation(
            outputs, target_sizes=[(image_rgb.shape[0], image_rgb.shape[1])]
        )[0]["predicted_depth"]
        depth = depth.float().cpu().numpy()

        if cache_path:
            np.save(cache_path, depth)
        return depth


def build_pseudo_chm(depth: np.ndarray, crown_px: float, detrend_factor: float) -> np.ndarray:
    """Depth -> surrogate CHM: inverted, large-scale trend removed."""
    surface = -depth.astype(np.float32)
    trend = cv2.GaussianBlur(surface, (0, 0), max(1.0, crown_px * detrend_factor))
    return surface - trend


# --------------------------------------------------------------------------- #
# Delineation
# --------------------------------------------------------------------------- #


def delineate(
    chm: np.ndarray, crown_px: float, gap_percentile: float, smooth_factor: float, prominence: float
) -> tuple[np.ndarray, np.ndarray]:
    """Watershed delineation with a prominence check on the treetops.

    Watershed is a partition, not a detector: it divides the mask into exactly as
    many parts as markers go into it. A pure local maximum is therefore far too
    weak a criterion -- in flat areas noise creates arbitrarily many of them and
    hence pseudo-crowns. h_maxima instead demands that a treetop rise at least
    `prominence` above its surroundings before it counts.
    """
    smoothed = cv2.GaussianBlur(chm, (0, 0), max(0.8, crown_px * smooth_factor))

    # Canopy mask: the lowest areas are gaps, ground or shadow.
    canopy = smoothed > np.percentile(smoothed, gap_percentile)

    # Prominence threshold relative to the robust range of the surface, so that it
    # does not depend on the arbitrary scale of the depth model.
    low, high = np.percentile(smoothed[canopy], [5, 95])
    height_threshold = max(1e-6, (high - low) * prominence)

    seeds = h_maxima(np.where(canopy, smoothed, smoothed.min()), height_threshold)
    markers = cc_label(seeds > 0)
    if markers.max() == 0:
        return np.zeros_like(chm, dtype=np.int32), np.empty((0, 2))

    peaks = np.array([region.centroid for region in regionprops(markers)])
    labels = watershed(-smoothed, markers, mask=canopy)
    return labels, peaks


def crowns_to_frame(labels: np.ndarray, chm: np.ndarray, crown_px: float, args) -> pd.DataFrame:
    """Measure the segments and filter them by area and shape."""
    expected_area = np.pi * (crown_px / 2) ** 2
    min_area = expected_area * args.min_area_factor
    max_area = expected_area * args.max_area_factor

    records = []
    for region in regionprops(labels, intensity_image=chm):
        if not (min_area <= region.area <= max_area):
            continue
        # Very elongated segments are usually two merged crowns or a shadow band,
        # not a single crown.
        if region.axis_major_length > 0 and (
            region.axis_minor_length / region.axis_major_length < args.min_axis_ratio
        ):
            continue

        cy, cx = region.centroid
        y0, x0, y1, x1 = region.bbox
        records.append(
            {
                "cx": cx,
                "cy": cy,
                "xmin": x0,
                "ymin": y0,
                "xmax": x1,
                "ymax": y1,
                "area_px": region.area,
                "durchmesser_px": region.equivalent_diameter_area,
                "achsenverhaeltnis": (
                    region.axis_minor_length / region.axis_major_length if region.axis_major_length else 0.0
                ),
                "chm_mean": region.intensity_mean,
                "label": region.label,
            }
        )
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #


def draw_crowns(image_bgr: np.ndarray, labels: np.ndarray, crowns: pd.DataFrame, caption: str) -> np.ndarray:
    canvas = image_bgr.copy()
    keep = set(crowns["label"].tolist())

    # Draw the boundaries of the kept segments as lines.
    kept_mask = np.isin(labels, list(keep)) if keep else np.zeros_like(labels, dtype=bool)
    for label in keep:
        mask = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (80, 230, 120), 2)

    # Discarded segments discreetly in red, so the filter can be judged.
    discarded = (labels > 0) & ~kept_mask
    canvas[discarded] = (0.65 * canvas[discarded] + 0.35 * np.array([60, 60, 200])).astype(np.uint8)

    for row in crowns.itertuples():
        cv2.circle(canvas, (int(row.cx), int(row.cy)), 3, (0, 220, 255), -1)

    cv2.rectangle(canvas, (0, 0), (760, 34), (0, 0, 0), -1)
    cv2.putText(canvas, caption, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--frames", nargs="*", default=None, help="Paths relative to --input; None = all.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_segment")
    parser.add_argument("--depth-cache", type=Path, default=Path("/scratch/shared") / "nik" / "data" / "treeclf" / "depth_cache")

    parser.add_argument("--crown-px", type=float, default=100.0,
                        help="Expected crown diameter in pixels. Drives every scale.")
    parser.add_argument("--detrend-factor", type=float, default=3.0,
                        help="Sigma of the trend filter as a multiple of --crown-px.")
    parser.add_argument("--smooth-factor", type=float, default=0.06,
                        help="Sigma of the smoothing as a multiple of --crown-px.")
    parser.add_argument("--peak-prominence", type=float, default=0.08,
                        help="How far a treetop has to rise above its surroundings, "
                             "as a fraction of the 5-95 percentile range of the surrogate CHM.")
    parser.add_argument("--gap-percentile", type=float, default=10.0,
                        help="Surface percentile below which a pixel counts as gap/ground.")
    parser.add_argument("--min-area-factor", type=float, default=0.15)
    parser.add_argument("--max-area-factor", type=float, default=4.0)
    parser.add_argument("--min-axis-ratio", type=float, default=0.35)

    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--save-chm", action="store_true", help="Also save the surrogate CHM as an image.")
    return parser.parse_args()


def collect_frames(args) -> list[Path]:
    if args.frames:
        return [args.input / relative for relative in args.frames]
    frames = []
    for folder in sorted(p for p in args.input.iterdir() if p.is_dir()):
        frames.extend(sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES))
    return frames


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Device: {device} | erwarteter Kronendurchmesser: {args.crown_px:.0f} px")

    estimator = DepthEstimator(DEPTH_MODEL, device, args.depth_cache)
    args.out.mkdir(parents=True, exist_ok=True)

    all_crowns = []
    for frame_path in collect_frames(args):
        if not frame_path.exists():
            print(f"  fehlt: {frame_path}")
            continue

        image_bgr = cv2.imread(str(frame_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        depth = estimator(image_rgb, f"{frame_path.parent.name}__{frame_path.stem}")
        chm = build_pseudo_chm(depth, args.crown_px, args.detrend_factor)
        labels, peaks = delineate(
            chm, args.crown_px, args.gap_percentile, args.smooth_factor, args.peak_prominence
        )
        crowns = crowns_to_frame(labels, chm, args.crown_px, args)

        # How much of the segmented canopy survives the filter?
        segmented = labels > 0
        kept = np.isin(labels, crowns["label"].to_numpy()) if len(crowns) else np.zeros_like(segmented)
        abdeckung = kept.sum() / max(1, segmented.sum())

        out_folder = args.out / frame_path.parent.name
        out_folder.mkdir(parents=True, exist_ok=True)

        caption = (
            f"crown_px {args.crown_px:.0f} | {len(peaks)} Wipfel -> {len(crowns)} Kronen | "
            f"Durchmesser med {crowns['durchmesser_px'].median():.0f} px | "
            f"Abdeckung {abdeckung:.0%}"
            if len(crowns)
            else f"{len(peaks)} Wipfel -> 0 Kronen nach Filter"
        )
        cv2.imwrite(
            str(out_folder / f"{frame_path.stem}_crowns.jpg"),
            draw_crowns(image_bgr, labels, crowns, caption),
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
        if args.save_chm:
            normalized = cv2.normalize(chm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imwrite(str(out_folder / f"{frame_path.stem}_chm.jpg"), cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO))

        if len(crowns):
            crowns["abdeckung"] = abdeckung
            crowns.insert(0, "frame", frame_path.name)
            crowns.insert(0, "folder", frame_path.parent.name)
            all_crowns.append(crowns)

        print(f"  {frame_path.parent.name}/{frame_path.name}: {caption}")

    if not all_crowns:
        print("Keine Kronen gefunden.")
        return

    combined = pd.concat(all_crowns, ignore_index=True)
    combined.to_csv(args.out / "all_crowns.csv", index=False)

    print(f"\n{len(combined)} Kronen -> {args.out}/all_crowns.csv")
    summary = combined.groupby("folder").agg(
        kronen=("frame", "count"),
        pro_frame=("frame", lambda s: round(len(s) / s.nunique(), 1)),
        durchmesser_px=("durchmesser_px", "median"),
        achsenverhaeltnis=("achsenverhaeltnis", "median"),
        abdeckung=("abdeckung", "mean"),
    ).round(2)
    print(summary.to_string())


if __name__ == "__main__":
    main()
