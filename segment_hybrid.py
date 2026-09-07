"""Hybrid crown delineation: SAM first, depth watershed only for the remainder.

The two previous methods have complementary strengths and weaknesses:

  SAM       finds precise crown boundaries where there is edge contrast and
            leaves out the uncertain parts -- high precision, coverage 50-74 %.
  Watershed partitions the area completely, but it is a partition, not a
            detector: it also tiles over places where there is no tree.

The hybrid uses each method where it is strong: SAM fixes the certain crowns,
then the watershed runs **exclusively on the area left over** between the SAM
crowns. It can therefore no longer tile over the whole image, and its known
weakness -- pseudo-crowns in structurally poor areas -- is limited by the
prominence check and the same shape filters as for SAM.

Every crown carries its origin in the output (`quelle` = sam | watershed), so
that the quality of the two parts can be judged separately.

Example:
    python segment_hybrid.py --frames 100/frame_000537.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.measure import label as cc_label, regionprops
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from infer_species import IMAGE_SUFFIXES, REPO_ROOT, resolve_device
from segment_sam import SAM_MODEL, build_generator, mask_metrics, select_crowns
from segment_trees import DEPTH_MODEL, DepthEstimator, build_pseudo_chm


def residual_crowns(
    chm: np.ndarray, residual: np.ndarray, args
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Watershed on the area SAM did not capture.

    Important: the prominence threshold is computed on the *remaining area*, not
    on the whole image. Otherwise the relief of the SAM crowns already found
    dominates the statistics and the rest falls below the threshold wholesale.
    """
    smoothed = cv2.GaussianBlur(chm, (0, 0), max(0.8, args.crown_px * args.smooth_factor))

    # Exclude gaps, ground and shadow within the remaining area.
    if residual.sum() < 10:
        return pd.DataFrame(), []
    canopy = residual & (smoothed > np.percentile(smoothed[residual], args.gap_percentile))
    if canopy.sum() < 10:
        return pd.DataFrame(), []

    low, high = np.percentile(smoothed[canopy], [5, 95])
    seeds = h_maxima(np.where(canopy, smoothed, smoothed.min()), max(1e-6, (high - low) * args.peak_prominence))
    markers = cc_label(seeds > 0)
    if markers.max() == 0:
        return pd.DataFrame(), []

    labels = watershed(-smoothed, markers, mask=canopy)

    records, masks = [], []
    for region in regionprops(labels):
        mask = labels == region.label
        metrics = mask_metrics(mask)
        if metrics is None:
            continue
        records.append(metrics)
        masks.append(mask)

    if not records:
        return pd.DataFrame(), []

    # The same shape filters as for SAM, so both parts stay comparable.
    frame = pd.DataFrame(records)
    expected_area = np.pi * (args.crown_px / 2) ** 2
    keep = (
        frame["area_px"].between(expected_area * args.min_area_factor, expected_area * args.max_area_factor)
        & (frame["kompaktheit"] >= args.min_compactness)
        & (frame["solidity"] >= args.min_solidity)
    )
    frame["score"] = np.nan  # the watershed provides no confidence value
    return frame[keep].reset_index(drop=True), [m for m, k in zip(masks, keep) if k]


def draw_hybrid(image_bgr: np.ndarray, sam_masks, ws_masks, crowns: pd.DataFrame, caption: str) -> np.ndarray:
    canvas = image_bgr.copy()
    all_masks = list(sam_masks) + list(ws_masks)
    if all_masks:
        covered = np.any(np.stack(all_masks), axis=0)
        canvas[~covered] = (canvas[~covered] * 0.45).astype(np.uint8)

    for mask in sam_masks:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (80, 230, 120), 2)  # green = SAM
    for mask in ws_masks:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (255, 190, 60), 2)  # blue = watershed addition

    for row in crowns.itertuples():
        cv2.circle(canvas, (int(row.cx), int(row.cy)), 3, (0, 220, 255), -1)

    cv2.rectangle(canvas, (0, 0), (900, 34), (0, 0, 0), -1)
    cv2.putText(canvas, caption, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--frames", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_hybrid")
    parser.add_argument("--depth-cache", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/depth_cache"))
    parser.add_argument("--sam-model", default=SAM_MODEL)

    parser.add_argument("--crown-px", type=float, default=100.0)
    parser.add_argument("--min-area-factor", type=float, default=0.12)
    parser.add_argument("--max-area-factor", type=float, default=5.0)
    parser.add_argument("--min-compactness", type=float, default=0.25)
    parser.add_argument("--min-solidity", type=float, default=0.65)
    parser.add_argument("--max-overlap", type=float, default=0.30)

    parser.add_argument("--points-per-crop", type=int, default=48)
    parser.add_argument("--crop-layers", type=int, default=2)
    parser.add_argument("--points-per-batch", type=int, default=256)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.70)
    parser.add_argument("--stability-score-thresh", type=float, default=0.85)

    parser.add_argument("--detrend-factor", type=float, default=3.0)
    parser.add_argument("--smooth-factor", type=float, default=0.06)
    parser.add_argument("--gap-percentile", type=float, default=15.0)
    parser.add_argument("--peak-prominence", type=float, default=0.10)
    parser.add_argument("--dilate-sam", type=int, default=3,
                        help="Dilate the SAM masks slightly before forming the remainder, so "
                             "that margin seams do not survive as mini crowns of their own.")

    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
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
    print(f"Device: {device} | SAM: {args.sam_model}")

    generator = build_generator(args.sam_model, device, args)
    estimator = DepthEstimator(DEPTH_MODEL, device, args.depth_cache)
    args.out.mkdir(parents=True, exist_ok=True)

    all_crowns = []
    for frame_path in collect_frames(args):
        if not frame_path.exists():
            print(f"  fehlt: {frame_path}")
            continue

        image_bgr = cv2.imread(str(frame_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            output = generator(Image.fromarray(image_rgb))
        sam_crowns, sam_masks = select_crowns(
            [np.asarray(m, dtype=bool) for m in output["masks"]],
            np.asarray(output["scores"], dtype=np.float32),
            args,
        )

        covered = np.any(np.stack(sam_masks), axis=0) if sam_masks else np.zeros(image_rgb.shape[:2], dtype=bool)
        if args.dilate_sam > 0 and sam_masks:
            kernel = np.ones((args.dilate_sam, args.dilate_sam), np.uint8)
            covered = cv2.dilate(covered.astype(np.uint8), kernel).astype(bool)

        depth = estimator(image_rgb, f"{frame_path.parent.name}__{frame_path.stem}")
        chm = build_pseudo_chm(depth, args.crown_px, args.detrend_factor)
        ws_crowns, ws_masks = residual_crowns(chm, ~covered, args)

        if len(sam_crowns):
            sam_crowns["quelle"] = "sam"
        if len(ws_crowns):
            ws_crowns["quelle"] = "watershed"
        crowns = pd.concat([f for f in (sam_crowns, ws_crowns) if len(f)], ignore_index=True)

        # Label map: instance ids 1..N in the same order as the CSV rows. That
        # allows the visualisation to be re-rendered later at will, without
        # running SAM and the depth model again.
        label_map = np.zeros(image_rgb.shape[:2], dtype=np.uint16)
        for index, mask in enumerate(sam_masks + ws_masks, start=1):
            label_map[mask] = index
        if len(crowns):
            crowns.insert(0, "id", np.arange(1, len(crowns) + 1))

        total = np.any(np.stack(sam_masks + ws_masks), axis=0).mean() if (sam_masks or ws_masks) else 0.0
        caption = (
            f"{len(sam_crowns)} SAM + {len(ws_crowns)} Watershed = {len(crowns)} Kronen | "
            f"Abdeckung {total:.0%}"
        )

        out_folder = args.out / frame_path.parent.name
        out_folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(out_folder / f"{frame_path.stem}_hybrid.jpg"),
            draw_hybrid(image_bgr, sam_masks, ws_masks, crowns, caption),
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
        cv2.imwrite(str(out_folder / f"{frame_path.stem}_labels.png"), label_map)

        if len(crowns):
            crowns["abdeckung"] = total
            crowns.insert(0, "frame", frame_path.name)
            crowns.insert(0, "folder", frame_path.parent.name)
            all_crowns.append(crowns)

        print(f"  {frame_path.parent.name}/{frame_path.name}: {caption}")

    if not all_crowns:
        print("Keine Kronen gefunden.")
        return

    combined = pd.concat(all_crowns, ignore_index=True)
    combined.to_csv(args.out / "all_crowns_hybrid.csv", index=False)

    print(f"\n{len(combined)} Kronen -> {args.out}/all_crowns_hybrid.csv")
    print(
        combined.groupby("folder")
        .agg(
            kronen=("frame", "count"),
            sam=("quelle", lambda s: (s == "sam").sum()),
            watershed=("quelle", lambda s: (s == "watershed").sum()),
            durchmesser_px=("durchmesser_px", "median"),
            abdeckung=("abdeckung", "mean"),
        )
        .round(2)
        .to_string()
    )
    print("\nFormguete nach Quelle (Kompaktheit, 1.0 = Kreis):")
    print(combined.groupby("quelle")[["kompaktheit", "solidity", "durchmesser_px"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
