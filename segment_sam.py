"""Crown delineation with SAM (Segment Anything), automatic mask generation.

A different approach from segment_trees.py: instead of building a surface from
estimated depth and partitioning it, SAM segments along real image edges. That
avoids the weakness of the depth model in low-contrast areas -- SAM sees the
crown boundaries directly.

SAM produces masks at all scales simultaneously (leaf, branch, crown, whole
stand). The work is therefore in the selection:
  1. An area window around the expected crown size.
  2. Compactness -- crowns are reasonably round, shadow bands are not.
  3. Overlap resolution: accept greedily by score, discard masks with a high IoU
     against already accepted ones (this removes the scale stacks).

Optionally the depth map from segment_trees.py is used as an extra filter: a
crown should rise above its own margin.

Example:
    python segment_sam.py --frames 100/frame_000537.jpg --crown-px 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from infer_species import IMAGE_SUFFIXES, REPO_ROOT, resolve_device

SAM_MODEL = "facebook/sam-vit-large"  # better than vit-huge per the ablation, see the README


def build_generator(model_id: str, device, args):
    """Build the mask-generation pipeline.

    pred_iou_thresh and stability_score_thresh are the real knobs for the yield:
    SAM uses them internally to discard uncertain masks before they ever come
    out. The defaults (0.88 / 0.95) are meant for everyday objects -- in a canopy,
    where boundaries are objectively fuzzy, they discard the bulk of the crowns.
    """
    from transformers import pipeline

    return pipeline(
        "mask-generation",
        model=model_id,
        device=0 if str(device).startswith("cuda") else -1,
        points_per_crop=args.points_per_crop,
        crop_n_layers=args.crop_layers,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
    )


def mask_metrics(mask: np.ndarray) -> dict[str, float] | None:
    """Area, compactness and bounding box of one binary mask."""
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    if area <= 0 or perimeter <= 0:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    hull_area = float(cv2.contourArea(cv2.convexHull(contour))) or area
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None

    return {
        "cx": moments["m10"] / moments["m00"],
        "cy": moments["m01"] / moments["m00"],
        "xmin": x, "ymin": y, "xmax": x + w, "ymax": y + h,
        "area_px": area,
        # 1.0 = a perfect circle; shadow bands and branch parts fall well below.
        "kompaktheit": 4 * np.pi * area / (perimeter**2),
        "solidity": area / hull_area,
        "seitenverhaeltnis": min(w, h) / max(w, h),
        "durchmesser_px": 2 * np.sqrt(area / np.pi),
    }


def metrics_from_region(region) -> dict[str, float] | None:
    """mask_metrics on the bounding-box crop instead of on the whole image.

    With several hundred instances per frame the difference is considerable: a
    full-frame mask per instance costs 2 megapixels, the crop only the crown.
    """
    metrics = mask_metrics(region.image)
    if metrics is None:
        return None
    y0, x0 = region.bbox[0], region.bbox[1]
    for key in ("cx", "xmin", "xmax"):
        metrics[key] += x0
    for key in ("cy", "ymin", "ymax"):
        metrics[key] += y0
    return metrics


def select_crowns(masks: list[np.ndarray], scores: np.ndarray, args) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Filter the candidates and resolve overlaps greedily by score."""
    expected_area = np.pi * (args.crown_px / 2) ** 2
    min_area, max_area = expected_area * args.min_area_factor, expected_area * args.max_area_factor

    candidates = []
    for mask, score in zip(masks, scores):
        metrics = mask_metrics(mask)
        if metrics is None:
            continue
        if not (min_area <= metrics["area_px"] <= max_area):
            continue
        if metrics["kompaktheit"] < args.min_compactness:
            continue
        if metrics["solidity"] < args.min_solidity:
            continue
        metrics["score"] = float(score)
        candidates.append((metrics, mask))

    candidates.sort(key=lambda item: (-item[0]["score"], -item[0]["area_px"]))

    accepted_records, accepted_masks = [], []
    occupied = None
    for metrics, mask in candidates:
        if occupied is None:
            occupied = np.zeros_like(mask, dtype=bool)
        overlap = np.logical_and(mask, occupied).sum() / max(1, mask.sum())
        if overlap > args.max_overlap:
            continue
        occupied |= mask
        accepted_records.append(metrics)
        accepted_masks.append(mask)

    return pd.DataFrame(accepted_records), accepted_masks


def draw(image_bgr: np.ndarray, masks: list[np.ndarray], crowns: pd.DataFrame, caption: str) -> np.ndarray:
    canvas = image_bgr.copy()
    if masks:
        covered = np.any(np.stack(masks), axis=0)
        # Darken the uncaptured area -- that shows at a glance what is missing.
        canvas[~covered] = (canvas[~covered] * 0.45).astype(np.uint8)
        for mask in masks:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, (80, 230, 120), 2)
    for row in crowns.itertuples():
        cv2.circle(canvas, (int(row.cx), int(row.cy)), 3, (0, 220, 255), -1)

    cv2.rectangle(canvas, (0, 0), (780, 34), (0, 0, 0), -1)
    cv2.putText(canvas, caption, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--frames", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_sam")
    parser.add_argument("--model", default=SAM_MODEL)

    parser.add_argument("--crown-px", type=float, default=100.0)
    parser.add_argument("--min-area-factor", type=float, default=0.12)
    parser.add_argument("--max-area-factor", type=float, default=5.0)
    parser.add_argument("--min-compactness", type=float, default=0.25)
    parser.add_argument("--min-solidity", type=float, default=0.65)
    parser.add_argument("--max-overlap", type=float, default=0.30)

    parser.add_argument("--points-per-crop", type=int, default=48, help="Point grid per tile (48 -> 2304 points).")
    parser.add_argument("--crop-layers", type=int, default=2, help="Extra zoom levels for small objects.")
    parser.add_argument("--points-per-batch", type=int, default=256)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.70,
                        help="SAM-internal quality filter. Lowering it raises the yield markedly.")
    parser.add_argument("--stability-score-thresh", type=float, default=0.85,
                        help="SAM-internal stability filter. Too strict for a canopy.")
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
    print(f"Device: {device} | Modell: {args.model}")

    generator = build_generator(args.model, device, args)
    args.out.mkdir(parents=True, exist_ok=True)

    all_crowns = []
    for frame_path in collect_frames(args):
        if not frame_path.exists():
            print(f"  fehlt: {frame_path}")
            continue

        image_bgr = cv2.imread(str(frame_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # The mask-generation pipeline expects PIL/a path, not a numpy array.
        with torch.no_grad():
            output = generator(Image.fromarray(image_rgb))
        masks = [np.asarray(m, dtype=bool) for m in output["masks"]]
        scores = np.asarray(output["scores"], dtype=np.float32)

        crowns, kept_masks = select_crowns(masks, scores, args)
        covered = np.any(np.stack(kept_masks), axis=0).mean() if kept_masks else 0.0

        caption = (
            f"{len(masks)} SAM-Masken -> {len(crowns)} Kronen | "
            f"Durchmesser med {crowns['durchmesser_px'].median():.0f} px | "
            f"Bildabdeckung {covered:.0%}"
            if len(crowns)
            else f"{len(masks)} SAM-Masken -> 0 Kronen nach Filter"
        )

        out_folder = args.out / frame_path.parent.name
        out_folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(out_folder / f"{frame_path.stem}_sam.jpg"),
            draw(image_bgr, kept_masks, crowns, caption),
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )

        if len(crowns):
            crowns["abdeckung"] = covered
            crowns.insert(0, "frame", frame_path.name)
            crowns.insert(0, "folder", frame_path.parent.name)
            all_crowns.append(crowns)

        print(f"  {frame_path.parent.name}/{frame_path.name}: {caption}")

    if not all_crowns:
        print("Keine Kronen gefunden.")
        return

    combined = pd.concat(all_crowns, ignore_index=True)
    combined.to_csv(args.out / "all_crowns_sam.csv", index=False)
    print(f"\n{len(combined)} Kronen -> {args.out}/all_crowns_sam.csv")
    print(
        combined.groupby("folder")
        .agg(
            kronen=("frame", "count"),
            durchmesser_px=("durchmesser_px", "median"),
            kompaktheit=("kompaktheit", "median"),
            abdeckung=("abdeckung", "mean"),
        )
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
