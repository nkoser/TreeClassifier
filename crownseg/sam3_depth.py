"""SAM 3 with Depth Pro: the text prompt finds crowns, the depth splits and completes.

On BAMFORESTS test1 (Hain) every SAM variant showed the same profile: the
boundaries sit excellently -- mean IoU of the hits 0.75 to 0.77, better than
anything trained -- but too little is found. SAM 3 with a text prompt reaches a
recall of 0.30, the depth-prompt variant 0.16. The bottleneck is missing
instances, not poor delineation.

The depth therefore gets clearly separated jobs here:

  split        A SAM mask spanning several treetops is divided at the treetops
               (watershed inside the mask). `segment_hybrid.py` does not handle
               that case at all -- there a SAM mask is always exactly one crown.
  complete     Treetops without a SAM mask get a watershed basin on the
               remaining area. That is the part `segment_hybrid.py` already does,
               reused here unchanged.
  seed         A treetop lying in no SAM 3 mask becomes a point prompt. The depth
               supplies only the where; an image model draws the boundary again --
               the difference from the completion step, where the shape came from
               the watershed and hit nothing.
  confirm      A mask without any treetop survives only if its shape fits.

Why Depth Pro and not Depth-Anything-V2 any longer: for splitting, what counts
is not the metric correctness of the depth but how sharp the edge between two
neighbouring treetops is. Depth-Anything-V2-Metric-Outdoor delivers a smooth
surface on which two touching crowns merge into one hill. That is a conjecture,
though, not a measurement -- so the model is a switch (`--depth-model`) and both
variants are scored against the same ground truth.

    python crownseg/sam3_depth.py --input <ordner> --depth-model depthpro
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.measure import label as cc_label, regionprops
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infer_species import IMAGE_SUFFIXES, REPO_ROOT, resolve_device  # noqa: E402
from segment_hybrid import residual_crowns  # noqa: E402
from segment_sam import mask_metrics  # noqa: E402
from segment_prompted import find_peaks, pick_candidates, prompt_sam  # noqa: E402
from segment_sam import SAM_MODEL  # noqa: E402
from segment_sam3 import SAM3_MODEL, segment_tile  # noqa: E402
from segment_trees import DepthEstimator, build_pseudo_chm  # noqa: E402

DEPTH_MODELS = {
    "depthpro": "apple/DepthPro-hf",
    "dav2": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
}


def treetops(chm: np.ndarray, args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Treetops as local maxima in the surrogate CHM, computed over the canopy."""
    smoothed = cv2.GaussianBlur(chm, (0, 0), max(0.8, args.crown_px * args.smooth_factor))
    canopy = smoothed > np.percentile(smoothed, args.gap_percentile)
    if canopy.sum() < 10:
        return np.zeros(chm.shape, np.int32), smoothed, canopy

    low, high = np.percentile(smoothed[canopy], [5, 95])
    seeds = h_maxima(np.where(canopy, smoothed, smoothed.min()),
                     max(1e-6, (high - low) * args.peak_prominence))
    return cc_label(seeds > 0), smoothed, canopy


def split_by_tops(mask: np.ndarray, markers: np.ndarray, smoothed: np.ndarray) -> list[np.ndarray]:
    """Split a mask at its treetops; unchanged if it has at most one."""
    local = np.where(mask, markers, 0)
    present = [value for value in np.unique(local) if value > 0]
    if len(present) <= 1:
        return [mask]

    labels = watershed(-smoothed, local, mask=mask)
    parts = [labels == value for value in present]
    return [part for part in parts if part.sum() > 0]


def shape_filter(mask: np.ndarray, args) -> dict | None:
    metrics = mask_metrics(mask)
    if metrics is None:
        return None
    expected = np.pi * (args.crown_px / 2) ** 2
    if not expected * args.min_area_factor <= metrics["area_px"] <= expected * args.max_area_factor:
        return None
    if metrics["kompaktheit"] < args.min_compactness or metrics["solidity"] < args.min_solidity:
        return None
    return metrics


def crowns_from_frame(model, processor, estimator, image_rgb, key, args, device, seeder=None):
    depth = estimator(image_rgb, key)
    chm = build_pseudo_chm(depth, args.crown_px, args.detrend_factor)
    markers, smoothed, canopy = treetops(chm, args)
    # The basins serve only as a size reference for the candidate choice -- SAM
    # draws the boundary. Exactly the role in which the watershed is any good.
    basins = watershed(-smoothed, markers, mask=canopy) if seeder is not None else None

    masks, scores = segment_tile(model, processor, image_rgb, args.prompt, args.threshold, device)
    order = np.argsort(-np.asarray(scores)) if len(scores) else []

    occupied = np.zeros(image_rgb.shape[:2], dtype=bool)
    records, kept, sources = [], [], []
    split_count = 0

    for index in order:
        mask = np.asarray(masks[index], dtype=bool)
        if mask.sum() == 0:
            continue
        # Subtract already claimed area instead of discarding the mask entirely --
        # SAM 3 regularly returns nested candidates.
        if np.logical_and(mask, occupied).sum() / mask.sum() > args.max_overlap:
            continue
        mask = mask & ~occupied

        parts = split_by_tops(mask, markers, smoothed)
        split_count += len(parts) - 1
        for part in parts:
            metrics = shape_filter(part, args)
            if metrics is None:
                continue
            records.append({**metrics, "score": float(scores[index])})
            kept.append(part)
            sources.append("sam3")
        occupied |= mask

    # Seeding: free treetops as point prompts to SAM.
    #
    # Measured on test1: of the treetops lying in no SAM 3 mask, two thirds
    # (66.7 %) at prominence 0.02 lie in a crown SAM 3 missed. So the positions
    # are good, only the watershed shapes at the same places hit nothing. Ceiling
    # of this step: recall 0.554 instead of 0.296, F1 0.606 instead of 0.342.
    saat = dict(frei=0, kandidaten=0, ueberlappt=0, form=0, genommen=0)
    if seeder is not None:
        occupied_now = occupied.copy()
        free = []
        for region in regionprops(markers):
            y, x = int(region.centroid[0]), int(region.centroid[1])
            if not occupied_now[y, x]:
                free.append((x, y, region.label))

        saat["frei"] = len(free)
        if free:
            points = np.array([[x, y] for x, y, _ in free], dtype=np.float32)
            basin_ids = np.array([label for _, _, label in free])
            sam_masks, sam_scores = prompt_sam(
                seeder[0], seeder[1], image_rgb, points, device, args.chunk)
            chosen = pick_candidates(sam_masks, sam_scores, basins, basin_ids, args)
            saat["kandidaten"] = len(chosen)

            for mask, score, metrics in chosen:
                mask = np.asarray(mask, dtype=bool)
                if mask.sum() == 0:
                    continue
                # Crowns touch; a freshly seeded crown almost always overlaps its
                # neighbours. So it is measured against its own, more generous
                # threshold than SAM 3 itself uses.
                if np.logical_and(mask, occupied).sum() / mask.sum() > args.seed_max_overlap:
                    saat["ueberlappt"] += 1
                    continue
                filtered = shape_filter(mask, args)
                if filtered is None:
                    saat["form"] += 1
                    continue
                records.append({**filtered, "score": float(score)})
                kept.append(mask)
                sources.append("saat")
                saat["genommen"] += 1
                occupied |= mask

    # Remaining area: treetops for which SAM 3 delivered nothing.
    #
    # Measured on test1 (Hain): of 291 crowns added this way, not a single one
    # hits a real crown at IoU 0.5; at IoU 0.1 it is 2.7 %. They sit in gaps and
    # shadows, not on trees -- by construction the remaining area is what SAM 3
    # considered not to be a tree, and in there the watershed reliably finds
    # nothing. Off by default for that reason.
    if not args.residual:
        frame = pd.DataFrame(records)
        if not frame.empty:
            frame["quelle"] = sources
        return frame, kept, split_count, len(np.unique(markers)) - 1, saat

    if args.dilate_sam > 0 and kept:
        kernel = np.ones((args.dilate_sam, args.dilate_sam), np.uint8)
        grown = cv2.dilate(occupied.astype(np.uint8), kernel).astype(bool)
    else:
        grown = occupied
    residual_frame, residual_masks = residual_crowns(chm, ~grown, args)
    for _, row in residual_frame.iterrows():
        records.append(row.to_dict())
        sources.append("tiefe")
    kept.extend(residual_masks)

    frame = pd.DataFrame(records)
    if not frame.empty:
        frame["quelle"] = sources
    return frame, kept, split_count, len(np.unique(markers)) - 1, saat


def write_outputs(out_dir: Path, stem: str, image_bgr, masks, frame, caption) -> None:
    labels = np.zeros(image_bgr.shape[:2], dtype=np.uint16)
    for index, mask in enumerate(masks, start=1):
        labels[mask] = index
    cv2.imwrite(str(out_dir / f"{stem}_labels.png"), labels)

    canvas = image_bgr.copy()
    for mask, source in zip(masks, frame.get("quelle", ["sam3"] * len(masks))):
        colour = (80, 230, 120) if source == "sam3" else (255, 190, 60)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, colour, 2)
    cv2.rectangle(canvas, (0, 0), (900, 34), (0, 0, 0), -1)
    cv2.putText(canvas, caption, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / f"{stem}_sam3depth.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_sam3depth")
    parser.add_argument("--depth-model", default="depthpro", choices=list(DEPTH_MODELS))
    parser.add_argument("--depth-cache", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/depth_cache"))
    parser.add_argument("--sam3-model", default=SAM3_MODEL)
    parser.add_argument("--prompt", default="tree")
    parser.add_argument("--threshold", type=float, default=0.15)

    parser.add_argument("--crown-px", type=float, default=275.0)
    parser.add_argument("--min-area-factor", type=float, default=0.12)
    parser.add_argument("--max-area-factor", type=float, default=5.0)
    parser.add_argument("--min-compactness", type=float, default=0.25)
    parser.add_argument("--min-solidity", type=float, default=0.65)
    parser.add_argument("--max-overlap", type=float, default=0.30)

    parser.add_argument("--detrend-factor", type=float, default=3.0)
    parser.add_argument("--smooth-factor", type=float, default=0.06)
    parser.add_argument("--gap-percentile", type=float, default=15.0)
    parser.add_argument("--peak-prominence", type=float, default=0.02,
                        help="0.10 was tuned for 100 px crowns; far too strict at 275 px.")
    parser.add_argument("--seed-free-peaks", action=argparse.BooleanOptionalAction, default=True,
                        help="Pass free treetops to SAM as point prompts.")
    parser.add_argument("--sam-model", default=SAM_MODEL, help="Point-promptable model for the seeding.")
    parser.add_argument("--select", choices=("basin", "score", "area"), default="basin")
    parser.add_argument("--chunk", type=int, default=24)
    parser.add_argument("--seed-max-overlap", type=float, default=0.60,
                        help="How much a seeded crown may share with already placed ones.")
    parser.add_argument("--residual", action=argparse.BooleanOptionalAction, default=False,
                        help="Add extra crowns from the remaining area (measured to be worthless).")
    parser.add_argument("--dilate-sam", type=int, default=3)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    return parser.parse_args()


def main() -> None:
    from transformers import Sam3Model, Sam3Processor

    args = parse_args()
    device = resolve_device(args.device)
    model_id = DEPTH_MODELS[args.depth_model]
    print(f"Device: {device} | SAM3 '{args.prompt}' | Tiefe: {model_id}", flush=True)

    processor = Sam3Processor.from_pretrained(args.sam3_model)
    model = Sam3Model.from_pretrained(args.sam3_model).to(device).eval()

    seeder = None
    if args.seed_free_peaks:
        from transformers import AutoProcessor, SamModel

        seeder = (SamModel.from_pretrained(args.sam_model).to(device).eval(),
                  AutoProcessor.from_pretrained(args.sam_model))
        print(f"Saat freier Wipfel ueber {args.sam_model}", flush=True)
    # A separate cache per depth model -- otherwise the Depth Pro run would read
    # the maps of the Depth Anything run and unknowingly measure the same twice.
    estimator = DepthEstimator(model_id, device, args.depth_cache / args.depth_model)

    folders = sorted(p for p in args.input.iterdir() if p.is_dir()) or [args.input]
    rows = []
    for folder in folders:
        frames = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        out_dir = args.out / folder.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for frame_path in frames:
            image_bgr = cv2.imread(str(frame_path))
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            key = f"{folder.name}_{frame_path.stem}"
            frame, masks, splits, tops, saat = crowns_from_frame(
                model, processor, estimator, image_rgb, key, args, device, seeder)

            counts = frame["quelle"].value_counts().to_dict() if not frame.empty else {}
            caption = (f"{len(masks)} Kronen | {counts.get('sam3', 0)} SAM3 (+{splits} geteilt) | "
                       f"{counts.get('saat', 0)} gesaet | {counts.get('tiefe', 0)} aus Tiefe | "
                       f"{tops} Wipfel")
            print(f"    Saat: {saat['frei']} frei -> {saat['kandidaten']} Kandidaten, "
                  f"{saat['ueberlappt']} zu ueberlappend, {saat['form']} Form, "
                  f"{saat['genommen']} genommen", flush=True)
            write_outputs(out_dir, frame_path.stem, image_bgr, masks, frame, caption)
            print(f"  {folder.name}/{frame_path.name}: {caption}", flush=True)

            if not frame.empty:
                frame.insert(0, "frame", frame_path.stem)
                frame.insert(0, "folder", folder.name)
                rows.append(frame)

    if rows:
        table = pd.concat(rows, ignore_index=True)
        table.to_csv(args.out / "all_crowns.csv", index=False)
        print(f"\n{len(table)} Kronen -> {args.out / 'all_crowns.csv'}")
        print(table.groupby(["folder", "quelle"]).size().to_string())


if __name__ == "__main__":
    main()
