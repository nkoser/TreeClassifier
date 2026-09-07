"""Crown delineation with SAM 3 (text-prompted instance segmentation).

A different mechanism from SAM 1/2: instead of sampling a blind point grid and
filtering crowns out afterwards, SAM 3 gets the term directly as text ("tree")
and returns instances with a score. That removes the scale stack -- SAM 1 returns
leaf, branch and crown for one point at the same time, and the overlap resolution
has to clean that up.

Exactly the cases where SAM 1 fails here -- shaded crowns and frayed conifers
without a closed edge -- are the ones where semantic knowledge about "tree" helps
more than edge contrast.

Access: facebook/sam3 is gated on HuggingFace. Put a token in $HF_HOME/token
(see the README), then it runs without any further change.

Example:
    python segment_sam3.py --prompt tree --frames dense/frame_000073.jpg
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
from segment_sam import mask_metrics

SAM3_MODEL = "facebook/sam3"


def tile_boxes(width: int, height: int, tiles: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Overlapping tiles. Without tiling, a 100 px crown shrinks to a good 50 px
    in the internal resize -- small enough for instances to get lost."""
    if tiles <= 1:
        return [(0, 0, width, height)]

    step_x, step_y = width / tiles, height / tiles
    pad_x, pad_y = step_x * overlap, step_y * overlap
    boxes = []
    for iy in range(tiles):
        for ix in range(tiles):
            x0 = int(max(0, ix * step_x - pad_x))
            y0 = int(max(0, iy * step_y - pad_y))
            x1 = int(min(width, (ix + 1) * step_x + pad_x))
            y1 = int(min(height, (iy + 1) * step_y + pad_y))
            boxes.append((x0, y0, x1, y1))
    return boxes


def touches_inner_edge(mask: np.ndarray, at_image_edge: tuple[bool, bool, bool, bool]) -> bool:
    """Does the mask touch a tile edge that is not an image edge?

    Such instances are cut off. Because the tiles overlap, the same object is
    contained completely in the neighbouring tile -- so the cut version may be
    discarded. Without this you get dead-straight cuts across crowns along the
    tile grid.
    """
    left, top, right, bottom = at_image_edge
    return (
        (not left and mask[:, 0].any())
        or (not right and mask[:, -1].any())
        or (not top and mask[0, :].any())
        or (not bottom and mask[-1, :].any())
    )


@torch.no_grad()
def segment_tile(model, processor, image_rgb: np.ndarray, prompt: str, threshold: float, device):
    """Instance masks of one tile, in tile coordinates."""
    inputs = processor(images=Image.fromarray(image_rgb), text=prompt, return_tensors="pt").to(device)
    outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, threshold=threshold, mask_threshold=0.5, target_sizes=[image_rgb.shape[:2]]
    )[0]

    masks = results.get("masks")
    scores = results.get("scores")
    if masks is None:
        raise RuntimeError(f"Unerwartete Ausgabestruktur: {list(results)}")

    masks = masks.cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
    scores = scores.cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
    return [np.asarray(m, dtype=bool) for m in masks], scores


def merge_instances(candidates: list[tuple[np.ndarray, float]], max_overlap: float) -> list[tuple[np.ndarray, float]]:
    """Merge across tiles: accept greedily by score, discard strongly overlapping
    duplicates coming from the tile margins."""
    candidates.sort(key=lambda item: -item[1])
    accepted: list[tuple[np.ndarray, float]] = []
    occupied: np.ndarray | None = None

    for mask, score in candidates:
        if occupied is None:
            occupied = np.zeros_like(mask, dtype=bool)
        if np.logical_and(mask, occupied).sum() / max(1, mask.sum()) > max_overlap:
            continue
        occupied |= mask
        accepted.append((mask, score))
    return accepted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--frames", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_sam3")
    parser.add_argument("--model", default=SAM3_MODEL)
    parser.add_argument("--prompt", default="tree", help='Text prompt, e.g. "tree", "tree crown", "treetop".')
    parser.add_argument("--threshold", type=float, default=0.3, help="Score threshold for the instances.")

    parser.add_argument("--tiles", type=int, default=2, help="Tiles per axis (1 = whole image).")
    parser.add_argument("--tiles-multi", type=int, nargs="*", default=None,
                        help="Combine several tile levels, e.g. --tiles-multi 1 2 3. "
                             "Default: only --tiles.")
    parser.add_argument("--tile-overlap", type=float, default=0.15)
    parser.add_argument("--max-overlap", type=float, default=0.30)
    parser.add_argument("--drop-cut", action=argparse.BooleanOptionalAction, default=True,
                        help="Discard instances cut by a tile edge (on by default).")

    # Shape filter identical to segment_sam.py, so the ablation stays comparable.
    parser.add_argument("--crown-px", type=float, default=100.0)
    parser.add_argument("--min-area-factor", type=float, default=0.12)
    parser.add_argument("--max-area-factor", type=float, default=5.0)
    parser.add_argument("--min-compactness", type=float, default=0.25)
    parser.add_argument("--min-solidity", type=float, default=0.65)
    parser.add_argument("--no-shape-filter", action="store_true",
                        help="Turn the shape filter off -- SAM 3 already returns instances, not a scale stack.")

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
    print(f"Device: {device} | Modell: {args.model} | Prompt: {args.prompt!r}")

    from transformers import Sam3Model, Sam3Processor

    processor = Sam3Processor.from_pretrained(args.model)
    model = Sam3Model.from_pretrained(args.model).to(device).eval()
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.tiles_multi:
        args.tiles_multi = [args.tiles]
    print(f"Kachelstufen: {args.tiles_multi}")

    expected_area = np.pi * (args.crown_px / 2) ** 2
    all_crowns = []

    for frame_path in collect_frames(args):
        if not frame_path.exists():
            print(f"  fehlt: {frame_path}")
            continue

        image_bgr = cv2.imread(str(frame_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]

        # Multi-scale: one tiling fixes how large a crown appears to the model.
        # In a closed canopy the crown sizes vary a lot, and a fixed level only
        # catches part of them. Running several levels and merging the instances
        # afterwards by score catches small and large crowns alike.
        candidates: list[tuple[np.ndarray, float]] = []
        angeschnitten = 0
        for tiles in args.tiles_multi:
            for x0, y0, x1, y1 in tile_boxes(width, height, tiles, args.tile_overlap):
                tile_masks, tile_scores = segment_tile(
                    model, processor, image_rgb[y0:y1, x0:x1], args.prompt, args.threshold, device
                )
                at_edge = (x0 == 0, y0 == 0, x1 == width, y1 == height)
                for mask, score in zip(tile_masks, tile_scores):
                    if args.drop_cut and touches_inner_edge(mask, at_edge):
                        angeschnitten += 1
                        continue
                    full = np.zeros((height, width), dtype=bool)
                    full[y0:y1, x0:x1] = mask
                    candidates.append((full, float(score)))

        raw_count = len(candidates)
        merged = merge_instances(candidates, args.max_overlap)

        records, masks = [], []
        for mask, score in merged:
            metrics = mask_metrics(mask)
            if metrics is None:
                continue
            if not args.no_shape_filter:
                if not (expected_area * args.min_area_factor <= metrics["area_px"] <= expected_area * args.max_area_factor):
                    continue
                if metrics["kompaktheit"] < args.min_compactness or metrics["solidity"] < args.min_solidity:
                    continue
            metrics["score"] = score
            records.append(metrics)
            masks.append(mask)

        crowns = pd.DataFrame(records)
        covered = float(np.any(np.stack(masks), axis=0).mean()) if masks else 0.0

        out_folder = args.out / frame_path.parent.name
        out_folder.mkdir(parents=True, exist_ok=True)

        label_map = np.zeros((height, width), dtype=np.uint16)
        for index, mask in enumerate(masks, start=1):
            label_map[mask] = index
        cv2.imwrite(str(out_folder / f"{frame_path.stem}_labels.png"), label_map)

        caption = (
            f"SAM3 '{args.prompt}' | {raw_count} roh -> {len(crowns)} Kronen | "
            f"Abdeckung {covered:.0%} | {angeschnitten} angeschnitten verworfen"
        )
        canvas = image_bgr.copy()
        for mask in masks:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, (80, 230, 120), 2)
        cv2.rectangle(canvas, (0, 0), (900, 34), (0, 0, 0), -1)
        cv2.putText(canvas, caption, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_folder / f"{frame_path.stem}_sam3.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])

        if len(crowns):
            crowns.insert(0, "id", np.arange(1, len(crowns) + 1))
            crowns["abdeckung"] = covered
            crowns.insert(0, "frame", frame_path.name)
            crowns.insert(0, "folder", frame_path.parent.name)
            all_crowns.append(crowns)

        print(f"  {frame_path.parent.name}/{frame_path.name}: {caption}")

    if not all_crowns:
        print("Keine Kronen gefunden.")
        return

    combined = pd.concat(all_crowns, ignore_index=True)
    combined.to_csv(args.out / "all_crowns_sam3.csv", index=False)
    print(f"\n{len(combined)} Kronen -> {args.out}/all_crowns_sam3.csv")
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
