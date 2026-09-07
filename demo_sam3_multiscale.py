#!/usr/bin/env python3
"""Tree crown segmentation with SAM 3, multi-scale — standalone demo.

Segments individual tree crowns in a drone frame and writes an overlay image, a
label map and a table of per-crown measurements. This file is self-contained:
copy it anywhere, no other file of this repository is needed.

WHAT IT DOES

SAM 3 takes the word "tree" as a text prompt and returns one mask per instance
with a confidence score. Two things are added around it:

  * Tiling. A 1920x1080 frame is squeezed to the model's input resolution, and a
    100 px crown shrinks to about 50 px on the way in — small enough to be lost.
    The frame is therefore cut into overlapping tiles that are segmented
    separately.
  * Multiple tile levels. One tiling fixes how large a crown appears to the
    model. Crown sizes vary a lot within a stand, so several levels are run
    (default 2x2, 3x3, 4x4) and the instances are merged afterwards, greedily by
    score. That catches small and large crowns alike.

Instances cut by an inner tile edge are dropped: because the tiles overlap, the
same object is contained completely in the neighbouring tile. Without this you
get dead-straight cuts across crowns along the tile grid.

SETUP

  pip install "transformers>=4.57" torch opencv-python pillow numpy pandas

  A GPU is strongly recommended. On CPU one frame takes several minutes;
  on a recent GPU it is a few seconds per tile level.

ACCESS

  facebook/sam3 is a gated model on Hugging Face:

    1. Open https://huggingface.co/facebook/sam3 and request access.
    2. Create a read token at https://huggingface.co/settings/tokens
    3. Make it available, either with
         huggingface-cli login
       or by exporting it:
         export HF_TOKEN=hf_xxxxxxxxxxxx

INPUT DATA

  --image takes either a single image file or a folder. A folder is searched
  recursively, so any layout works; nothing has to be renamed or moved:

      frames/                          python demo_sam3_multiscale.py \
        pines/frame_000006.jpg             --image frames/ --out results/
        pines/frame_000042.jpg
        dense/frame_000073.jpg

  Accepted: .jpg .jpeg .png .tif .tiff, any resolution. Paths may be relative
  or absolute; nothing outside --image and --out is read or written.

  The one real requirement is not the path but the SCALE. SAM 3 has to see a
  crown as an object, so a crown should be roughly 60-200 px across in the input
  image. Our drone frames at 1920x1080 from 30-100 m sit in that range. If your
  imagery is much coarser (a crown of 20 px), upscale it beforehand; if it is
  much finer, downscale it. Getting no crowns at all is almost always a scale
  problem, not a threshold problem.

USAGE

  python demo_sam3_multiscale.py --image frame.jpg
  python demo_sam3_multiscale.py --image frames/ --out results/
  python demo_sam3_multiscale.py --image frame.jpg --tile-levels 2 3 --threshold 0.25

OUTPUT

  The folder structure below --image is mirrored under --out, so frames of the
  same name in different folders keep their own results:

      results/
        pines/frame_000006_sam3.jpg      the frame with crown outlines drawn on it
        pines/frame_000006_labels.png    16-bit label map, 0 = background,
                                         1..N = one crown each
        pines/frame_000006_crowns.csv    one row per crown: position, area,
                                         shape, score
        all_crowns.csv                   every crown of the run, with an `image`
                                         column holding the path

DEFAULTS

  The defaults are the configuration that worked best in our tests: prompt
  "tree", score threshold 0.15, tile levels 2/3/4, and no shape filter. SAM 3
  already returns instances rather than a stack of scales, so filtering by shape
  mostly discards correct crowns — pass --shape-filter to see for yourself.
  "tree crown" and "treetop" as prompts return almost nothing; SAM 3 knows the
  term, not the paraphrase.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #

def tile_boxes(width: int, height: int, tiles: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Overlapping tiles as (x0, y0, x1, y1). tiles=1 returns the whole image."""
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
    """Does the mask touch a tile edge that is not also an image edge?

    Such an instance is cut off. Since the tiles overlap, the same object is
    contained completely in the neighbouring tile, so the cut version can be
    discarded without losing the crown.
    """
    left, top, right, bottom = at_image_edge
    return (
        (not left and mask[:, 0].any())
        or (not right and mask[:, -1].any())
        or (not top and mask[0, :].any())
        or (not bottom and mask[-1, :].any())
    )


def mask_metrics(mask: np.ndarray) -> dict[str, float] | None:
    """Position, area and shape of one binary mask. None if it is degenerate."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    if area <= 0 or perimeter <= 0:
        return None

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    hull_area = float(cv2.contourArea(cv2.convexHull(contour))) or area

    return {
        "cx": moments["m10"] / moments["m00"],
        "cy": moments["m01"] / moments["m00"],
        "xmin": x, "ymin": y, "xmax": x + w, "ymax": y + h,
        "area_px": area,
        # 1.0 = a perfect circle; shadow bands and branch parts fall well below.
        "compactness": 4 * np.pi * area / (perimeter**2),
        "solidity": area / hull_area,
        "aspect_ratio": min(w, h) / max(w, h),
        "diameter_px": 2 * np.sqrt(area / np.pi),
    }


def merge_instances(candidates: list[tuple[np.ndarray, float]], max_overlap: float) -> list[tuple[np.ndarray, float]]:
    """Merge across tiles and scales: accept greedily by score, drop a candidate
    that is already covered by more than `max_overlap` of its own area."""
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


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def load_model(name: str, device: torch.device):
    """Load SAM 3, turning the two common failure modes into readable errors."""
    try:
        from transformers import Sam3Model, Sam3Processor
    except ImportError:
        sys.exit(
            "transformers is too old or missing — SAM 3 needs transformers >= 4.57.\n"
            '  pip install --upgrade "transformers>=4.57"'
        )

    try:
        processor = Sam3Processor.from_pretrained(name)
        model = Sam3Model.from_pretrained(name).to(device).eval()
    except OSError as error:
        sys.exit(
            f"Could not load {name}: {error}\n\n"
            "facebook/sam3 is gated. Request access at\n"
            "  https://huggingface.co/facebook/sam3\n"
            "then log in with `huggingface-cli login` or export HF_TOKEN=hf_..."
        )
    return model, processor


@torch.no_grad()
def segment_tile(model, processor, image_rgb: np.ndarray, prompt: str, threshold: float, device):
    """Instance masks of one tile, in tile coordinates."""
    inputs = processor(images=Image.fromarray(image_rgb), text=prompt, return_tensors="pt").to(device)
    outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, threshold=threshold, mask_threshold=0.5, target_sizes=[image_rgb.shape[:2]]
    )[0]

    masks, scores = results.get("masks"), results.get("scores")
    if masks is None:
        raise RuntimeError(f"Unexpected output structure: {list(results)}")

    masks = masks.cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
    scores = scores.cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
    return [np.asarray(m, dtype=bool) for m in masks], scores


def segment_frame(model, processor, image_rgb: np.ndarray, args, device) -> tuple[list[np.ndarray], list[float], dict]:
    """Run every tile level over one frame and merge the instances."""
    height, width = image_rgb.shape[:2]
    candidates: list[tuple[np.ndarray, float]] = []
    cut_away = 0

    for tiles in args.tile_levels:
        for x0, y0, x1, y1 in tile_boxes(width, height, tiles, args.tile_overlap):
            tile_masks, tile_scores = segment_tile(
                model, processor, image_rgb[y0:y1, x0:x1], args.prompt, args.threshold, device
            )
            at_edge = (x0 == 0, y0 == 0, x1 == width, y1 == height)
            for mask, score in zip(tile_masks, tile_scores):
                if args.drop_cut and touches_inner_edge(mask, at_edge):
                    cut_away += 1
                    continue
                full = np.zeros((height, width), dtype=bool)
                full[y0:y1, x0:x1] = mask
                candidates.append((full, float(score)))

    raw_count = len(candidates)
    merged = merge_instances(candidates, args.max_overlap)

    # Optional shape filter. Off by default: SAM 3 returns instances rather than
    # a stack of leaf/branch/crown, so the filter mostly removes correct crowns.
    expected_area = np.pi * (args.crown_px / 2) ** 2
    masks, records = [], []
    for mask, score in merged:
        metrics = mask_metrics(mask)
        if metrics is None:
            continue
        if args.shape_filter:
            if not (expected_area * args.min_area_factor <= metrics["area_px"] <= expected_area * args.max_area_factor):
                continue
            if metrics["compactness"] < args.min_compactness or metrics["solidity"] < args.min_solidity:
                continue
        metrics["score"] = score
        records.append(metrics)
        masks.append(mask)

    stats = {"raw": raw_count, "merged": len(merged), "cut_away": cut_away}
    return masks, records, stats


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def draw_overlay(image_bgr: np.ndarray, masks: list[np.ndarray], caption: str) -> np.ndarray:
    """Crown outlines on the frame, with a caption bar along the top."""
    canvas = image_bgr.copy()
    for mask in masks:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (80, 230, 120), 2)

    bar = min(canvas.shape[1], 12 + 11 * len(caption))
    cv2.rectangle(canvas, (0, 0), (bar, 34), (0, 0, 0), -1)
    cv2.putText(canvas, caption, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def write_results(out_dir: Path, stem: str, image_bgr: np.ndarray, masks, records, stats) -> pd.DataFrame:
    """Write overlay, label map and CSV. Returns the table for the summary."""
    height, width = image_bgr.shape[:2]
    covered = float(np.any(np.stack(masks), axis=0).mean()) if masks else 0.0

    # Label map: 0 = background, 1..N = one crown each. uint16, so up to 65535
    # crowns fit; the PNG is lossless, so the ids survive a round trip.
    label_map = np.zeros((height, width), dtype=np.uint16)
    for index, mask in enumerate(masks, start=1):
        label_map[mask] = index
    cv2.imwrite(str(out_dir / f"{stem}_labels.png"), label_map)

    caption = (f"SAM3 multi-scale | {stats['raw']} raw -> {len(records)} crowns | "
               f"{covered:.0%} covered | {stats['cut_away']} cut discarded")
    cv2.imwrite(str(out_dir / f"{stem}_sam3.jpg"), draw_overlay(image_bgr, masks, caption),
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    crowns = pd.DataFrame(records)
    if len(crowns):
        crowns.insert(0, "id", np.arange(1, len(crowns) + 1))
        crowns["covered_fraction"] = covered
    crowns.to_csv(out_dir / f"{stem}_crowns.csv", index=False)

    print(f"  {caption}")
    print(f"  -> {out_dir / (stem + '_sam3.jpg')}")
    return crowns


def collect_images(target: Path) -> list[Path]:
    """One image, or every image below a folder, subfolders included."""
    if target.is_dir():
        return sorted(p for p in target.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    return [target]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", type=Path, required=True,
                        help="A single image, or a folder of images.")
    parser.add_argument("--out", type=Path, default=Path("sam3_demo_out"),
                        help="Output folder (created if missing).")

    parser.add_argument("--prompt", default="tree",
                        help='Text prompt. "tree" works; "tree crown" and "treetop" do not.')
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Score threshold for instances. Lower = more crowns, more false ones.")
    parser.add_argument("--tile-levels", type=int, nargs="+", default=[2, 3, 4],
                        help="Tiles per axis, one entry per scale level. 1 = whole image.")
    parser.add_argument("--tile-overlap", type=float, default=0.15,
                        help="Tile overlap as a fraction of the tile size.")
    parser.add_argument("--max-overlap", type=float, default=0.30,
                        help="Discard an instance if more of its area than this is already taken.")
    parser.add_argument("--drop-cut", action=argparse.BooleanOptionalAction, default=True,
                        help="Discard instances cut by an inner tile edge (default on).")

    parser.add_argument("--shape-filter", action="store_true",
                        help="Additionally filter crowns by area and roundness (off by default).")
    parser.add_argument("--crown-px", type=float, default=100.0,
                        help="Expected crown diameter in pixels; only used by --shape-filter.")
    parser.add_argument("--min-area-factor", type=float, default=0.12)
    parser.add_argument("--max-area-factor", type=float, default=5.0)
    parser.add_argument("--min-compactness", type=float, default=0.25)
    parser.add_argument("--min-solidity", type=float, default=0.65)

    parser.add_argument("--model", default="facebook/sam3", help="Hugging Face model id.")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    images = collect_images(args.image)
    if not images:
        sys.exit(f"No image found at {args.image}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    if device.type == "cpu":
        print("No GPU in use — expect several minutes per frame.", flush=True)

    print(f"Device: {device} | model: {args.model} | prompt: {args.prompt!r} | "
          f"tile levels: {args.tile_levels}", flush=True)
    model, processor = load_model(args.model, device)

    args.out.mkdir(parents=True, exist_ok=True)
    all_crowns = []

    root = args.image if args.image.is_dir() else args.image.parent

    for path in images:
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            print(f"  unreadable, skipped: {path}")
            continue

        # Mirror the input folder structure under --out. Frames of the same name
        # in different folders (pines/frame_000006.jpg, dense/frame_000006.jpg)
        # would otherwise overwrite each other's results.
        relative = path.relative_to(root)
        out_dir = args.out / relative.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"{relative} ({image_bgr.shape[1]}x{image_bgr.shape[0]})", flush=True)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        masks, records, stats = segment_frame(model, processor, image_rgb, args, device)

        crowns = write_results(out_dir, path.stem, image_bgr, masks, records, stats)
        if len(crowns):
            crowns.insert(0, "image", str(relative))
            all_crowns.append(crowns)

    if not all_crowns:
        print("\nNo crowns found. Try a lower --threshold, or check the scale: "
              "a crown should be roughly 60-200 px across in the input image.")
        return

    combined = pd.concat(all_crowns, ignore_index=True)
    combined.to_csv(args.out / "all_crowns.csv", index=False)
    print(f"\n{len(combined)} crowns in {len(all_crowns)} image(s) "
          f"-> {args.out / 'all_crowns.csv'}")
    print(f"median diameter {combined['diameter_px'].median():.0f} px | "
          f"median compactness {combined['compactness'].median():.2f} | "
          f"mean coverage {combined['covered_fraction'].mean():.0%}")


if __name__ == "__main__":
    main()
