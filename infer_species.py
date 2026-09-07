"""Tree species classification on our own drone frames with DINOvTree.

A two-stage pipeline, because DINOvTree finds no instances of its own:

  1. DeepForest (pretrained on NEON, ~10 cm/px) detects crown boxes in the frame.
  2. A tree-centred crop is cut around every box, scaled to 512x512 and
     classified by DINOvTree-B (checkpoint: Quebec Trees).

The crop is chosen so that it covers the same *ground area* as in training
(512 px * 1.9 cm/px = 9.73 m). The GSD for that is estimated from flight
altitude and horizontal field of view. Alternatively (--crop-mode relative) a
fixed multiple of the detected crown box is used, which needs no knowledge of
the camera.

Example:
    python infer_species.py --input /cold/Mahfuz/chosen_frames --out results
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent
DINOVTREE_DIR = REPO_ROOT / "third_party" / "DINOvTree"
sys.path.insert(0, str(DINOVTREE_DIR))

# Training geometry of the Quebec Trees checkpoint: the tiles were cut at native
# resolution, without resampling.
TRAIN_GSD_M = 0.019
TRAIN_TILE_PX = 512
TRAIN_FOOTPRINT_M = TRAIN_GSD_M * TRAIN_TILE_PX  # 9.728 m

# Classes excluded in the paper (supercategories = annotator uncertainty).
QUEBEC_TREES_EXCLUDE = (2, 3, 10)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
# Class mapping
# --------------------------------------------------------------------------- #


def load_class_names(categories_path: Path, exclude: tuple[int, ...]) -> list[str]:
    """Reconstructs the logit order of the checkpoint.

    Replicates the logic of ``BaseModel._setup_categories`` and
    ``BaseLabeledRasterCocoDataset._remap_class_ids``: drop the excluded ids, sort
    the rest by original id, renumber from zero.
    """
    with open(categories_path) as f:
        categories = json.load(f)["categories"]

    kept = sorted((c for c in categories if c["id"] not in exclude), key=lambda c: c["id"])
    return [c["name"] for c in kept]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def resolve_device(requested: str) -> torch.device:
    """'auto' -> cuda if available, otherwise cpu."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_dinovtree(
    ckpt_path: Path, n_classes: int, max_height: float, device: torch.device | str = "cpu"
) -> torch.nn.Module:
    """Builds the DINOvTree core model and loads the published checkpoint.

    The checkpoint contains the complete fine-tuned backbone, so we do not need
    Meta's DINOv3 weights. The upstream repo insists on fetching them from a URL
    in ``dinov3_urls.json``, so ``torch.hub.load`` is briefly redirected in order
    to build the architecture uninitialised.
    """
    original_hub_load = torch.hub.load

    def hub_load_without_weights(repo, name, source=None, weights=None, **kwargs):
        return original_hub_load(repo, name, source=source, pretrained=False, **kwargs)

    torch.hub.load = hub_load_without_weights
    try:
        from src.models.dinovtree_model.core.dinovtree import DINOvTree

        cfg = OmegaConf.merge(
            OmegaConf.load(DINOVTREE_DIR / "configs" / "model" / "dinovtree.yaml"),
            OmegaConf.create(
                {"tasks": ["classification", "height"], "n_classes": n_classes, "max_height": max_height}
            ),
        )
        model = DINOvTree(cfg)
    finally:
        torch.hub.load = original_hub_load

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = {k[len("model.") :]: v for k, v in state_dict.items() if k.startswith("model.")}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model.to(device)


def build_detector(model_name: str):
    """Loads the pretrained DeepForest crown detector."""
    from deepforest import main as deepforest_main

    detector = deepforest_main.deepforest()
    detector.load_model(model_name)
    detector.eval()
    return detector


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def gsd_from_altitude(altitude_m: float, hfov_deg: float, image_width_px: int) -> float:
    """Ground sampling in m/px from flight altitude and horizontal FOV (nadir)."""
    ground_width_m = 2.0 * altitude_m * math.tan(math.radians(hfov_deg) / 2.0)
    return ground_width_m / image_width_px


def resolve_altitude(folder_name: str, overrides: dict[str, float], fallback: float) -> tuple[float, str]:
    """Determines the flight altitude of a folder and where the value came from.

    Order: explicit --altitudes > a number in the folder name > the default.
    """
    if folder_name in overrides:
        return overrides[folder_name], "override"

    match = re.fullmatch(r"(\d{2,3})\s*m?", folder_name.strip(), flags=re.IGNORECASE)
    if match:
        return float(match.group(1)), "folder_name"

    return fallback, "default"


def parse_altitude_overrides(items: list[str] | None) -> dict[str, float]:
    """'pines=35' 'dense=60' -> {'pines': 35.0, 'dense': 60.0}"""
    overrides = {}
    for item in items or []:
        folder, _, value = item.partition("=")
        if not value:
            raise ValueError(f"--altitudes erwartet ORDNER=HOEHE, bekam: {item!r}")
        overrides[folder.strip()] = float(value)
    return overrides


@dataclass
class CropSpec:
    """Describes how large the crop around a detection should be."""

    mode: str  # "gsd" or "relative"
    gsd_m: float | None
    relative_factor: float
    footprint_m: float = TRAIN_FOOTPRINT_M

    def crop_size_px(self, box_w: float, box_h: float) -> int:
        if self.mode == "gsd":
            assert self.gsd_m is not None
            return max(16, int(round(self.footprint_m / self.gsd_m)))
        return max(16, int(round(self.relative_factor * max(box_w, box_h))))


def crop_centered(image: np.ndarray, cx: float, cy: float, size_px: int) -> np.ndarray:
    """Cuts a square window around (cx, cy), mirroring at the image border.

    The training tiles always lay completely inside the orthomosaic. Trees at the
    frame border would otherwise get black bars here, which the backbone has
    never seen -- mirroring is the more harmless continuation.
    """
    half = size_px // 2
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    x1, y1 = x0 + size_px, y0 + size_px

    pad_left, pad_top = max(0, -x0), max(0, -y0)
    pad_right, pad_bottom = max(0, x1 - image.shape[1]), max(0, y1 - image.shape[0])

    patch = image[max(0, y0) : min(image.shape[0], y1), max(0, x0) : min(image.shape[1], x1)]
    if pad_left or pad_top or pad_right or pad_bottom:
        patch = cv2.copyMakeBorder(
            patch, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_REFLECT_101
        )
    return patch


def to_model_input(patch: np.ndarray) -> np.ndarray:
    """RGB crop -> (3, 512, 512) float32 in [0, 1], as in the upstream dataset."""
    interpolation = cv2.INTER_AREA if patch.shape[0] > TRAIN_TILE_PX else cv2.INTER_LINEAR
    resized = cv2.resize(patch, (TRAIN_TILE_PX, TRAIN_TILE_PX), interpolation=interpolation)
    return np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #


@torch.no_grad()
def classify_crops(
    model: torch.nn.Module, crops: np.ndarray, batch_size: int, device: torch.device | str = "cpu"
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (probabilities [N, C], heights [N])."""
    probabilities, heights = [], []
    for start in range(0, len(crops), batch_size):
        batch = torch.from_numpy(crops[start : start + batch_size]).to(device)
        logits, height = model(batch, {})
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        heights.append(height.cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(heights)


def detect(detector, image_rgb: np.ndarray, args) -> pd.DataFrame:
    """Crown detection over the whole frame, via a sliding window."""
    boxes = detector.predict_tile(
        image=image_rgb.astype("float32"),
        patch_size=args.detector_patch_size,
        patch_overlap=args.detector_patch_overlap,
        iou_threshold=args.detector_iou,
    )
    if boxes is None or len(boxes) == 0:
        return pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "score"])

    boxes = boxes[boxes["score"] >= args.min_score].copy()
    boxes["box_w"] = boxes["xmax"] - boxes["xmin"]
    boxes["box_h"] = boxes["ymax"] - boxes["ymin"]
    boxes = boxes[(boxes["box_w"] >= args.min_box_px) & (boxes["box_h"] >= args.min_box_px)]
    return boxes.sort_values("score", ascending=False).head(args.max_trees_per_frame).reset_index(drop=True)


def process_frame(
    image_path: Path, detector, model, class_names: list[str], crop_spec: CropSpec, args, device
) -> pd.DataFrame:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Bild nicht lesbar: {image_path}")
    image_rgb = image_bgr[:, :, ::-1]

    boxes = detect(detector, image_rgb, args)
    if boxes.empty:
        return pd.DataFrame()

    crops = np.stack(
        [
            to_model_input(
                crop_centered(
                    image_rgb,
                    (row.xmin + row.xmax) / 2.0,
                    (row.ymin + row.ymax) / 2.0,
                    crop_spec.crop_size_px(row.box_w, row.box_h),
                )
            )
            for row in boxes.itertuples()
        ]
    )

    probabilities, heights = classify_crops(model, crops, args.batch_size, device)
    order = np.argsort(-probabilities, axis=1)

    result = pd.DataFrame(
        {
            "frame": image_path.name,
            "xmin": boxes["xmin"].to_numpy(),
            "ymin": boxes["ymin"].to_numpy(),
            "xmax": boxes["xmax"].to_numpy(),
            "ymax": boxes["ymax"].to_numpy(),
            "detector_score": boxes["score"].to_numpy(),
            "crop_px": [crop_spec.crop_size_px(r.box_w, r.box_h) for r in boxes.itertuples()],
            "species_top1": [class_names[i] for i in order[:, 0]],
            "prob_top1": probabilities[np.arange(len(order)), order[:, 0]],
            "species_top2": [class_names[i] for i in order[:, 1]],
            "prob_top2": probabilities[np.arange(len(order)), order[:, 1]],
            "species_top3": [class_names[i] for i in order[:, 2]],
            "prob_top3": probabilities[np.arange(len(order)), order[:, 2]],
            "height_m_pred": heights,
        }
    )
    result["entropy"] = -(probabilities * np.log(probabilities + 1e-12)).sum(axis=1)
    return result


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #


PALETTE = [
    (228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163), (255, 127, 0),
    (255, 255, 51), (166, 86, 40), (247, 129, 191), (153, 153, 153), (0, 206, 209),
    (106, 61, 154), (177, 89, 40), (31, 120, 180), (178, 223, 138),
]


def short_name(name: str) -> str:
    """'Acer saccharum Marshall' -> 'A. saccharum', 'dead' -> 'dead'."""
    parts = name.split()
    if len(parts) >= 2 and parts[0][0].isupper():
        return f"{parts[0][0]}. {parts[1]}"
    return parts[0]


def draw_overlay(image_path: Path, predictions: pd.DataFrame, class_names: list[str], out_path: Path) -> None:
    image = cv2.imread(str(image_path))
    color_of = {name: PALETTE[i % len(PALETTE)][::-1] for i, name in enumerate(class_names)}

    for row in predictions.itertuples():
        color = color_of[row.species_top1]
        p1, p2 = (int(row.xmin), int(row.ymin)), (int(row.xmax), int(row.ymax))
        cv2.rectangle(image, p1, p2, color, 2)
        label = f"{short_name(row.species_top1)} {row.prob_top1:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(image, (p1[0], p1[1] - th - 4), (p1[0] + tw + 2, p1[1]), color, -1)
        cv2.putText(image, label, (p1[0] + 1, p1[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])


# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--ckpt", type=Path, default=REPO_ROOT / "checkpoints" / "dinovtreeb_quebectrees.pth")
    parser.add_argument("--categories", type=Path, default=REPO_ROOT / "third_party" / "quebec_trees_categories.json")

    parser.add_argument("--crop-mode", choices=("gsd", "relative"), default="gsd",
                        help="gsd: the crop covers 9.73 m as in training. relative: a multiple of the crown box.")
    parser.add_argument("--altitude", type=float, default=100.0,
                        help="Flight altitude in m when neither --altitudes nor the folder name says.")
    parser.add_argument("--altitudes", nargs="*", metavar="ORDNER=HOEHE",
                        help="Flight altitude per folder, e.g. --altitudes pines=35 dense=60")
    parser.add_argument("--hfov-deg", type=float, default=73.7,
                        help="Horizontal field of view of the camera in degrees (DJI 24 mm equiv. 16:9 ~ 73.7).")
    parser.add_argument("--relative-factor", type=float, default=2.5,
                        help="Crop size as a multiple of the crown box (--crop-mode relative).")
    parser.add_argument("--footprint-m", type=float, default=TRAIN_FOOTPRINT_M,
                        help="Edge length of the crop on the ground in m (default = the training value 9.73 m).")

    parser.add_argument("--min-score", type=float, default=0.35, help="DeepForest confidence threshold.")
    parser.add_argument("--min-box-px", type=float, default=25.0, help="Discard smaller crown boxes.")
    parser.add_argument("--max-trees-per-frame", type=int, default=40,
                        help="Classify only the N most confident detections (CPU runtime).")
    parser.add_argument("--detector-patch-size", type=int, default=400)
    parser.add_argument("--detector-patch-overlap", type=float, default=0.1)
    parser.add_argument("--detector-iou", type=float, default=0.15)
    parser.add_argument("--detector-model", default="weecology/deepforest-tree")

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--threads", type=int, default=0, help="0 = the torch default.")
    parser.add_argument("--no-overlays", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    altitude_overrides = parse_altitude_overrides(args.altitudes)
    class_names = load_class_names(args.categories, QUEBEC_TREES_EXCLUDE)
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"{len(class_names)} Klassen: {', '.join(class_names)}\n")

    model = build_dinovtree(args.ckpt, n_classes=len(class_names), max_height=30.0, device=device)
    detector = build_detector(args.detector_model)

    args.out.mkdir(parents=True, exist_ok=True)
    all_results = []

    folders = sorted(p for p in args.input.iterdir() if p.is_dir())
    for folder in folders:
        frames = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not frames:
            continue

        altitude, altitude_source = resolve_altitude(folder.name, altitude_overrides, args.altitude)
        out_folder = args.out / folder.name
        out_folder.mkdir(parents=True, exist_ok=True)

        for frame_path in frames:
            width = cv2.imread(str(frame_path)).shape[1]
            gsd = gsd_from_altitude(altitude, args.hfov_deg, width)
            crop_spec = CropSpec(args.crop_mode, gsd, args.relative_factor, args.footprint_m)

            predictions = process_frame(frame_path, detector, model, class_names, crop_spec, args, device)
            if predictions.empty:
                print(f"  {folder.name}/{frame_path.name}: keine Detektion")
                continue

            predictions.insert(0, "folder", folder.name)
            predictions["altitude_m"] = altitude
            predictions["altitude_source"] = altitude_source
            predictions["gsd_m_per_px"] = gsd
            all_results.append(predictions)

            predictions.to_csv(out_folder / f"{frame_path.stem}_trees.csv", index=False)
            if not args.no_overlays:
                draw_overlay(frame_path, predictions, class_names, out_folder / f"{frame_path.stem}_overlay.jpg")

            top = predictions["species_top1"].value_counts().head(3)
            summary = ", ".join(f"{short_name(k)} {v}" for k, v in top.items())
            print(
                f"  {folder.name}/{frame_path.name}: {len(predictions)} Baeume, "
                f"{altitude:.0f} m ({altitude_source}), GSD {gsd*100:.1f} cm/px, "
                f"Crop {predictions['crop_px'].iloc[0]} px | {summary}"
            )

    if not all_results:
        print("Keine Ergebnisse.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(args.out / "all_trees.csv", index=False)

    pivot = (
        combined.pivot_table(index="folder", columns="species_top1", values="frame", aggfunc="count")
        .fillna(0)
        .astype(int)
    )
    pivot.to_csv(args.out / "summary_by_folder.csv")
    print(f"\n{len(combined)} Baeume insgesamt -> {args.out}/all_trees.csv")
    print(f"mittlere Top-1-Wahrscheinlichkeit: {combined['prob_top1'].mean():.3f}")
    print(pivot.to_string())


if __name__ == "__main__":
    main()
