"""Probe: does monocular depth estimation give a usable crown relief?

Crown delineation in a closed canopy fails for lack of height information -- two
neighbouring green crowns often have no visible boundary in RGB. In forestry
practice a CHM (canopy height model) solves that: treetops as local maxima,
boundaries by watershed. Without photogrammetry there is no CHM here, but a
monocular depth model might serve as a substitute.

This script computes the depth maps and stores them colourised, plus a
hillshade, in which crown relief is most visible to the eye. Absolute metric
accuracy is irrelevant here and unusable anyway -- the models are trained on
ground perspectives, not on nadir from 80 m. All that counts is whether the
*relative* structure separates treetops from canopy gaps.

Example:
    python depth_probe.py --frames pines/frame_000006.jpg dense/frame_000073.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from infer_species import REPO_ROOT, resolve_device

MODELS = {
    "depth_anything": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    "depthpro": "apple/DepthPro-hf",
}

DEFAULT_FRAMES = [
    "pines/frame_000006.jpg",
    "dense/frame_000073.jpg",
    "mixed1/frame_000594.jpg",
    "80m/frame_000297.jpg",
]


def normalize(surface: np.ndarray) -> np.ndarray:
    """To [0, 1], robust against outliers (2nd/98th percentile)."""
    low, high = np.percentile(surface, [2, 98])
    if high - low < 1e-9:
        return np.zeros_like(surface, dtype=np.float32)
    return np.clip((surface - low) / (high - low), 0, 1).astype(np.float32)


def hillshade(surface: np.ndarray, azimuth_deg: float = 315.0, altitude_deg: float = 45.0) -> np.ndarray:
    """Hillshading -- makes fine height differences visible."""
    dy, dx = np.gradient(cv2.GaussianBlur(surface, (0, 0), 2.0))
    slope = np.arctan(np.hypot(dx, dy) * 40.0)
    aspect = np.arctan2(-dx, dy)
    az, alt = np.radians(360.0 - azimuth_deg + 90.0), np.radians(altitude_deg)
    shaded = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(shaded, 0, 1)


@torch.no_grad()
def estimate_depth(model_id: str, image_rgb: np.ndarray, device) -> np.ndarray:
    """Returns the predicted depth at the original resolution."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()

    inputs = processor(images=image_rgb, return_tensors="pt").to(device)
    outputs = model(**inputs)
    depth = processor.post_process_depth_estimation(
        outputs, target_sizes=[(image_rgb.shape[0], image_rgb.shape[1])]
    )[0]["predicted_depth"]
    return depth.float().cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--frames", nargs="*", default=DEFAULT_FRAMES)
    parser.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_depth")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    for name in args.models:
        model_id = MODELS[name]
        print(f"\n=== {name} ({model_id}) ===")

        for relative in args.frames:
            frame_path = args.input / relative
            if not frame_path.exists():
                print(f"  fehlt: {frame_path}")
                continue

            # cvtColor rather than [:, :, ::-1]: the slice view has negative
            # strides, which the HF image processor cannot turn into a tensor.
            image_rgb = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)
            depth = estimate_depth(model_id, image_rgb, device)

            # Closer to the camera = taller tree, hence the inversion.
            surface = normalize(-depth)
            out_folder = args.out / frame_path.parent.name
            out_folder.mkdir(parents=True, exist_ok=True)
            stem = f"{frame_path.stem}_{name}"

            cv2.imwrite(
                str(out_folder / f"{stem}_depth.jpg"),
                cv2.applyColorMap((surface * 255).astype(np.uint8), cv2.COLORMAP_TURBO),
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            cv2.imwrite(
                str(out_folder / f"{stem}_hillshade.jpg"),
                (hillshade(surface) * 255).astype(np.uint8),
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )

            # Contrast of the surface: how much relief is in there?
            smoothed = cv2.GaussianBlur(surface, (0, 0), 3.0)
            print(
                f"  {relative}: Tiefe {depth.min():.2f}..{depth.max():.2f} "
                f"(median {np.median(depth):.2f}), Relief-Std nach Glaettung {smoothed.std():.4f}, "
                f"lokale Gradientenenergie {np.abs(np.gradient(smoothed)).mean():.5f}"
            )

    print(f"\nBilder in {args.out}")


if __name__ == "__main__":
    main()
