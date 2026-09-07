"""Real parallax from consecutive video frames instead of estimated depth.

So far the height information came from a monocular depth model -- that is, from
an estimate that can measure nothing in a single image and that invents smooth
surfaces in low-contrast areas. That was consistently the weakest point of the
pipeline.

The frames of a folder, however, come from the same video, a few tenths of a
second to seconds apart. The drone moved in between, so two frames contain real
parallax: tall objects shift more than the ground.

Method:
  1. SIFT correspondences between two frames.
  2. A homography via RANSAC. It describes the mapping of a *plane* -- for nadir
     captures essentially the ground -- and at the same time absorbs the rotation
     and zoom of the camera.
  3. Dense optical flow between frame A and the homography-warped frame B.
  4. Whatever residual flow remains is the parallax. Its magnitude grows with the
     height above the fitted plane -- that is a measured surrogate CHM.

The scale is unknown (no camera calibration), but for treetop finding and
watershed a relative height is entirely sufficient -- exactly as with the
monocular surrogate CHM, only measured instead of guessed.

Example:
    python stereo_probe.py --folders 80m dense1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from depth_probe import hillshade, normalize
from infer_species import IMAGE_SUFFIXES, REPO_ROOT


def match_frames(gray_a: np.ndarray, gray_b: np.ndarray, max_features: int):
    """SIFT correspondences with a ratio test."""
    sift = cv2.SIFT_create(nfeatures=max_features)
    kp_a, desc_a = sift.detectAndCompute(gray_a, None)
    kp_b, desc_b = sift.detectAndCompute(gray_b, None)
    if desc_a is None or desc_b is None or len(kp_a) < 20 or len(kp_b) < 20:
        return np.empty((0, 2)), np.empty((0, 2))

    matcher = cv2.BFMatcher()
    pairs = matcher.knnMatch(desc_a, desc_b, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 10:
        return np.empty((0, 2)), np.empty((0, 2))

    return (
        np.float32([kp_a[m.queryIdx].pt for m in good]),
        np.float32([kp_b[m.trainIdx].pt for m in good]),
    )


def parallax_map(image_a: np.ndarray, image_b: np.ndarray, args) -> tuple[np.ndarray, dict] | None:
    """Residual flow after homography warping = parallax."""
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

    points_a, points_b = match_frames(gray_a, gray_b, args.max_features)
    if len(points_a) < 20:
        return None

    homography, inliers = cv2.findHomography(points_b, points_a, cv2.RANSAC, args.ransac_thresh)
    if homography is None:
        return None

    displacement = np.linalg.norm(points_a - points_b, axis=1)
    stats = {
        "korrespondenzen": len(points_a),
        "inlier": int(inliers.sum()),
        "verschiebung_median_px": float(np.median(displacement)),
        "verschiebung_p90_px": float(np.percentile(displacement, 90)),
    }

    warped = cv2.warpPerspective(gray_b, homography, (gray_a.shape[1], gray_a.shape[0]))

    # Optical flow on the warped pair: the global part is gone, what remains is
    # height-induced. The residual flow is only a few pixels, so sub-pixel
    # accuracy is decisive -- Farneback with a large window smears exactly the
    # crown detail that matters.
    if args.flow == "dis":
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        dis.setFinestScale(args.finest_scale)
        dis.setPatchSize(args.patch_size)
        dis.setUseSpatialPropagation(True)
        flow = dis.calc(gray_a, warped, None)
    else:
        flow = cv2.calcOpticalFlowFarneback(
            gray_a, warped, None,
            pyr_scale=0.5, levels=args.levels, winsize=args.winsize,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
    residual = np.linalg.norm(flow, axis=2)

    # Mask out areas without overlap (black after the warp).
    valid = warped > 0
    residual = np.where(valid, residual, 0.0)

    stats["restfluss_median_px"] = float(np.median(residual[valid])) if valid.any() else 0.0
    stats["restfluss_p95_px"] = float(np.percentile(residual[valid], 95)) if valid.any() else 0.0
    stats["ueberlappung"] = float(valid.mean())
    return residual, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--folders", nargs="*", default=None, help="Folders; None = all of them.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_stereo")
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--ransac-thresh", type=float, default=3.0)
    parser.add_argument("--flow", choices=("dis", "farneback"), default="dis",
                        help="DIS is more sub-pixel accurate and resolves crown detail.")
    parser.add_argument("--finest-scale", type=int, default=0, help="0 = finest level, more detail.")
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--winsize", type=int, default=41, help="Farneback window; large = smoother.")
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--smooth", type=float, default=2.5, help="Smoothing of the parallax map.")
    parser.add_argument("--pair-stride", type=int, default=1,
                        help="Spacing of the pairs in the frame list. Larger = longer baseline, "
                             "hence stronger parallax, but less overlap.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    folders = (
        [args.input / f for f in args.folders]
        if args.folders
        else sorted(p for p in args.input.iterdir() if p.is_dir())
    )

    for folder in folders:
        frames = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if len(frames) < 2:
            print(f"{folder.name}: zu wenige Frames")
            continue

        out_folder = args.out / folder.name
        out_folder.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {folder.name} ===")

        pairs = list(zip(frames, frames[args.pair_stride :]))
        for first, second in pairs:
            image_a, image_b = cv2.imread(str(first)), cv2.imread(str(second))
            if image_a is None or image_b is None or image_a.shape != image_b.shape:
                print(f"  {first.stem} -> {second.stem}: Groessen passen nicht")
                continue

            result = parallax_map(image_a, image_b, args)
            if result is None:
                print(f"  {first.stem} -> {second.stem}: zu wenige Korrespondenzen")
                continue

            residual, stats = result
            smoothed = cv2.GaussianBlur(residual, (0, 0), args.smooth)
            surface = normalize(smoothed)

            stem = f"{first.stem}__{second.stem}"
            np.save(out_folder / f"{stem}_parallax.npy", smoothed)
            cv2.imwrite(str(out_folder / f"{stem}_parallax.jpg"),
                        cv2.applyColorMap((surface * 255).astype(np.uint8), cv2.COLORMAP_TURBO))
            cv2.imwrite(str(out_folder / f"{stem}_hillshade.jpg"),
                        (hillshade(surface) * 255).astype(np.uint8))

            print(
                f"  {first.stem} -> {second.stem}: "
                f"{stats['inlier']}/{stats['korrespondenzen']} Inlier, "
                f"Verschiebung {stats['verschiebung_median_px']:.0f} px (p90 {stats['verschiebung_p90_px']:.0f}), "
                f"Restfluss {stats['restfluss_median_px']:.2f} px (p95 {stats['restfluss_p95_px']:.2f}), "
                f"Ueberlappung {stats['ueberlappung']:.0%}"
            )

    print(f"\nParallaxenkarten in {args.out}")


if __name__ == "__main__":
    main()
