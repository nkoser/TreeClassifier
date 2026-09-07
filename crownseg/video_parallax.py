"""Measured height maps from a drone video instead of estimated depth.

Two things failed with monocular depth, both measured: it can only partly
separate merged crowns (+0.030 F1), and it does not tell lawn from canopy -- the
filter attempt against that brought nothing. Both are jobs for a *measured*
height, and a video supplies one.

The drone moves between two frames. Tall objects shift more than the ground in
doing so, and this residual flow after subtracting the ground plane is real
parallax -- a measured surrogate CHM, not an estimate. The method for it is
already in `stereo_probe.parallax_map`; only the choice of image pairs is added
here.

Difference from `build_parallax.py`: that one tries every pair of a folder and
keeps all images in memory. With four frames per folder that works; with a video
of 2224 frames it does not -- quadratically many pairs. Here every frame instead
gets fixed partners at a defined temporal distance.

The choice of distances is the real parameter: too close and there is no baseline
(the drone was stationary or barely moved), too far and the images no longer
overlap enough for a common homography. Hence several distances at once, each
normalised to its baseline and averaged.

    python crownseg/video_parallax.py --video /cold/Mahfuz/DJI_...MP4 --count 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from depth_probe import hillshade, normalize  # noqa: E402
from stereo_probe import parallax_map  # noqa: E402


class Settings:
    """The parameters `parallax_map` expects."""

    def __init__(self, args) -> None:
        self.max_features = args.max_features
        self.ransac_thresh = args.ransac_thresh
        self.flow = args.flow
        self.finest_scale = args.finest_scale
        self.patch_size = args.patch_size
        self.winsize = args.winsize
        self.levels = args.levels
        self.smooth = args.smooth


def destripe(surface: np.ndarray, window: int) -> np.ndarray:
    """Remove a per-row offset -- off by default, because it has no effect.

    The optical flow produces narrow horizontal bands that are not present in the
    raw frames. This function subtracts the outlier of the row median against its
    running median. In an A/B test on the same map that changes nothing (36
    conspicuous rows with and without, amplitude 0.00366 against 0.00368) -- so
    the bands are not an offset of whole rows.

    More important is the amplitude measurement: the bands amount to **2.9 % of
    the relief range**, and the crown relief is about 35 times stronger. For
    treetop finding and watershed they are in the noise. That they look so strong
    in the hillshading is a matter of rendering -- a hillshade shows derivatives,
    and a small disturbance with a sharp edge creates more contrast in it than a
    large, soft dome. The function stays in place for the case that another video
    does show real row offsets.
    """
    rows = np.median(surface, axis=1)
    half = window // 2
    padded = np.pad(rows, half, mode="edge")
    baseline = np.array([np.median(padded[i : i + window]) for i in range(len(rows))])
    return (surface - (rows - baseline)[:, None]).astype(np.float32)


def extract(video: Path, out_dir: Path, start: int, stride: int, count: int) -> list[Path]:
    """Store every n-th frame as JPEG -- the rest of the pipeline reads images."""
    out_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Video nicht lesbar: {video}")

    paths, index = [], 0
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    while len(paths) < count:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride == 0:
            path = out_dir / f"frame_{start + index:06d}.jpg"
            if not path.exists():
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            paths.append(path)
        index += 1
    capture.release()
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, default=Path("/cold/Mahfuz/DJI_20230506174726_0004_Z_80m.MP4"))
    parser.add_argument("--out", type=Path, default=Path("/scratch/shared/nik/data/treeclf/video"))
    parser.add_argument("--name", default=None, help="Folder name; defaults to the video name.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=10, help="Every n-th video frame.")
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--offsets", type=int, nargs="*", default=[2, 4, 8],
                        help="Partner distances in extracted frames, forwards and backwards.")

    parser.add_argument("--min-displacement", type=float, default=8.0,
                        help="Pairs with less camera motion return only noise.")
    parser.add_argument("--min-overlap", type=float, default=0.5)
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--ransac-thresh", type=float, default=3.0)
    parser.add_argument("--flow", choices=("dis", "farneback"), default="dis")
    parser.add_argument("--finest-scale", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--winsize", type=int, default=41)
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--destripe", type=int, default=0,
                        help="Row correction; measured to have no effect, see destripe(). 0 = off.")
    parser.add_argument("--smooth", type=float, default=2.5)
    parser.add_argument("--detrend-sigma", type=float, default=180.0,
                        help="Preview only: width of the subtracted trend.")
    args = parser.parse_args()

    name = args.name or args.video.stem
    frames_dir = args.out / name / "frames"
    cache_dir = args.out / name / "parallax"
    preview_dir = args.out / name / "vorschau"
    cache_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    paths = extract(args.video, frames_dir, args.start, args.stride, args.count)
    print(f"{len(paths)} Frames aus {args.video.name} (jeder {args.stride}., ab {args.start})", flush=True)
    if len(paths) < max(args.offsets) + 1:
        print("Zu wenige Frames fuer die gewaehlten Abstaende.")
        return

    images = [cv2.imread(str(p)) for p in paths]
    settings = Settings(args)
    written = 0

    for index, target in enumerate(paths):
        # A per-pixel sum and counter instead of a list: a partner frame covers
        # the target image only partly, and `parallax_map` returns zero outside
        # the overlap. A plain mean pulls the value down where fewer partners
        # contribute -- which creates straight seam edges across the map, exactly
        # along the image borders of the warped partners.
        total = np.zeros(images[index].shape[:2], np.float32)
        counts = np.zeros(images[index].shape[:2], np.float32)
        used = []
        for offset in args.offsets:
            for other in (index - offset, index + offset):
                if not 0 <= other < len(paths):
                    continue
                result = parallax_map(images[index], images[other], settings)
                if result is None:
                    continue
                residual, stats = result
                if (stats["verschiebung_median_px"] < args.min_displacement
                        or stats["ueberlappung"] < args.min_overlap):
                    continue
                # Normalise to baseline 1, otherwise the widest pair dominates.
                valid = residual > 0
                total += np.where(valid, residual / stats["verschiebung_median_px"], 0.0)
                counts += valid
                used.append(f"{other - index:+d}({stats['verschiebung_median_px']:.0f}px)")

        if not used or counts.max() == 0:
            print(f"  {target.stem}: kein brauchbares Paar", flush=True)
            continue

        merged = np.divide(total, counts, out=np.zeros_like(total), where=counts > 0)
        # Fill holes with no contribution at all with the image mean, so that the
        # later smoothing does not drag them into their surroundings.
        if (counts == 0).any():
            merged[counts == 0] = float(merged[counts > 0].mean())
        if args.destripe > 1:
            merged = destripe(merged, args.destripe)
        merged = cv2.GaussianBlur(merged, (0, 0), args.smooth).astype(np.float32)
        np.save(cache_dir / f"{name}__{target.stem}.npy", merged)
        # The homography fits a plane that only approximates the ground -- what
        # remains is a large-scale gradient across the image. Subtracted for the
        # preview; the raw map is what gets stored, so that downstream scripts can
        # decide for themselves.
        trend = cv2.GaussianBlur(merged, (0, 0), args.detrend_sigma)
        surface = normalize(merged - trend)
        cv2.imwrite(str(preview_dir / f"{target.stem}_parallax.jpg"),
                    cv2.applyColorMap((surface * 255).astype(np.uint8), cv2.COLORMAP_TURBO))
        cv2.imwrite(str(preview_dir / f"{target.stem}_hillshade.jpg"), (hillshade(surface) * 255).astype(np.uint8))
        written += 1
        print(f"  {target.stem}: {len(used)} Paare {' '.join(used)}", flush=True)

    print(f"\n{written} Hoehenkarten -> {cache_dir}")
    print(f"Frames -> {frames_dir}")


if __name__ == "__main__":
    main()
