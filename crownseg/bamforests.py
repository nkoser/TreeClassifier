"""BAMFORESTS as a training source for individual tree crowns.

The dataset (Troles et al. 2024, Remote Sensing 16(11), 1935; CC BY-NC-SA 4.0)
supplies what the project had been missing: real, hand-digitised instance
boundaries. 2456 tiles of 2048x2048 px from four forest areas around Bamberg,
92,445 crown polygons in COCO format, a single class `tree`.

    train   1439 tiles  58,228 crowns   Stadtwald, Tretzendorf
    val      382 tiles  15,177 crowns   Stadtwald, Tretzendorf
    test1    313 tiles   6,720 crowns   Hain          <- unseen area
    test2    322 tiles  12,320 crowns   Stadtwald, Tretzendorf

`test1` is the only real transfer test: Hain occurs in neither training nor
validation. `test2` only measures how well the known areas fit -- report both
numbers separately, otherwise the result looks better than it is.

Difference from `crownnet.py`: there the polygons were merged into a single
label map (`fillPoly` with a running index), where overlapping crowns overwrite
each other -- with interlocking broadleaves that is the rule, not the exception.
Here every crown stays its own mask; nothing is lost.

Preparation (once, ~2456 TIFFs of 16 MB each):

    python crownseg/bamforests.py --split all

writes `<stem>.jpg` at original resolution per split, plus one shared
`annotations.json` (stem -> list of polygons). The TIFFs themselves have four
bands; the fourth is the alpha channel of the orthomosaic and is discarded.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

# OpenCV starts its own thread pool per process. In forked DataLoader workers
# that can deadlock against the parent process pool -- two training runs hung on
# this after 17 and 16 epochs (main and worker processes all in `do_poll`, GPU
# idle, while SLURM kept reporting RUNNING). One thread per worker is plenty
# anyway; the parallelism comes from the number of workers.
cv2.setNumThreads(0)

BAMFORESTS = Path(f"/scratch/shared/{os.environ.get('USER', 'nik')}/data/bamforests")

SPLITS = {
    "train": ("instances_tree_train2023.json", "train2023"),
    "val": ("instances_tree_eval2023.json", "val2023"),
    "test1": ("instances_tree_TestSet12023.json", "test2023/Test-Set-1"),
    "test2": ("instances_tree_TestSet22023.json", "test2023/Test-Set-2"),
}


# --------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------- #


def _convert_tile(job: tuple[str, str]) -> str | None:
    source, target = Path(job[0]), Path(job[1])
    if target.exists():
        return target.stem
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    cv2.imwrite(str(target), image[:, :, :3], [cv2.IMWRITE_JPEG_QUALITY, 95])
    return target.stem


def prepare_split(root: Path, split: str, out_root: Path, workers: int) -> int:
    """Rewrite the TIFF tiles as JPEG and store the polygons per tile."""
    annotation_file, image_dir = SPLITS[split]
    data = json.loads((root / "coco2048" / "annotations" / annotation_file).read_text())
    image_root = root / "coco2048" / image_dir
    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    names = {info["id"]: Path(info["file_name"]).stem for info in data["images"]}
    polygons: dict[str, list[list[float]]] = {name: [] for name in names.values()}
    for annotation in data["annotations"]:
        stem = names.get(annotation["image_id"])
        if stem is not None:
            # Always exactly one ring per crown -- checked, no multipolygon in the set.
            polygons[stem].append(annotation["segmentation"][0])

    jobs = [
        (str(image_root / f"{stem}.tif"), str(out_dir / f"{stem}.jpg"))
        for stem in names.values()
        if (image_root / f"{stem}.tif").exists()
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        written = [stem for stem in pool.map(_convert_tile, jobs, chunksize=4) if stem]

    index = {stem: polygons[stem] for stem in written if polygons.get(stem)}
    (out_dir / "annotations.json").write_text(json.dumps(index))
    print(f"{split:6s} {len(index):5d} Kacheln  {sum(len(v) for v in index.values()):6d} Kronen -> {out_dir}")
    return len(index)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


def load_tile(directory: Path, stem: str, index: dict) -> tuple[np.ndarray, list[np.ndarray]]:
    """A whole tile plus its polygons -- for evaluation and for looking at."""
    image = cv2.cvtColor(cv2.imread(str(directory / f"{stem}.jpg")), cv2.COLOR_BGR2RGB)
    rings = [np.asarray(p, dtype=np.float32).reshape(-1, 2) for p in index[stem]]
    return image, rings


def masks_from_rings(rings: list[np.ndarray], height: int, width: int,
                     min_area: int, min_visible: float) -> tuple[np.ndarray, np.ndarray]:
    """Rasterise polygons into individual masks and drop crowns that are cut off.

    A crown of which only a sliver lies inside the crop is harmful as a target:
    the network learned to take fragments for complete crowns. So everything
    below `min_visible` of the original area is dropped.
    """
    masks, boxes = [], []
    for ring in rings:
        points = np.round(ring).astype(np.int32)
        full_area = abs(cv2.contourArea(points))
        if full_area < min_area:
            continue
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        if x1 <= 0 or y1 <= 0 or x0 >= width or y0 >= height:
            continue

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 1)
        visible = int(mask.sum())
        if visible < min_area or visible < min_visible * full_area:
            continue

        ys, xs = np.nonzero(mask)
        masks.append(mask)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])

    if not masks:
        return np.zeros((0, height, width), np.uint8), np.zeros((0, 4), np.float32)
    return np.stack(masks), np.asarray(boxes, dtype=np.float32)


class CrownCrops(torch.utils.data.Dataset):
    """Random crops of a tile, with one mask per crown.

    The scale is jittered during training (`scale_jitter`). On BAMFORESTS itself
    that costs hardly any accuracy, but it is the only precaution for the later
    application to our own drone frames, whose GSD is only roughly known.
    """

    def __init__(self, root: Path, split: str, crop: int, length: int, augment: bool,
                 scale_jitter: tuple[float, float] = (1.0, 1.0),
                 min_area: int = 400, min_visible: float = 0.35,
                 depth_dir: Path | None = None, depth_only: bool = False) -> None:
        self.directory = root / split
        self.index = json.loads((self.directory / "annotations.json").read_text())
        self.stems = sorted(self.index)
        self.crop, self.length, self.augment = crop, length, augment
        self.scale_jitter, self.min_area, self.min_visible = scale_jitter, min_area, min_visible
        self.depth_dir = depth_dir
        # The height map only, copied three times. That leaves the pretrained
        # input convolution unchanged, and the comparison against the RGB run
        # really measures only the information content of the depth.
        self.depth_only = depth_only

    def __len__(self) -> int:
        return self.length

    def _sample(self, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        stem = self.stems[int(rng.integers(len(self.stems)))]
        image = cv2.cvtColor(cv2.imread(str(self.directory / f"{stem}.jpg")), cv2.COLOR_BGR2RGB)
        if self.depth_dir is not None:
            depth = cv2.imread(str(self.depth_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
            image = np.dstack([depth] * 3) if self.depth_only else np.dstack([image, depth])
        height, width = image.shape[:2]

        scale = float(rng.uniform(*self.scale_jitter))
        window = int(round(self.crop / scale))
        window = max(64, min(window, min(height, width)))
        y = int(rng.integers(0, height - window + 1))
        x = int(rng.integers(0, width - window + 1))

        patch = image[y : y + window, x : x + window]
        rings = [np.asarray(p, np.float32).reshape(-1, 2) - (x, y) for p in self.index[stem]]
        if window != self.crop:
            factor = self.crop / window
            patch = cv2.resize(patch, (self.crop, self.crop),
                               interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR)
            rings = [ring * factor for ring in rings]

        if self.augment:
            # Brightness, contrast, colour cast. Without this the training loss
            # fell to a sixth of the validation loss within two epochs -- the
            # network was memorising the exposure of the two training areas.
            # Colour channels only -- depth has no exposure, and a brightness
            # perturbation on it would be a height perturbation. In depth-only
            # mode it is therefore dropped entirely.
            if self.depth_only:
                gain, bias, tint = 1.0, 0.0, np.ones(3)
            else:
                gain = rng.uniform(0.8, 1.25)
                bias = rng.uniform(-25, 25)
                tint = rng.uniform(0.92, 1.08, size=3)
            colour = np.clip(patch[..., :3].astype(np.float32) * gain * tint + bias, 0, 255).astype(np.uint8)
            patch = np.dstack([colour, patch[..., 3:]]) if patch.shape[2] > 3 else colour

            k = int(rng.integers(4))
            flip = rng.random() < 0.5
            patch = np.rot90(patch, k, (0, 1))
            if flip:
                patch = patch[:, ::-1]
            size = self.crop
            for _ in range(k):
                rings = [np.stack([ring[:, 1], size - 1 - ring[:, 0]], axis=1) for ring in rings]
            if flip:
                rings = [np.stack([size - 1 - ring[:, 0], ring[:, 1]], axis=1) for ring in rings]

        masks, boxes = masks_from_rings(rings, self.crop, self.crop, self.min_area, self.min_visible)
        return np.ascontiguousarray(patch), masks, boxes

    def __getitem__(self, index: int):
        seed = index if not self.augment else (torch.initial_seed() + index) % (2**32)
        rng = np.random.default_rng(seed)
        for _ in range(8):  # redraw empty crops (paths, clearings)
            patch, masks, boxes = self._sample(rng)
            if len(boxes):
                break

        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "masks": torch.from_numpy(masks),
        }
        return torch.from_numpy(patch.transpose(2, 0, 1).copy()).float() / 255.0, target


def collate(batch):
    return tuple(zip(*batch))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=BAMFORESTS)
    parser.add_argument("--out", type=Path, default=BAMFORESTS / "crownseg")
    parser.add_argument("--split", default="all", choices=("all", *SPLITS))
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = list(SPLITS) if args.split == "all" else [args.split]
    for split in splits:
        prepare_split(args.root, split, args.out, args.workers)


if __name__ == "__main__":
    main()
