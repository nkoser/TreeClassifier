"""Training crops with a freely chosen field of view, straight from the orthomosaic.

The prepared tiles have a fixed size of 2048 px. That limits how far the scale
can be widened during training: the largest possible window is the tile itself,
which at a 640 px input gives about 5.4 cm/px. Our urban captures, however, are
at roughly 20 cm/px -- measured on the wrongly segmented cars, which at 18 to
28 px length and a 4.5 m vehicle length imply 17 to 25 cm/px.

Quebec, by contrast, supplies the complete orthomosaics (around 40,000 x 42,000
px per zone). A window of any size can be cut from them, and because the COGs
carry overview levels down to 1/128, a 7500 px window read down to 640 px costs
only 7 ms -- less than reading a JPEG tile.

The limit is a different one: **EoMT has 200 fixed queries.** At around 400 to
440 crowns per hectare, about 150 crowns fit into a field of view of 0.31 ha,
which corresponds to 8.7 cm/px. Widened further, the image contains more trees
than the model can output at all, and the surplus ones count as misses during
training. Crops with too many crowns are therefore discarded and redrawn,
instead of giving the network an unsolvable target.

    from quebec_cog import CogCrops
    CogCrops(root, zones=["zone1"], gsd_range=(2.7, 8.7), ...)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from quebec import load_polygons


class CogCrops(torch.utils.data.Dataset):
    """Random crops from the orthomosaic, with a randomly chosen field of view.

    `gsd_range` gives the scale the model is meant to see, in cm per input pixel
    -- not the window size. That is the quantity that matters: it determines how
    large a crown appears in the input image.
    """

    def __init__(self, root: Path, zones: list[str], date: str, size: int, length: int,
                 gsd_range: tuple[float, float] = (2.7, 8.7), augment: bool = True,
                 min_area_px: int = 60, max_instances: int = 150, tries: int = 12) -> None:
        self.root, self.date, self.size, self.length = Path(root), date, size, length
        self.gsd_range, self.augment = gsd_range, augment
        self.min_area_px, self.max_instances, self.tries = min_area_px, max_instances, tries
        self.zones = zones
        self._sources: dict[int, list] = {}   # separate handles per process

        # Load the polygons once in world coordinates; the conversion into the
        # window happens later and is only a shift with a scaling.
        self.rings: dict[str, list[np.ndarray]] = {}
        for zone in zones:
            polygons = self.root / f"Z{zone[-1]}_polygons.gpkg"
            self.rings[zone] = [r for r, _ in load_polygons(polygons, f"Z{zone[-1]}_polygons")]

    def _open(self):
        """rasterio handles are not fork-safe -- open separate ones per worker."""
        import rasterio

        key = os.getpid()
        if key not in self._sources:
            entries = []
            for zone in self.zones:
                path = next((self.root / self.date / zone).glob("*-cog.tif"))
                src = rasterio.open(path)
                inverse = ~src.transform
                boxes, pixels = [], []
                for ring in self.rings[zone]:
                    cols, rows = inverse * (ring[:, 0], ring[:, 1])
                    p = np.stack([cols, rows], axis=1).astype(np.float32)
                    pixels.append(p)
                    boxes.append([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()])
                entries.append((src, np.asarray(boxes, np.float32), pixels, abs(src.transform.a)))
            self._sources[key] = entries
        return self._sources[key]

    def __len__(self) -> int:
        return self.length

    def _sample(self, rng):
        import rasterio

        for _ in range(self.tries):
            src, boxes, pixels, gsd = self._sources[os.getpid()][int(rng.integers(len(self.zones)))]
            # Draw the scale logarithmically: the range 2.7 to 8.7 cm is a factor
            # of 3, and drawn linearly, wide fields of view would be too rare.
            gsd_model = float(np.exp(rng.uniform(*np.log(self.gsd_range)))) / 100.0
            window_px = int(round(gsd_model * self.size / gsd))
            if window_px > min(src.width, src.height):
                continue

            x0 = int(rng.integers(0, src.width - window_px))
            y0 = int(rng.integers(0, src.height - window_px))
            inside = np.flatnonzero(
                (boxes[:, 0] >= x0) & (boxes[:, 1] >= y0)
                & (boxes[:, 2] < x0 + window_px) & (boxes[:, 3] < y0 + window_px))
            if not len(inside) or len(inside) > self.max_instances:
                continue

            patch = src.read((1, 2, 3), window=rasterio.windows.Window(x0, y0, window_px, window_px),
                             out_shape=(3, self.size, self.size))
            patch = np.ascontiguousarray(np.transpose(patch, (1, 2, 0)))
            if (patch.max(axis=2) == 0).mean() > 0.05:   # mosaic margin
                continue

            factor = self.size / window_px
            rings = [(pixels[i] - (x0, y0)) * factor for i in inside]
            return patch, rings
        return None

    def __getitem__(self, index: int):
        from bamforests import masks_from_rings

        self._open()
        seed = index if not self.augment else (torch.initial_seed() + index) % (2**32)
        rng = np.random.default_rng(seed)

        sample = self._sample(rng)
        if sample is None:
            patch = np.zeros((self.size, self.size, 3), np.uint8)
            rings = []
        else:
            patch, rings = sample

        if self.augment and rings:
            k = int(rng.integers(4))
            flip = rng.random() < 0.5
            patch = np.rot90(patch, k, (0, 1))
            for _ in range(k):
                rings = [np.stack([r[:, 1], self.size - 1 - r[:, 0]], axis=1) for r in rings]
            if flip:
                patch = patch[:, ::-1]
                rings = [np.stack([self.size - 1 - r[:, 0], r[:, 1]], axis=1) for r in rings]
            gain, bias = rng.uniform(0.8, 1.25), rng.uniform(-25, 25)
            tint = rng.uniform(0.92, 1.08, size=3)
            patch = np.clip(patch.astype(np.float32) * gain * tint + bias, 0, 255).astype(np.uint8)

        masks, _ = masks_from_rings(rings, self.size, self.size, self.min_area_px, 0.35)
        return (torch.from_numpy(np.ascontiguousarray(patch).transpose(2, 0, 1).copy()).float() / 255.0,
                {"masks": torch.from_numpy(masks),
                 "labels": torch.ones(len(masks), dtype=torch.int64)})
