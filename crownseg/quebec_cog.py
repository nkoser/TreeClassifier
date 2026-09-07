"""Trainingsausschnitte mit frei waehlbarem Bildfeld direkt aus dem Orthomosaik.

Die aufbereiteten Kacheln haben eine feste Groesse von 2048 px. Damit laesst sich
der Massstab beim Training nur begrenzt aufweiten: das groesste moegliche Fenster
ist die Kachel selbst, was bei 640 px Eingabe rund 5.4 cm/px ergibt. Nik's
urbane Aufnahmen liegen aber bei etwa 20 cm/px -- gemessen an den faelschlich
segmentierten Autos, die mit 18 bis 28 px Laenge bei 4.5 m Fahrzeuglaenge auf
17 bis 25 cm/px fuehren.

Quebec liefert dagegen die vollstaendigen Orthomosaike (rund 40 000 x 42 000 px
je Zone). Daraus laesst sich ein Fenster beliebiger Groesse schneiden, und weil
die COGs Uebersichtsstufen bis 1/128 mitbringen, kostet ein 7500-px-Fenster auf
640 px heruntergelesen nur 7 ms -- weniger als das Lesen einer JPEG-Kachel.

Die Grenze ist eine andere: **EoMT hat 200 feste Anfragen.** Bei rund 400 bis 440
Kronen je Hektar passen in ein Bildfeld von 0.31 ha etwa 150 Kronen, das
entspricht 8.7 cm/px. Weiter aufgeweitet enthaelt das Bild mehr Baeume, als das
Modell ueberhaupt ausgeben kann, und die ueberzaehligen zaehlen im Training als
verfehlt. Ausschnitte mit zu vielen Kronen werden deshalb verworfen und neu
gezogen, statt dem Netz ein unloesbares Ziel zu geben.

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
    """Zufaellige Ausschnitte aus dem Orthomosaik, Bildfeld zufaellig gewaehlt.

    `gsd_range` gibt den Massstab an, den das Modell sehen soll, in cm je Pixel
    der Eingabe -- nicht die Fenstergroesse. Das ist die Groesse, auf die es
    ankommt: sie bestimmt, wie gross eine Krone im Eingabebild erscheint.
    """

    def __init__(self, root: Path, zones: list[str], date: str, size: int, length: int,
                 gsd_range: tuple[float, float] = (2.7, 8.7), augment: bool = True,
                 min_area_px: int = 60, max_instances: int = 150, tries: int = 12) -> None:
        self.root, self.date, self.size, self.length = Path(root), date, size, length
        self.gsd_range, self.augment = gsd_range, augment
        self.min_area_px, self.max_instances, self.tries = min_area_px, max_instances, tries
        self.zones = zones
        self._sources: dict[int, list] = {}   # je Prozess eigene Handles

        # Polygone einmal in Weltkoordinaten laden; die Umrechnung ins Fenster
        # passiert spaeter und ist nur eine Verschiebung mit Skalierung.
        self.rings: dict[str, list[np.ndarray]] = {}
        for zone in zones:
            polygons = self.root / f"Z{zone[-1]}_polygons.gpkg"
            self.rings[zone] = [r for r, _ in load_polygons(polygons, f"Z{zone[-1]}_polygons")]

    def _open(self):
        """rasterio-Handles sind nicht fork-sicher -- je Worker eigene oeffnen."""
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
            # Massstab logarithmisch ziehen: die Spanne 2.7 bis 8.7 cm ist ein
            # Faktor 3, linear gezogen kaemen weite Bildfelder zu selten vor.
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
            if (patch.max(axis=2) == 0).mean() > 0.05:   # Mosaikrand
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
