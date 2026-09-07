"""Quebec Trees as a second training source -- smaller crowns than BAMFORESTS.

BAMFORESTS annotates crowns with a median of 4.78 m (Stadtwald) to 6.66 m
(Hain). A model trained on it carries that notion of size along and lumps
several trees into one mask in fine-crowned stands -- clearly visible on our
pine frames. The image scale does not help against it: at x1.2 as at x2.8 the
model predicts crowns with a median of 2.6 m, so it follows its learned notion
and not the resolution.

Quebec Trees (Cloutier et al. 2023, CC-BY-4.0) covers the missing range.
Measured on the polygons themselves, not taken from the publication:

    22,933 crowns | median 4.09 m | p5 1.82 m | p95 8.54 m | GSD 1.64 cm/px

    Abies balsamea   2.78 m   n=2895     Acer rubrum      4.23 m   n=5857
    Thuja occid.     2.97 m   n=1510     Betula papyr.    4.84 m   n=5894
    Picea spp.       3.02 m   n= 599     Pinus strobus    7.25 m   n= 569

At 2.8 to 3.0 m the conifers are in the range of the pines in `pines`.

Difference from the BAMFORESTS preparation: there, finished COCO tiles were
available. Here there are three large orthomosaics as cloud-optimized GeoTIFF and
the annotations as a GeoPackage in UTM coordinates -- so tiling and the
conversion from world to pixel coordinates come on top. What is written out is
exactly the format of `bamforests.py` (one JPEG per tile plus
`annotations.json`), so that everything downstream keeps working unchanged.

Split: zone 3 is held back completely as the test area -- the same principle as
Hain in BAMFORESTS, where a spatially separate area provides the only honest
transfer test.

    python crownseg/quebec.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import struct
from pathlib import Path

import cv2
import numpy as np

ZONE_SPLITS = {"zone1": "train", "zone2": "val", "zone3": "test"}


def wkb_rings(blob: bytes) -> list[np.ndarray]:
    """Rings of a GeoPackage blob as world coordinates.

    The header is 8 bytes plus an optional envelope; normal WKB follows. Only the
    outer ring of each polygon is read -- there are no holes in a crown in this
    dataset, and the target masks do not know them anyway.
    """
    flags = blob[3]
    envelope = {0: 0, 1: 4, 2: 6, 3: 6, 4: 8}[(flags >> 1) & 0x07]
    offset = 8 + 8 * envelope

    endian = "<" if blob[offset] == 1 else ">"
    gtype, = struct.unpack(endian + "I", blob[offset + 1 : offset + 5])
    offset += 5
    rings: list[np.ndarray] = []

    def read_ring(offset: int) -> int:
        count, = struct.unpack(endian + "I", blob[offset : offset + 4])
        offset += 4
        points = np.frombuffer(blob, dtype=endian + "f8", count=2 * count, offset=offset)
        rings.append(points.reshape(-1, 2))
        return offset + 16 * count

    base = gtype % 1000
    if base == 3:
        count, = struct.unpack(endian + "I", blob[offset : offset + 4])
        offset += 4
        for index in range(count):
            offset = read_ring(offset)
            if index == 0:
                continue
            rings.pop()  # discard inner rings
    elif base == 6:
        polygons, = struct.unpack(endian + "I", blob[offset : offset + 4])
        offset += 4
        for _ in range(polygons):
            offset += 5
            count, = struct.unpack(endian + "I", blob[offset : offset + 4])
            offset += 4
            for index in range(count):
                offset = read_ring(offset)
                if index > 0:
                    rings.pop()
    return rings


def load_polygons(path: Path, table: str) -> list[tuple[np.ndarray, str]]:
    con = sqlite3.connect(path)
    out = []
    for geometry, label in con.execute(f"SELECT Shape, Label FROM {table}"):
        for ring in wkb_rings(geometry):
            out.append((ring, label))
    con.close()
    return out


def prepare_zone(cog: Path, polygons: Path, table: str, out_dir: Path,
                 tile: int, overlap: int, min_crowns: int) -> tuple[int, int]:
    import rasterio

    rings = load_polygons(polygons, table)
    out_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, list] = {}
    written = crowns = 0

    with rasterio.open(cog) as src:
        transform = ~src.transform  # world -> pixel
        # All crowns once in pixel coordinates, after that only shifted.
        in_pixels = []
        for ring, label in rings:
            cols, rows = transform * (ring[:, 0], ring[:, 1])
            in_pixels.append(np.stack([cols, rows], axis=1).astype(np.float32))

        boxes = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]
                          for p in in_pixels], dtype=np.float32)
        step = tile - overlap

        for y0 in range(0, src.height - tile + 1, step):
            for x0 in range(0, src.width - tile + 1, step):
                # Crowns that lie completely inside the tile. Cut ones are harmful
                # as a target -- the same rule as for BAMFORESTS.
                inside = np.flatnonzero(
                    (boxes[:, 0] >= x0) & (boxes[:, 1] >= y0)
                    & (boxes[:, 2] < x0 + tile) & (boxes[:, 3] < y0 + tile))
                if len(inside) < min_crowns:
                    continue

                window = rasterio.windows.Window(x0, y0, tile, tile)
                patch = src.read((1, 2, 3), window=window)
                patch = np.transpose(patch, (1, 2, 0))
                # The margins of the orthomosaics are black or transparent.
                if (patch.max(axis=2) == 0).mean() > 0.05:
                    continue

                stem = f"{out_dir.name}_{x0:06d}_{y0:06d}"
                cv2.imwrite(str(out_dir / f"{stem}.jpg"),
                            cv2.cvtColor(patch, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                index[stem] = [(in_pixels[i] - (x0, y0)).ravel().tolist() for i in inside]
                written += 1
                crowns += len(inside)

    (out_dir / "annotations.json").write_text(json.dumps(index))
    return written, crowns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path,
                        default=Path("/scratch/shared/nik/data/quebec_trees/quebec_trees_dataset_2021-09-02"))
    parser.add_argument("--date", default="2021-09-02")
    parser.add_argument("--out", type=Path,
                        default=Path("/scratch/shared/nik/data/quebec_trees/crownseg"))
    parser.add_argument("--tile", type=int, default=2048, help="As in BAMFORESTS.")
    parser.add_argument("--overlap", type=int, default=1024)
    parser.add_argument("--min-crowns", type=int, default=5, help="Skip empty tiles.")
    args = parser.parse_args()

    for zone, split in ZONE_SPLITS.items():
        cog = args.root / args.date / zone / f"{args.date}-sbl-{zone.replace('zone', 'z')}-rgb-cog.tif"
        polygons = args.root / f"Z{zone[-1]}_polygons.gpkg"
        if not cog.exists() or not polygons.exists():
            print(f"{zone}: fehlt ({cog.name})")
            continue
        written, crowns = prepare_zone(cog, polygons, f"Z{zone[-1]}_polygons",
                                       args.out / split, args.tile, args.overlap, args.min_crowns)
        print(f"{zone} -> {split:5s}: {written:4d} Kacheln, {crowns:6d} Kronen", flush=True)

    print(f"\n{args.out}")


if __name__ == "__main__":
    main()
