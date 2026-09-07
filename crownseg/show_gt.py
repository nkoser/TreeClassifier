"""Make the BAMFORESTS ground truth visible -- image, outlines, instances.

No model, no GPU. It exists to let you look at what is actually annotated: where
the boundaries lie, how large the crowns are and what share of the tile carries
no annotation at all.

    python crownseg/show_gt.py --split test1 --tiles 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bamforests as bam  # noqa: E402

# Repeats after 20 crowns, which is enough -- neighbours get different colours.
PALETTE = np.array([
    (231, 76, 60), (46, 204, 113), (52, 152, 219), (241, 196, 15), (155, 89, 182),
    (26, 188, 156), (230, 126, 34), (52, 73, 94), (149, 165, 166), (211, 84, 0),
    (39, 174, 96), (41, 128, 185), (243, 156, 18), (142, 68, 173), (22, 160, 133),
    (192, 57, 43), (127, 140, 141), (44, 62, 80), (46, 134, 193), (203, 67, 53),
], dtype=np.uint8)


def render(image_bgr: np.ndarray, rings: list[np.ndarray]) -> np.ndarray:
    """Three views side by side: raw, outlines, filled instances."""
    outlines = image_bgr.copy()
    filled = image_bgr.copy()
    overlay = image_bgr.copy()

    for number, ring in enumerate(rings):
        points = np.round(ring).astype(np.int32)
        colour = tuple(int(v) for v in PALETTE[number % len(PALETTE)])
        cv2.fillPoly(overlay, [points], colour)
        cv2.polylines(outlines, [points], True, (80, 230, 120), 3)
        cv2.polylines(overlay, [points], True, (255, 255, 255), 2)

    filled = cv2.addWeighted(overlay, 0.45, filled, 0.55, 0)

    covered = np.zeros(image_bgr.shape[:2], np.uint8)
    for ring in rings:
        cv2.fillPoly(covered, [np.round(ring).astype(np.int32)], 1)
    share = covered.mean()

    panels = [(image_bgr, "Kachel"),
              (outlines, f"{len(rings)} Kronen, Umrisse"),
              (filled, f"Instanzen, {share:.0%} der Flaeche annotiert")]
    for panel, caption in panels:
        cv2.rectangle(panel, (0, 0), (760, 46), (0, 0, 0), -1)
        cv2.putText(panel, caption, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return np.hstack([p for p, _ in panels])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", type=Path, default=bam.BAMFORESTS / "crownseg")
    parser.add_argument("--split", default="test1",
                        help="Folder name under --prepared; BAMFORESTS uses test1/test2, Quebec test.")
    parser.add_argument("--stems", nargs="*", default=None, help="Specific tiles instead of an even sample.")
    parser.add_argument("--tiles", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("results_crownseg/gt"))
    parser.add_argument("--scale", type=float, default=0.5, help="Shrink the output; 1.0 = full resolution.")
    parser.add_argument("--labels", type=Path, default=None,
                        help="Folder with <stem>_labels.png; added as a fourth view alongside.")
    args = parser.parse_args()

    directory = args.prepared / args.split
    index = json.loads((directory / "annotations.json").read_text())
    stems = args.stems or sorted(index)[:: max(1, len(index) // args.tiles)][: args.tiles]
    args.out.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        image, rings = bam.load_tile(directory, stem, index)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        panel = render(bgr, rings)

        if args.labels:
            predicted = cv2.imread(str(args.labels / f"{stem}_labels.png"), cv2.IMREAD_UNCHANGED)
            if predicted is not None:
                if predicted.shape[:2] != bgr.shape[:2]:
                    predicted = cv2.resize(predicted, bgr.shape[1::-1], interpolation=cv2.INTER_NEAREST)
                view = bgr.copy()
                tint = bgr.copy()
                for value in np.unique(predicted):
                    if value == 0:
                        continue
                    mask = predicted == value
                    tint[mask] = PALETTE[int(value) % len(PALETTE)]
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(view, contours, -1, (255, 255, 255), 2)
                view = cv2.addWeighted(tint, 0.45, view, 0.55, 0)
                caption = f"Vorhersage: {int(predicted.max())} Kronen"
                cv2.rectangle(view, (0, 0), (760, 46), (0, 0, 0), -1)
                cv2.putText(view, caption, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                panel = np.hstack([panel, view])
        if args.scale != 1.0:
            panel = cv2.resize(panel, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        path = args.out / f"{args.split}_{stem}_gt.jpg"
        cv2.imwrite(str(path), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  {stem}: {len(rings)} Kronen -> {path}", flush=True)


if __name__ == "__main__":
    main()
