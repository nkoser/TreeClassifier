"""Diagnose: Wie stark haengt die Artvorhersage vom gewaehlten Bildmassstab ab?

Der Quebec-Trees-Checkpoint hat immer Ausschnitte von 9.73 m Kantenlaenge gesehen.
Fuer eigene Frames ohne Georeferenzierung ist der Massstab unbekannt. Dieses Skript
klassifiziert dieselben Detektionen bei verschiedenen angenommenen Ausschnitts-
groessen (in Quell-Pixeln) und zeigt, wie sich Vorhersage und Konfidenz aendern.

Praktisch heisst das: welcher Pixel-Ausschnitt entspricht 9.73 m in deinen Bildern?
Die Groesse mit den plausibelsten/stabilsten Vorhersagen ist ein Indiz -- ersetzt
aber keine echte Massstabsangabe (Flughoehe + Bildwinkel oder Referenzstrecke).

Beispiel:
    python scale_sweep.py --frames /cold/Mahfuz/chosen_frames/pines/frame_000006.jpg \
        --crop-sizes 100 150 200 300 400 600 --n-trees 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from infer_species import (
    QUEBEC_TREES_EXCLUDE,
    REPO_ROOT,
    TRAIN_FOOTPRINT_M,
    build_detector,
    build_dinovtree,
    classify_crops,
    crop_centered,
    load_class_names,
    resolve_device,
    short_name,
    to_model_input,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=Path, nargs="+", required=True)
    parser.add_argument("--crop-sizes", type=int, nargs="+", default=[100, 150, 200, 300, 450, 650])
    parser.add_argument("--n-trees", type=int, default=8, help="Detektionen pro Frame (nach Score).")
    parser.add_argument("--min-score", type=float, default=0.4)
    parser.add_argument("--ckpt", type=Path, default=REPO_ROOT / "checkpoints" / "dinovtreeb_quebectrees.pth")
    parser.add_argument("--categories", type=Path, default=REPO_ROOT / "third_party" / "quebec_trees_categories.json")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "scale_sweep.csv")
    parser.add_argument("--montage-dir", type=Path, default=REPO_ROOT / "results" / "scale_montages")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_names = load_class_names(args.categories, QUEBEC_TREES_EXCLUDE)
    device = resolve_device(args.device)
    print(f"Device: {device}")
    model = build_dinovtree(args.ckpt, n_classes=len(class_names), max_height=30.0, device=device)
    detector = build_detector("weecology/deepforest-tree")
    args.montage_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for frame_path in args.frames:
        image = cv2.imread(str(frame_path))[:, :, ::-1]
        boxes = detector.predict_tile(image=image.astype("float32"), patch_size=400, patch_overlap=0.1)
        boxes = boxes[boxes["score"] >= args.min_score].nlargest(args.n_trees, "score").reset_index(drop=True)
        centers = [((r.xmin + r.xmax) / 2, (r.ymin + r.ymax) / 2) for r in boxes.itertuples()]

        montage_rows = []
        for crop_px in args.crop_sizes:
            patches = [crop_centered(image, cx, cy, crop_px) for cx, cy in centers]
            crops = np.stack([to_model_input(p) for p in patches])
            probabilities, heights = classify_crops(model, crops, args.batch_size, device)
            best = probabilities.argmax(axis=1)

            for i, (cx, cy) in enumerate(centers):
                rows.append(
                    {
                        "frame": frame_path.name,
                        "tree": i,
                        "crop_px": crop_px,
                        "implied_gsd_cm": 100 * TRAIN_FOOTPRINT_M / crop_px,
                        "species": class_names[best[i]],
                        "prob": probabilities[i, best[i]],
                        "height_m_pred": heights[i],
                    }
                )
            montage_rows.append(
                np.hstack([cv2.resize(p, (160, 160), interpolation=cv2.INTER_AREA) for p in patches])
            )

        montage = np.vstack(montage_rows)
        cv2.imwrite(str(args.montage_dir / f"{frame_path.stem}_scales.jpg"), montage[:, :, ::-1])

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    for frame, group in df.groupby("frame"):
        print(f"\n=== {frame} ===")
        table = group.pivot_table(
            index="crop_px", columns="tree", values="species", aggfunc="first"
        ).map(short_name)
        table["mean_prob"] = group.groupby("crop_px")["prob"].mean().round(2)
        table["mean_h"] = group.groupby("crop_px")["height_m_pred"].mean().round(1)
        table.insert(0, "implied_gsd_cm", group.groupby("crop_px")["implied_gsd_cm"].first().round(1))
        print(table.to_string())


if __name__ == "__main__":
    main()
