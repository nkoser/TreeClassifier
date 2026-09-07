"""Look at the DeepForest detections only, without classification.

Draws all raw detections of a frame and highlights which of them survive the
filters from infer_species.py (--min-score, --min-box-px). That makes it
possible to judge whether the tree instances are right at all -- independently
of what DINOvTree later makes of them.

Colour = detector confidence:
    red < 0.2, orange < 0.35, yellow < 0.5, green >= 0.5
Thick boxes = survives the filters and would be classified.

Example:
    python detect_only.py --input /cold/Mahfuz/chosen_frames --out results_detect
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from infer_species import IMAGE_SUFFIXES, REPO_ROOT, build_detector

SCORE_COLORS = [  # (threshold, BGR)
    (0.20, (60, 60, 220)),    # red
    (0.35, (60, 150, 240)),   # orange
    (0.50, (60, 220, 240)),   # yellow
    (1.01, (80, 200, 80)),    # green
]


def color_for_score(score: float) -> tuple[int, int, int]:
    for threshold, color in SCORE_COLORS:
        if score < threshold:
            return color
    return SCORE_COLORS[-1][1]


def draw_detections(image_path: Path, boxes: pd.DataFrame, out_path: Path, show_scores: bool) -> None:
    image = cv2.imread(str(image_path))

    for row in boxes.itertuples():
        color = color_for_score(row.score)
        thickness = 2 if row.kept else 1
        cv2.rectangle(image, (int(row.xmin), int(row.ymin)), (int(row.xmax), int(row.ymax)), color, thickness)
        if show_scores and row.kept:
            cv2.putText(
                image, f"{row.score:.2f}", (int(row.xmin), int(row.ymin) - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
            )

    kept = int(boxes["kept"].sum())
    caption = f"{len(boxes)} Detektionen, {kept} nach Filter"
    cv2.rectangle(image, (0, 0), (430, 34), (0, 0, 0), -1)
    cv2.putText(image, caption, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_detect")
    parser.add_argument("--frames-per-folder", type=int, default=0, help="0 = every frame.")
    parser.add_argument("--min-score", type=float, default=0.35)
    parser.add_argument("--min-box-px", type=float, default=25.0)
    parser.add_argument("--max-trees-per-frame", type=int, default=150)
    parser.add_argument("--patch-size", type=int, default=400)
    parser.add_argument("--patch-overlap", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.15)
    parser.add_argument("--show-scores", action="store_true")
    parser.add_argument("--detector-model", default="weecology/deepforest-tree")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = build_detector(args.detector_model)
    args.out.mkdir(parents=True, exist_ok=True)

    all_boxes = []
    for folder in sorted(p for p in args.input.iterdir() if p.is_dir()):
        frames = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if args.frames_per_folder:
            frames = frames[: args.frames_per_folder]
        if not frames:
            continue

        out_folder = args.out / folder.name
        out_folder.mkdir(parents=True, exist_ok=True)

        for frame_path in frames:
            image = cv2.imread(str(frame_path))[:, :, ::-1]
            boxes = detector.predict_tile(
                image=image.astype("float32"),
                patch_size=args.patch_size,
                patch_overlap=args.patch_overlap,
                iou_threshold=args.iou,
            )
            if boxes is None or len(boxes) == 0:
                print(f"  {folder.name}/{frame_path.name}: keine Detektion")
                continue

            boxes = boxes.copy()
            boxes["box_w"] = boxes["xmax"] - boxes["xmin"]
            boxes["box_h"] = boxes["ymax"] - boxes["ymin"]
            passes = (
                (boxes["score"] >= args.min_score)
                & (boxes["box_w"] >= args.min_box_px)
                & (boxes["box_h"] >= args.min_box_px)
            )
            # The rank filter (--max-trees-per-frame) only applies after the thresholds.
            ranked = boxes[passes].sort_values("score", ascending=False).head(args.max_trees_per_frame).index
            boxes["kept"] = boxes.index.isin(ranked)

            draw_detections(frame_path, boxes, out_folder / f"{frame_path.stem}_detect.jpg", args.show_scores)

            boxes.insert(0, "frame", frame_path.name)
            boxes.insert(0, "folder", folder.name)
            all_boxes.append(boxes.drop(columns=[c for c in ("geometry", "image_path") if c in boxes]))

            print(
                f"  {folder.name}/{frame_path.name}: {len(boxes)} roh -> {int(boxes['kept'].sum())} behalten | "
                f"Score med {boxes['score'].median():.2f}, "
                f"Box med {boxes[['box_w', 'box_h']].max(axis=1).median():.0f} px"
            )

    if not all_boxes:
        print("Keine Detektionen.")
        return

    combined = pd.concat(all_boxes, ignore_index=True)
    combined.to_csv(args.out / "all_detections.csv", index=False)

    print(f"\n{len(combined)} Rohdetektionen, {int(combined['kept'].sum())} nach Filter -> {args.out}")
    print("\nScore-Verteilung (alle Rohdetektionen):")
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    print(pd.cut(combined["score"], bins).value_counts().sort_index().to_string())

    print("\nPro Ordner:")
    summary = combined.groupby("folder").agg(
        roh=("score", "count"),
        behalten=("kept", "sum"),
        score_median=("score", "median"),
        box_median_px=("box_w", "median"),
    ).round(2)
    print(summary.to_string())


if __name__ == "__main__":
    main()
