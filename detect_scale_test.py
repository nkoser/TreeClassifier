"""Test der Massstabs-Hypothese fuer die Detektion.

DeepForest ist auf NEON-Daten bei ~10 cm/px trainiert. Wenn die Frames aus
geringerer Flughoehe stammen als angenommen, erscheinen die Kronen dem Detektor
zu gross und er zerlegt sie in viele kleine Kandidaten. Dieses Skript prueft das
direkt: dasselbe Bild wird um verschiedene Faktoren verkleinert (das entspricht
einem hoeheren Flug), detektiert und die Boxen werden in Originalkoordinaten
zurueckgerechnet.

Erwartung, falls die Hypothese stimmt: mit kleinerem Faktor sinkt die Anzahl
Detektionen deutlich, die Boxen werden (in Originalpixeln) groesser und die
Scores steigen -- bis zu einem Optimum, jenseits dessen Baeume verschwinden.

Beispiel:
    python detect_scale_test.py --scales 1.0 0.7 0.5 0.35 0.25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from infer_species import IMAGE_SUFFIXES, REPO_ROOT, build_detector

DEFAULT_FRAMES = [
    "pines/frame_000006.jpg",
    "dense/frame_000073.jpg",
    "mixed1/frame_000594.jpg",
    "80m/frame_000297.jpg",
]


def detect_at_scale(detector, image_rgb: np.ndarray, scale: float, args) -> pd.DataFrame:
    """Detektiert auf dem skalierten Bild und rechnet die Boxen zurueck."""
    if scale != 1.0:
        resized = cv2.resize(
            image_rgb, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )
    else:
        resized = image_rgb

    boxes = detector.predict_tile(
        image=resized.astype("float32"),
        patch_size=args.patch_size,
        patch_overlap=args.patch_overlap,
        iou_threshold=args.iou,
    )
    if boxes is None or len(boxes) == 0:
        return pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "score", "box_px_detector", "box_px_original"])

    boxes = boxes.copy()
    # Groesse, wie der Detektor sie gesehen hat (entscheidend fuer seinen Prior) ...
    boxes["box_px_detector"] = np.maximum(boxes["xmax"] - boxes["xmin"], boxes["ymax"] - boxes["ymin"])
    # ... und dieselbe Box in Originalkoordinaten, damit Massstaebe vergleichbar sind.
    for column in ("xmin", "ymin", "xmax", "ymax"):
        boxes[column] = boxes[column] / scale
    boxes["box_px_original"] = boxes["box_px_detector"] / scale
    return boxes


def draw(image_bgr: np.ndarray, boxes: pd.DataFrame, caption: str, out_path: Path) -> None:
    canvas = image_bgr.copy()
    for row in boxes.itertuples():
        color = (80, 200, 80) if row.score >= 0.5 else (60, 220, 240) if row.score >= 0.35 else (60, 150, 240)
        cv2.rectangle(canvas, (int(row.xmin), int(row.ymin)), (int(row.xmax), int(row.ymax)), color, 2)
    cv2.rectangle(canvas, (0, 0), (620, 34), (0, 0, 0), -1)
    cv2.putText(canvas, caption, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--frames", nargs="*", default=DEFAULT_FRAMES, help="Pfade relativ zu --input.")
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.7, 0.5, 0.35, 0.25])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_scaletest")
    parser.add_argument("--patch-size", type=int, default=400)
    parser.add_argument("--patch-overlap", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.15)
    parser.add_argument("--detector-model", default="weecology/deepforest-tree")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = build_detector(args.detector_model)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for relative in args.frames:
        frame_path = args.input / relative
        if frame_path.suffix.lower() not in IMAGE_SUFFIXES or not frame_path.exists():
            print(f"uebersprungen: {frame_path}")
            continue

        image_bgr = cv2.imread(str(frame_path))
        image_rgb = image_bgr[:, :, ::-1]
        out_folder = args.out / frame_path.parent.name
        out_folder.mkdir(parents=True, exist_ok=True)

        for scale in args.scales:
            boxes = detect_at_scale(detector, image_rgb, scale, args)
            strong = boxes[boxes["score"] >= 0.5]
            caption = (
                f"scale {scale:.2f} ({int(image_rgb.shape[1] * scale)}px breit): "
                f"{len(boxes)} Detektionen, Box med {boxes['box_px_original'].median():.0f} px"
                if len(boxes)
                else f"scale {scale:.2f}: keine Detektion"
            )
            draw(image_bgr, boxes, caption, out_folder / f"{frame_path.stem}_s{scale:.2f}.jpg")

            rows.append(
                {
                    "frame": relative,
                    "scale": scale,
                    "detektionen": len(boxes),
                    "score_median": round(boxes["score"].median(), 3) if len(boxes) else float("nan"),
                    "score_ab_0.5": len(strong),
                    "box_detektor_px": round(boxes["box_px_detector"].median(), 1) if len(boxes) else float("nan"),
                    "box_original_px": round(boxes["box_px_original"].median(), 1) if len(boxes) else float("nan"),
                }
            )
            print(f"  {relative} @ {scale:.2f}: {rows[-1]}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "scale_test.csv", index=False)

    print("\n=== Detektionen pro Frame und Skalierung ===")
    print(df.pivot(index="scale", columns="frame", values="detektionen").to_string())
    print("\n=== Median-Boxgroesse in Originalpixeln (so gross ist die gefundene Krone wirklich) ===")
    print(df.pivot(index="scale", columns="frame", values="box_original_px").to_string())
    print("\n=== Median-Boxgroesse in Detektorpixeln (was DeepForest gesehen hat) ===")
    print(df.pivot(index="scale", columns="frame", values="box_detektor_px").to_string())
    print("\n=== Detektionen mit Score >= 0.5 ===")
    print(df.pivot(index="scale", columns="frame", values="score_ab_0.5").to_string())


if __name__ == "__main__":
    main()
