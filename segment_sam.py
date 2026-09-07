"""Kronenabgrenzung mit SAM (Segment Anything), automatische Maskengenerierung.

Anderer Ansatz als segment_trees.py: statt aus geschaetzter Tiefe eine Oberflaeche
zu bauen und sie zu partitionieren, segmentiert SAM entlang echter Bildkanten. Das
umgeht die Schwaeche des Tiefenmodells in kontrastarmen Bereichen -- SAM sieht die
Kronenraender direkt.

SAM erzeugt Masken auf allen Skalen gleichzeitig (Blatt, Ast, Krone, ganzer
Bestand). Die Arbeit steckt deshalb in der Auswahl:
  1. Flaechenfenster um die erwartete Kronengroesse.
  2. Kompaktheit -- Kronen sind halbwegs rund, Schattenbaender nicht.
  3. Ueberlappungsaufloesung: nach Score sortiert greedy annehmen, Masken mit
     hoher IoU zu bereits akzeptierten verwerfen (verhindert die Skalenstapel).

Optional wird die Tiefenkarte aus segment_trees.py als Zusatzfilter genutzt: eine
Krone sollte sich gegenueber ihrem Rand erheben.

Beispiel:
    python segment_sam.py --frames 100/frame_000537.jpg --crown-px 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from infer_species import IMAGE_SUFFIXES, REPO_ROOT, resolve_device

SAM_MODEL = "facebook/sam-vit-large"  # laut Ablation besser als vit-huge, siehe README


def build_generator(model_id: str, device, args):
    """Baut die Maskengenerator-Pipeline.

    pred_iou_thresh und stability_score_thresh sind die eigentlichen Stellschrauben
    fuer die Ausbeute: SAM verwirft damit intern unsichere Masken, bevor sie
    ueberhaupt herauskommen. Die Defaults (0.88 / 0.95) sind fuer alltagsuebliche
    Objekte gedacht -- im Kronendach, wo Grenzen objektiv unscharf sind, sortieren
    sie den Grossteil der Kronen aus.
    """
    from transformers import pipeline

    return pipeline(
        "mask-generation",
        model=model_id,
        device=0 if str(device).startswith("cuda") else -1,
        points_per_crop=args.points_per_crop,
        crop_n_layers=args.crop_layers,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
    )


def mask_metrics(mask: np.ndarray) -> dict[str, float] | None:
    """Flaeche, Kompaktheit und Bounding-Box einer Binaermaske."""
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    if area <= 0 or perimeter <= 0:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    hull_area = float(cv2.contourArea(cv2.convexHull(contour))) or area
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None

    return {
        "cx": moments["m10"] / moments["m00"],
        "cy": moments["m01"] / moments["m00"],
        "xmin": x, "ymin": y, "xmax": x + w, "ymax": y + h,
        "area_px": area,
        # 1.0 = perfekter Kreis; Schattenbaender und Astpartien liegen deutlich darunter.
        "kompaktheit": 4 * np.pi * area / (perimeter**2),
        "solidity": area / hull_area,
        "seitenverhaeltnis": min(w, h) / max(w, h),
        "durchmesser_px": 2 * np.sqrt(area / np.pi),
    }


def metrics_from_region(region) -> dict[str, float] | None:
    """mask_metrics auf dem Bounding-Box-Ausschnitt statt auf dem ganzen Bild.

    Bei mehreren hundert Instanzen je Frame ist der Unterschied erheblich: eine
    Vollbildmaske je Instanz kostet 2 Megapixel, der Ausschnitt nur die Krone.
    """
    metrics = mask_metrics(region.image)
    if metrics is None:
        return None
    y0, x0 = region.bbox[0], region.bbox[1]
    for key in ("cx", "xmin", "xmax"):
        metrics[key] += x0
    for key in ("cy", "ymin", "ymax"):
        metrics[key] += y0
    return metrics


def select_crowns(masks: list[np.ndarray], scores: np.ndarray, args) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Filtert Kandidaten und loest Ueberlappungen greedy nach Score auf."""
    expected_area = np.pi * (args.crown_px / 2) ** 2
    min_area, max_area = expected_area * args.min_area_factor, expected_area * args.max_area_factor

    candidates = []
    for mask, score in zip(masks, scores):
        metrics = mask_metrics(mask)
        if metrics is None:
            continue
        if not (min_area <= metrics["area_px"] <= max_area):
            continue
        if metrics["kompaktheit"] < args.min_compactness:
            continue
        if metrics["solidity"] < args.min_solidity:
            continue
        metrics["score"] = float(score)
        candidates.append((metrics, mask))

    candidates.sort(key=lambda item: (-item[0]["score"], -item[0]["area_px"]))

    accepted_records, accepted_masks = [], []
    occupied = None
    for metrics, mask in candidates:
        if occupied is None:
            occupied = np.zeros_like(mask, dtype=bool)
        overlap = np.logical_and(mask, occupied).sum() / max(1, mask.sum())
        if overlap > args.max_overlap:
            continue
        occupied |= mask
        accepted_records.append(metrics)
        accepted_masks.append(mask)

    return pd.DataFrame(accepted_records), accepted_masks


def draw(image_bgr: np.ndarray, masks: list[np.ndarray], crowns: pd.DataFrame, caption: str) -> np.ndarray:
    canvas = image_bgr.copy()
    if masks:
        covered = np.any(np.stack(masks), axis=0)
        # Nicht erfasste Flaeche abdunkeln -- so sieht man sofort, was fehlt.
        canvas[~covered] = (canvas[~covered] * 0.45).astype(np.uint8)
        for mask in masks:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, (80, 230, 120), 2)
    for row in crowns.itertuples():
        cv2.circle(canvas, (int(row.cx), int(row.cy)), 3, (0, 220, 255), -1)

    cv2.rectangle(canvas, (0, 0), (780, 34), (0, 0, 0), -1)
    cv2.putText(canvas, caption, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--frames", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_sam")
    parser.add_argument("--model", default=SAM_MODEL)

    parser.add_argument("--crown-px", type=float, default=100.0)
    parser.add_argument("--min-area-factor", type=float, default=0.12)
    parser.add_argument("--max-area-factor", type=float, default=5.0)
    parser.add_argument("--min-compactness", type=float, default=0.25)
    parser.add_argument("--min-solidity", type=float, default=0.65)
    parser.add_argument("--max-overlap", type=float, default=0.30)

    parser.add_argument("--points-per-crop", type=int, default=48, help="Punktraster je Kachel (48 -> 2304 Punkte).")
    parser.add_argument("--crop-layers", type=int, default=2, help="Zusaetzliche Zoomstufen fuer kleine Objekte.")
    parser.add_argument("--points-per-batch", type=int, default=256)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.70,
                        help="SAM-interne Guetefilterung. Senken erhoeht die Ausbeute deutlich.")
    parser.add_argument("--stability-score-thresh", type=float, default=0.85,
                        help="SAM-interne Stabilitaetsfilterung. Im Kronendach zu streng.")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    return parser.parse_args()


def collect_frames(args) -> list[Path]:
    if args.frames:
        return [args.input / relative for relative in args.frames]
    frames = []
    for folder in sorted(p for p in args.input.iterdir() if p.is_dir()):
        frames.extend(sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES))
    return frames


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Device: {device} | Modell: {args.model}")

    generator = build_generator(args.model, device, args)
    args.out.mkdir(parents=True, exist_ok=True)

    all_crowns = []
    for frame_path in collect_frames(args):
        if not frame_path.exists():
            print(f"  fehlt: {frame_path}")
            continue

        image_bgr = cv2.imread(str(frame_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Die mask-generation-Pipeline erwartet PIL/Pfad, kein numpy-Array.
        with torch.no_grad():
            output = generator(Image.fromarray(image_rgb))
        masks = [np.asarray(m, dtype=bool) for m in output["masks"]]
        scores = np.asarray(output["scores"], dtype=np.float32)

        crowns, kept_masks = select_crowns(masks, scores, args)
        covered = np.any(np.stack(kept_masks), axis=0).mean() if kept_masks else 0.0

        caption = (
            f"{len(masks)} SAM-Masken -> {len(crowns)} Kronen | "
            f"Durchmesser med {crowns['durchmesser_px'].median():.0f} px | "
            f"Bildabdeckung {covered:.0%}"
            if len(crowns)
            else f"{len(masks)} SAM-Masken -> 0 Kronen nach Filter"
        )

        out_folder = args.out / frame_path.parent.name
        out_folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(out_folder / f"{frame_path.stem}_sam.jpg"),
            draw(image_bgr, kept_masks, crowns, caption),
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )

        if len(crowns):
            crowns["abdeckung"] = covered
            crowns.insert(0, "frame", frame_path.name)
            crowns.insert(0, "folder", frame_path.parent.name)
            all_crowns.append(crowns)

        print(f"  {frame_path.parent.name}/{frame_path.name}: {caption}")

    if not all_crowns:
        print("Keine Kronen gefunden.")
        return

    combined = pd.concat(all_crowns, ignore_index=True)
    combined.to_csv(args.out / "all_crowns_sam.csv", index=False)
    print(f"\n{len(combined)} Kronen -> {args.out}/all_crowns_sam.csv")
    print(
        combined.groupby("folder")
        .agg(
            kronen=("frame", "count"),
            durchmesser_px=("durchmesser_px", "median"),
            kompaktheit=("kompaktheit", "median"),
            abdeckung=("abdeckung", "mean"),
        )
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
