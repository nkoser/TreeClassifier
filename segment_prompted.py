"""Tiefe als Prompt-Quelle: CHM-Wipfel steuern SAM an.

Bisher tasten SAM 1/2 ein blindes Punktraster ab und SAM 3 sucht per Textbegriff.
Beide muessen dabei selbst herausfinden, wo ueberhaupt ein Baum anfaengt. Die
monokulare Tiefe weiss das besser: ein Wipfel ist ein lokales Maximum im
Ersatz-CHM.

Dieses Skript kombiniert deshalb:

  Tiefe  ->  WO ist ein Baum      (Wipfel als Prompt-Punkt, mit Prominenzpruefung)
  SAM    ->  WO ist seine Grenze  (promptbare Segmentierung, ein Punkt pro Krone)

Gegenueber segment_trees.py (Watershed auf derselben Tiefe) kommt die Grenze
nicht aus der geglaetteten Tiefenoberflaeche, sondern aus den echten Bildkanten.
Gegenueber segment_sam.py entfaellt das Raten, welche der vielen SAM-Masken eine
Krone ist -- pro Wipfel wird genau eine erzeugt.

SAM liefert je Punkt drei Kandidaten unterschiedlicher Ausdehnung (Teil, Objekt,
Kontext). Welcher davon die Krone ist, entscheidet wieder die Tiefe: das
Wassereinzugsgebiet des Wipfels im Ersatz-CHM sagt bereits, wie weit diese Krone
reicht. Gewaehlt wird der Kandidat mit der hoechsten Ueberdeckung zu diesem
Becken (--select basin).

Die Alternative --select area presst jede Krone auf eine fest angenommene
Groesse und schneidet dadurch grosse Kronen auf den hellen Kern zurueck; --select
score nimmt SAMs eigene Bewertung, die oft auf das ganze Kronendach zielt.

Beispiel:
    python segment_prompted.py --frames 80m/frame_000297.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.measure import label as cc_label, regionprops
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from infer_species import IMAGE_SUFFIXES, REPO_ROOT, resolve_device
from segment_sam import mask_metrics
from segment_trees import DEPTH_MODEL, DepthEstimator, build_pseudo_chm

SAM_MODEL = "facebook/sam-vit-large"


def find_peaks(chm: np.ndarray, args) -> tuple[np.ndarray, np.ndarray]:
    """Wipfel [N, 2] als (x, y) und ihre Wassereinzugsgebiete als Labelbild.

    Die Becken dienen nur als Groessenreferenz fuer die Kandidatenauswahl -- die
    endgueltige Grenze zieht SAM, nicht das Watershed.
    """
    smoothed = cv2.GaussianBlur(chm, (0, 0), max(0.8, args.crown_px * args.smooth_factor))
    canopy = smoothed > np.percentile(smoothed, args.gap_percentile)

    low, high = np.percentile(smoothed[canopy], [5, 95])
    seeds = h_maxima(np.where(canopy, smoothed, smoothed.min()), max(1e-6, (high - low) * args.peak_prominence))
    markers = cc_label(seeds > 0)
    if markers.max() == 0:
        return np.empty((0, 2), dtype=np.float32), np.zeros_like(markers)

    regions = regionprops(markers)
    points = np.array([[r.centroid[1], r.centroid[0]] for r in regions], dtype=np.float32)
    basins = watershed(-smoothed, markers, mask=canopy)
    return points, basins


@torch.no_grad()
def prompt_sam(model, processor, image_rgb: np.ndarray, points: np.ndarray, device, chunk: int):
    """Je Punkt drei Maskenkandidaten mit Score."""
    pil = Image.fromarray(image_rgb)
    all_masks, all_scores = [], []

    for start in range(0, len(points), chunk):
        block = points[start : start + chunk]
        # Form [Bild, Punktgruppe, Punkte je Maske, 2] -- eine Gruppe je Wipfel.
        input_points = [[[[float(x), float(y)]] for x, y in block]]
        inputs = processor(pil, input_points=input_points, return_tensors="pt").to(device)
        outputs = model(**inputs, multimask_output=True)

        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )[0]
        all_masks.append(masks.numpy())
        all_scores.append(outputs.iou_scores[0].cpu().numpy())

    return np.concatenate(all_masks), np.concatenate(all_scores)


def pick_candidates(masks: np.ndarray, scores: np.ndarray, basins: np.ndarray, basin_ids: np.ndarray,
                    args) -> list[tuple[np.ndarray, float, dict]]:
    """Je Wipfel einen der drei SAM-Kandidaten waehlen."""
    expected_area = np.pi * (args.crown_px / 2) ** 2
    chosen = []

    for index in range(len(masks)):
        basin = basins == basin_ids[index] if args.select == "basin" else None
        best = None

        for option in range(masks.shape[1]):
            mask = masks[index, option].astype(bool)
            metrics = mask_metrics(mask)
            if metrics is None:
                continue
            # Grosszuegiges Fenster: es soll nur den Ausreisser "ganzes
            # Kronendach" abfangen, nicht die Groesse vorschreiben.
            if not (expected_area * args.min_area_factor <= metrics["area_px"] <= expected_area * args.max_area_factor):
                continue
            if metrics["kompaktheit"] < args.min_compactness:
                continue

            if args.select == "basin":
                union = np.logical_or(mask, basin).sum()
                quality = np.logical_and(mask, basin).sum() / max(1, union)
            elif args.select == "score":
                quality = float(scores[index, option])
            else:
                quality = -abs(np.log(metrics["area_px"] / expected_area))

            if best is None or quality > best[0]:
                best = (quality, mask, float(scores[index, option]), metrics)

        if best is not None:
            chosen.append((best[1], best[2], best[3]))
    return chosen


def resolve(chosen: list[tuple[np.ndarray, float, dict]], max_overlap: float):
    """Ueberlappungen aufloesen: nach SAM-Score gierig annehmen."""
    chosen = sorted(chosen, key=lambda item: -item[1])
    masks, records, occupied = [], [], None

    for mask, score, metrics in chosen:
        if occupied is None:
            occupied = np.zeros_like(mask, dtype=bool)
        if np.logical_and(mask, occupied).sum() / max(1, mask.sum()) > max_overlap:
            continue
        occupied |= mask
        metrics["score"] = score
        masks.append(mask)
        records.append(metrics)
    return masks, pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--frames", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_prompted")
    parser.add_argument("--depth-cache", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/depth_cache"))
    parser.add_argument("--sam-model", default=SAM_MODEL)

    parser.add_argument("--crown-px", type=float, default=100.0)
    parser.add_argument("--smooth-factor", type=float, default=0.06)
    parser.add_argument("--gap-percentile", type=float, default=10.0)
    parser.add_argument("--peak-prominence", type=float, default=0.04,
                        help="Niedriger als bei segment_trees.py: hier darf ein Wipfel schwach sein, "
                             "die Grenze zieht ohnehin SAM.")
    parser.add_argument("--detrend-factor", type=float, default=3.0)

    parser.add_argument("--select", choices=("basin", "score", "area"), default="basin",
                        help="Wie unter SAMs drei Kandidaten gewaehlt wird. basin: hoechste Ueberdeckung "
                             "mit dem Wassereinzugsgebiet des Wipfels (empfohlen).")
    parser.add_argument("--min-area-factor", type=float, default=0.10)
    parser.add_argument("--max-area-factor", type=float, default=5.0)
    parser.add_argument("--min-compactness", type=float, default=0.20)
    parser.add_argument("--max-overlap", type=float, default=0.30)
    parser.add_argument("--chunk", type=int, default=24, help="Wipfel je SAM-Durchlauf.")
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
    print(f"Device: {device} | SAM: {args.sam_model}")

    from transformers import SamModel, SamProcessor

    processor = SamProcessor.from_pretrained(args.sam_model)
    model = SamModel.from_pretrained(args.sam_model).to(device).eval()
    estimator = DepthEstimator(DEPTH_MODEL, device, args.depth_cache)
    args.out.mkdir(parents=True, exist_ok=True)

    all_crowns = []
    for frame_path in collect_frames(args):
        if not frame_path.exists():
            print(f"  fehlt: {frame_path}")
            continue

        image_bgr = cv2.imread(str(frame_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        depth = estimator(image_rgb, f"{frame_path.parent.name}__{frame_path.stem}")
        chm = build_pseudo_chm(depth, args.crown_px, args.detrend_factor)
        points, basins = find_peaks(chm, args)
        basin_ids = np.arange(1, len(points) + 1)
        if len(points) == 0:
            print(f"  {frame_path.parent.name}/{frame_path.name}: keine Wipfel")
            continue

        masks_raw, scores = prompt_sam(model, processor, image_rgb, points, device, args.chunk)
        masks, crowns = resolve(pick_candidates(masks_raw, scores, basins, basin_ids, args), args.max_overlap)
        covered = float(np.any(np.stack(masks), axis=0).mean()) if masks else 0.0

        out_folder = args.out / frame_path.parent.name
        out_folder.mkdir(parents=True, exist_ok=True)

        label_map = np.zeros(image_rgb.shape[:2], dtype=np.uint16)
        for index, mask in enumerate(masks, start=1):
            label_map[mask] = index
        cv2.imwrite(str(out_folder / f"{frame_path.stem}_labels.png"), label_map)

        caption = f"{len(points)} Wipfel -> {len(crowns)} Kronen | Abdeckung {covered:.0%}"
        canvas = image_bgr.copy()
        for mask in masks:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, (80, 230, 120), 2)
        for x, y in points:
            cv2.circle(canvas, (int(x), int(y)), 3, (0, 220, 255), -1)
        cv2.rectangle(canvas, (0, 0), (760, 34), (0, 0, 0), -1)
        cv2.putText(canvas, caption, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_folder / f"{frame_path.stem}_prompted.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])

        if len(crowns):
            crowns.insert(0, "id", np.arange(1, len(crowns) + 1))
            crowns["abdeckung"] = covered
            crowns.insert(0, "frame", frame_path.name)
            crowns.insert(0, "folder", frame_path.parent.name)
            all_crowns.append(crowns)

        print(f"  {frame_path.parent.name}/{frame_path.name}: {caption}")

    if not all_crowns:
        print("Keine Kronen gefunden.")
        return

    combined = pd.concat(all_crowns, ignore_index=True)
    combined.to_csv(args.out / "all_crowns_prompted.csv", index=False)
    print(f"\n{len(combined)} Kronen -> {args.out}/all_crowns_prompted.csv")
    print(
        combined.groupby("folder")
        .agg(kronen=("id", "count"), durchmesser_px=("durchmesser_px", "median"),
             kompaktheit=("kompaktheit", "median"), abdeckung=("abdeckung", "mean"))
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
