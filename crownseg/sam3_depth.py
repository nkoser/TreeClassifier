"""SAM 3 mit Depth Pro: der Textprompt findet Kronen, die Tiefe trennt und ergaenzt.

Auf BAMFORESTS test1 (Hain) haben alle SAM-Varianten dasselbe Profil gezeigt:
die Raender sitzen ausgezeichnet -- mittlere IoU der Treffer 0.75 bis 0.77, besser
als alles Trainierte --, aber es wird zu wenig gefunden. SAM 3 mit Textprompt
kommt auf eine Trefferquote von 0.30, die Tiefen-Prompt-Variante auf 0.16. Der
Engpass sind fehlende Instanzen, nicht schlechte Abgrenzung.

Die Tiefe bekommt hier deshalb drei klar getrennte Aufgaben:

  trennen      Eine SAM-Maske ueber mehreren Wipfeln wird an den Wipfeln
               aufgeteilt (Watershed im Inneren der Maske). Genau diesen Fall
               behandelt `segment_hybrid.py` nicht -- dort ist eine SAM-Maske
               immer genau eine Krone.
  ergaenzen    Wipfel ohne SAM-Maske bekommen ein Watershed-Becken auf der
               Restflaeche. Das ist der Teil, den `segment_hybrid.py` bereits
               kann und der hier unveraendert wiederverwendet wird.
  saeen       Ein Wipfel, der in keiner SAM-3-Maske liegt, wird zum Punkt-Prompt.
               Die Tiefe liefert nur das Wo, die Grenze zieht wieder ein
               Bildmodell -- der Unterschied zum Ergaenzungsschritt, bei dem die
               Form aus dem Watershed kam und nichts traf.
  bestaetigen  Eine Maske ohne jeden Wipfel ueberlebt nur, wenn ihre Form passt.

Warum Depth Pro und nicht weiter Depth-Anything-V2: fuer das Trennen zaehlt
nicht die metrische Richtigkeit der Tiefe, sondern wie scharf die Kante zwischen
zwei benachbarten Wipfeln ist. Depth-Anything-V2-Metric-Outdoor liefert eine
glatte Oberflaeche, auf der zwei sich beruehrende Kronen zu einem Huegel
verschmelzen. Das ist aber eine Vermutung, keine Messung -- deshalb ist das
Modell ein Schalter (`--depth-model`) und beide Varianten werden gegen dieselbe
Wahrheit gerechnet.

    python crownseg/sam3_depth.py --input <ordner> --depth-model depthpro
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.measure import label as cc_label, regionprops
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infer_species import IMAGE_SUFFIXES, REPO_ROOT, resolve_device  # noqa: E402
from segment_hybrid import residual_crowns  # noqa: E402
from segment_sam import mask_metrics  # noqa: E402
from segment_prompted import find_peaks, pick_candidates, prompt_sam  # noqa: E402
from segment_sam import SAM_MODEL  # noqa: E402
from segment_sam3 import SAM3_MODEL, segment_tile  # noqa: E402
from segment_trees import DepthEstimator, build_pseudo_chm  # noqa: E402

DEPTH_MODELS = {
    "depthpro": "apple/DepthPro-hf",
    "dav2": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
}


def treetops(chm: np.ndarray, args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wipfel als lokale Maxima im Ersatz-CHM, ueber dem Kronendach gerechnet."""
    smoothed = cv2.GaussianBlur(chm, (0, 0), max(0.8, args.crown_px * args.smooth_factor))
    canopy = smoothed > np.percentile(smoothed, args.gap_percentile)
    if canopy.sum() < 10:
        return np.zeros(chm.shape, np.int32), smoothed, canopy

    low, high = np.percentile(smoothed[canopy], [5, 95])
    seeds = h_maxima(np.where(canopy, smoothed, smoothed.min()),
                     max(1e-6, (high - low) * args.peak_prominence))
    return cc_label(seeds > 0), smoothed, canopy


def split_by_tops(mask: np.ndarray, markers: np.ndarray, smoothed: np.ndarray) -> list[np.ndarray]:
    """Maske an ihren Wipfeln aufteilen; bei hoechstens einem Wipfel unveraendert."""
    local = np.where(mask, markers, 0)
    present = [value for value in np.unique(local) if value > 0]
    if len(present) <= 1:
        return [mask]

    labels = watershed(-smoothed, local, mask=mask)
    parts = [labels == value for value in present]
    return [part for part in parts if part.sum() > 0]


def shape_filter(mask: np.ndarray, args) -> dict | None:
    metrics = mask_metrics(mask)
    if metrics is None:
        return None
    expected = np.pi * (args.crown_px / 2) ** 2
    if not expected * args.min_area_factor <= metrics["area_px"] <= expected * args.max_area_factor:
        return None
    if metrics["kompaktheit"] < args.min_compactness or metrics["solidity"] < args.min_solidity:
        return None
    return metrics


def crowns_from_frame(model, processor, estimator, image_rgb, key, args, device, seeder=None):
    depth = estimator(image_rgb, key)
    chm = build_pseudo_chm(depth, args.crown_px, args.detrend_factor)
    markers, smoothed, canopy = treetops(chm, args)
    # Die Becken dienen nur als Groessenreferenz bei der Kandidatenwahl -- die
    # Grenze zieht SAM. Genau die Rolle, in der das Watershed etwas taugt.
    basins = watershed(-smoothed, markers, mask=canopy) if seeder is not None else None

    masks, scores = segment_tile(model, processor, image_rgb, args.prompt, args.threshold, device)
    order = np.argsort(-np.asarray(scores)) if len(scores) else []

    occupied = np.zeros(image_rgb.shape[:2], dtype=bool)
    records, kept, sources = [], [], []
    split_count = 0

    for index in order:
        mask = np.asarray(masks[index], dtype=bool)
        if mask.sum() == 0:
            continue
        # Bereits vergebene Flaeche abziehen statt die Maske ganz zu verwerfen --
        # SAM 3 liefert regelmaessig ineinanderliegende Kandidaten.
        if np.logical_and(mask, occupied).sum() / mask.sum() > args.max_overlap:
            continue
        mask = mask & ~occupied

        parts = split_by_tops(mask, markers, smoothed)
        split_count += len(parts) - 1
        for part in parts:
            metrics = shape_filter(part, args)
            if metrics is None:
                continue
            records.append({**metrics, "score": float(scores[index])})
            kept.append(part)
            sources.append("sam3")
        occupied |= mask

    # Saeen: freie Wipfel als Punkt-Prompt an SAM.
    #
    # Gemessen auf test1: von den Wipfeln, die in keiner SAM-3-Maske liegen,
    # liegen bei Prominenz 0.02 zwei Drittel (66.7 %) in einer Krone, die SAM 3
    # verpasst hat. Die Positionen taugen also, nur die Watershed-Formen an
    # denselben Stellen trafen nichts. Obergrenze dieses Schritts: Trefferquote
    # 0.554 statt 0.296, F1 0.606 statt 0.342.
    saat = dict(frei=0, kandidaten=0, ueberlappt=0, form=0, genommen=0)
    if seeder is not None:
        occupied_now = occupied.copy()
        free = []
        for region in regionprops(markers):
            y, x = int(region.centroid[0]), int(region.centroid[1])
            if not occupied_now[y, x]:
                free.append((x, y, region.label))

        saat["frei"] = len(free)
        if free:
            points = np.array([[x, y] for x, y, _ in free], dtype=np.float32)
            basin_ids = np.array([label for _, _, label in free])
            sam_masks, sam_scores = prompt_sam(
                seeder[0], seeder[1], image_rgb, points, device, args.chunk)
            chosen = pick_candidates(sam_masks, sam_scores, basins, basin_ids, args)
            saat["kandidaten"] = len(chosen)

            for mask, score, metrics in chosen:
                mask = np.asarray(mask, dtype=bool)
                if mask.sum() == 0:
                    continue
                # Kronen beruehren sich; eine frisch gesaete Krone ueberlappt
                # ihre Nachbarn fast immer. Gemessen wird deshalb gegen einen
                # eigenen, grosszuegigeren Schwellwert als bei SAM 3 selbst.
                if np.logical_and(mask, occupied).sum() / mask.sum() > args.seed_max_overlap:
                    saat["ueberlappt"] += 1
                    continue
                filtered = shape_filter(mask, args)
                if filtered is None:
                    saat["form"] += 1
                    continue
                records.append({**filtered, "score": float(score)})
                kept.append(mask)
                sources.append("saat")
                saat["genommen"] += 1
                occupied |= mask

    # Restflaeche: Wipfel, zu denen SAM 3 nichts geliefert hat.
    #
    # Gemessen auf test1 (Hain): von 291 so ergaenzten Kronen trifft keine
    # einzige eine echte Krone bei IoU 0.5, bei IoU 0.1 sind es 2.7 %. Sie
    # liegen in Luecken und Schatten, nicht auf Baeumen -- die Restflaeche ist
    # per Konstruktion das, was SAM 3 fuer keinen Baum gehalten hat, und darin
    # findet das Watershed zuverlaessig nichts. Deshalb standardmaessig aus.
    if not args.residual:
        frame = pd.DataFrame(records)
        if not frame.empty:
            frame["quelle"] = sources
        return frame, kept, split_count, len(np.unique(markers)) - 1, saat

    if args.dilate_sam > 0 and kept:
        kernel = np.ones((args.dilate_sam, args.dilate_sam), np.uint8)
        grown = cv2.dilate(occupied.astype(np.uint8), kernel).astype(bool)
    else:
        grown = occupied
    residual_frame, residual_masks = residual_crowns(chm, ~grown, args)
    for _, row in residual_frame.iterrows():
        records.append(row.to_dict())
        sources.append("tiefe")
    kept.extend(residual_masks)

    frame = pd.DataFrame(records)
    if not frame.empty:
        frame["quelle"] = sources
    return frame, kept, split_count, len(np.unique(markers)) - 1, saat


def write_outputs(out_dir: Path, stem: str, image_bgr, masks, frame, caption) -> None:
    labels = np.zeros(image_bgr.shape[:2], dtype=np.uint16)
    for index, mask in enumerate(masks, start=1):
        labels[mask] = index
    cv2.imwrite(str(out_dir / f"{stem}_labels.png"), labels)

    canvas = image_bgr.copy()
    for mask, source in zip(masks, frame.get("quelle", ["sam3"] * len(masks))):
        colour = (80, 230, 120) if source == "sam3" else (255, 190, 60)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, colour, 2)
    cv2.rectangle(canvas, (0, 0), (900, 34), (0, 0, 0), -1)
    cv2.putText(canvas, caption, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / f"{stem}_sam3depth.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_sam3depth")
    parser.add_argument("--depth-model", default="depthpro", choices=list(DEPTH_MODELS))
    parser.add_argument("--depth-cache", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/depth_cache"))
    parser.add_argument("--sam3-model", default=SAM3_MODEL)
    parser.add_argument("--prompt", default="tree")
    parser.add_argument("--threshold", type=float, default=0.15)

    parser.add_argument("--crown-px", type=float, default=275.0)
    parser.add_argument("--min-area-factor", type=float, default=0.12)
    parser.add_argument("--max-area-factor", type=float, default=5.0)
    parser.add_argument("--min-compactness", type=float, default=0.25)
    parser.add_argument("--min-solidity", type=float, default=0.65)
    parser.add_argument("--max-overlap", type=float, default=0.30)

    parser.add_argument("--detrend-factor", type=float, default=3.0)
    parser.add_argument("--smooth-factor", type=float, default=0.06)
    parser.add_argument("--gap-percentile", type=float, default=15.0)
    parser.add_argument("--peak-prominence", type=float, default=0.02,
                        help="0.10 war auf 100-px-Kronen eingestellt; bei 275 px viel zu streng.")
    parser.add_argument("--seed-free-peaks", action=argparse.BooleanOptionalAction, default=True,
                        help="Freie Wipfel als Punkt-Prompt an SAM geben.")
    parser.add_argument("--sam-model", default=SAM_MODEL, help="Punkt-promptbares Modell fuer die Saat.")
    parser.add_argument("--select", choices=("basin", "score", "area"), default="basin")
    parser.add_argument("--chunk", type=int, default=24)
    parser.add_argument("--seed-max-overlap", type=float, default=0.60,
                        help="Wieviel eine gesaete Krone mit bereits gesetzten teilen darf.")
    parser.add_argument("--residual", action=argparse.BooleanOptionalAction, default=False,
                        help="Aus der Restflaeche zusaetzliche Kronen ergaenzen (gemessen wertlos).")
    parser.add_argument("--dilate-sam", type=int, default=3)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    return parser.parse_args()


def main() -> None:
    from transformers import Sam3Model, Sam3Processor

    args = parse_args()
    device = resolve_device(args.device)
    model_id = DEPTH_MODELS[args.depth_model]
    print(f"Device: {device} | SAM3 '{args.prompt}' | Tiefe: {model_id}", flush=True)

    processor = Sam3Processor.from_pretrained(args.sam3_model)
    model = Sam3Model.from_pretrained(args.sam3_model).to(device).eval()

    seeder = None
    if args.seed_free_peaks:
        from transformers import AutoProcessor, SamModel

        seeder = (SamModel.from_pretrained(args.sam_model).to(device).eval(),
                  AutoProcessor.from_pretrained(args.sam_model))
        print(f"Saat freier Wipfel ueber {args.sam_model}", flush=True)
    # Eigener Cache je Tiefenmodell -- sonst liest der Depth-Pro-Lauf die
    # Karten des Depth-Anything-Laufs und misst unbemerkt dasselbe zweimal.
    estimator = DepthEstimator(model_id, device, args.depth_cache / args.depth_model)

    folders = sorted(p for p in args.input.iterdir() if p.is_dir()) or [args.input]
    rows = []
    for folder in folders:
        frames = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        out_dir = args.out / folder.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for frame_path in frames:
            image_bgr = cv2.imread(str(frame_path))
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            key = f"{folder.name}_{frame_path.stem}"
            frame, masks, splits, tops, saat = crowns_from_frame(
                model, processor, estimator, image_rgb, key, args, device, seeder)

            counts = frame["quelle"].value_counts().to_dict() if not frame.empty else {}
            caption = (f"{len(masks)} Kronen | {counts.get('sam3', 0)} SAM3 (+{splits} geteilt) | "
                       f"{counts.get('saat', 0)} gesaet | {counts.get('tiefe', 0)} aus Tiefe | "
                       f"{tops} Wipfel")
            print(f"    Saat: {saat['frei']} frei -> {saat['kandidaten']} Kandidaten, "
                  f"{saat['ueberlappt']} zu ueberlappend, {saat['form']} Form, "
                  f"{saat['genommen']} genommen", flush=True)
            write_outputs(out_dir, frame_path.stem, image_bgr, masks, frame, caption)
            print(f"  {folder.name}/{frame_path.name}: {caption}", flush=True)

            if not frame.empty:
                frame.insert(0, "frame", frame_path.stem)
                frame.insert(0, "folder", folder.name)
                rows.append(frame)

    if rows:
        table = pd.concat(rows, ignore_index=True)
        table.to_csv(args.out / "all_crowns.csv", index=False)
        print(f"\n{len(table)} Kronen -> {args.out / 'all_crowns.csv'}")
        print(table.groupby(["folder", "quelle"]).size().to_string())


if __name__ == "__main__":
    main()
