"""Kronenabgrenzung ueber monokulare Tiefe + markerbasiertes Watershed.

Die Standardmethode der Forstfernerkundung fuer Einzelbaumabgrenzung arbeitet auf
einem CHM (Canopy Height Model): Wipfel sind lokale Maxima, Kronengrenzen liegen
in den Senken dazwischen, ein markerbasiertes Watershed zieht die Linien. Deine
Einzelframes haben kein CHM -- aber ein monokulares Tiefenmodell liefert eine
Ersatzoberflaeche, die dieselbe Struktur enthaelt.

Ablauf:
  1. Tiefe schaetzen, invertieren (naeher an der Kamera = hoeher).
  2. Detrend: grossskaligen Anteil abziehen. Das entfernt die Schraeglage der
     Kamera und den Bodenplanen-Prior des Modells -- analog zu DSM minus DTM.
  3. Glaetten, damit Blattwerk-Textur keine Scheinwipfel erzeugt.
  4. Lokale Maxima als Wipfelmarker, Mindestabstand = halber Kronendurchmesser.
  5. Watershed auf der invertierten Oberflaeche, begrenzt auf die Kronenmaske.
  6. Segmente nach Flaeche und Form filtern.

Die Tiefenkarten werden zwischengespeichert, damit das Nachjustieren der
Parameter ohne erneute Modellinferenz geht.

Beispiel:
    python segment_trees.py --crown-px 100 --frames 80m/frame_000297.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.measure import label as cc_label, regionprops
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from infer_species import IMAGE_SUFFIXES, REPO_ROOT, resolve_device

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"


# --------------------------------------------------------------------------- #
# Tiefenoberflaeche
# --------------------------------------------------------------------------- #


class DepthEstimator:
    """Laedt das Tiefenmodell einmal und cached die Karten auf Platte."""

    def __init__(self, model_id: str, device, cache_dir: Path | None) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def __call__(self, image_rgb: np.ndarray, key: str) -> np.ndarray:
        cache_path = self.cache_dir / f"{key}.npy" if self.cache_dir else None
        if cache_path and cache_path.exists():
            return np.load(cache_path)

        inputs = self.processor(images=image_rgb, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        depth = self.processor.post_process_depth_estimation(
            outputs, target_sizes=[(image_rgb.shape[0], image_rgb.shape[1])]
        )[0]["predicted_depth"]
        depth = depth.float().cpu().numpy()

        if cache_path:
            np.save(cache_path, depth)
        return depth


def build_pseudo_chm(depth: np.ndarray, crown_px: float, detrend_factor: float) -> np.ndarray:
    """Tiefe -> Ersatz-CHM: invertiert, grossskaliger Trend entfernt."""
    surface = -depth.astype(np.float32)
    trend = cv2.GaussianBlur(surface, (0, 0), max(1.0, crown_px * detrend_factor))
    return surface - trend


# --------------------------------------------------------------------------- #
# Abgrenzung
# --------------------------------------------------------------------------- #


def delineate(
    chm: np.ndarray, crown_px: float, gap_percentile: float, smooth_factor: float, prominence: float
) -> tuple[np.ndarray, np.ndarray]:
    """Watershed-Abgrenzung mit Prominenzpruefung der Wipfel.

    Watershed ist eine Partition, kein Detektor: es zerlegt die Maske in genau so
    viele Teile wie Marker hineingehen. Ein reines lokales Maximum ist deshalb ein
    zu schwaches Kriterium -- in flachen Bereichen erzeugt Rauschen beliebig viele
    davon und damit Pseudo-Kronen. h_maxima verlangt stattdessen, dass sich ein
    Wipfel um mindestens `prominence` ueber seine Umgebung erhebt, bevor er zaehlt.
    """
    smoothed = cv2.GaussianBlur(chm, (0, 0), max(0.8, crown_px * smooth_factor))

    # Kronenmaske: die tiefsten Bereiche sind Luecken, Boden oder Schatten.
    canopy = smoothed > np.percentile(smoothed, gap_percentile)

    # Prominenzschwelle relativ zur robusten Spannweite der Oberflaeche, damit sie
    # nicht von der willkuerlichen Skala des Tiefenmodells abhaengt.
    low, high = np.percentile(smoothed[canopy], [5, 95])
    height_threshold = max(1e-6, (high - low) * prominence)

    seeds = h_maxima(np.where(canopy, smoothed, smoothed.min()), height_threshold)
    markers = cc_label(seeds > 0)
    if markers.max() == 0:
        return np.zeros_like(chm, dtype=np.int32), np.empty((0, 2))

    peaks = np.array([region.centroid for region in regionprops(markers)])
    labels = watershed(-smoothed, markers, mask=canopy)
    return labels, peaks


def crowns_to_frame(labels: np.ndarray, chm: np.ndarray, crown_px: float, args) -> pd.DataFrame:
    """Segmente vermessen und nach Flaeche/Form filtern."""
    expected_area = np.pi * (crown_px / 2) ** 2
    min_area = expected_area * args.min_area_factor
    max_area = expected_area * args.max_area_factor

    records = []
    for region in regionprops(labels, intensity_image=chm):
        if not (min_area <= region.area <= max_area):
            continue
        # Sehr langgezogene Segmente sind meist zwei verschmolzene Kronen oder
        # ein Schattenband, keine Einzelkrone.
        if region.axis_major_length > 0 and (
            region.axis_minor_length / region.axis_major_length < args.min_axis_ratio
        ):
            continue

        cy, cx = region.centroid
        y0, x0, y1, x1 = region.bbox
        records.append(
            {
                "cx": cx,
                "cy": cy,
                "xmin": x0,
                "ymin": y0,
                "xmax": x1,
                "ymax": y1,
                "area_px": region.area,
                "durchmesser_px": region.equivalent_diameter_area,
                "achsenverhaeltnis": (
                    region.axis_minor_length / region.axis_major_length if region.axis_major_length else 0.0
                ),
                "chm_mean": region.intensity_mean,
                "label": region.label,
            }
        )
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Visualisierung
# --------------------------------------------------------------------------- #


def draw_crowns(image_bgr: np.ndarray, labels: np.ndarray, crowns: pd.DataFrame, caption: str) -> np.ndarray:
    canvas = image_bgr.copy()
    keep = set(crowns["label"].tolist())

    # Grenzen der behaltenen Segmente als Linien zeichnen.
    kept_mask = np.isin(labels, list(keep)) if keep else np.zeros_like(labels, dtype=bool)
    for label in keep:
        mask = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (80, 230, 120), 2)

    # Verworfene Segmente dezent in Rot, damit man den Filter beurteilen kann.
    discarded = (labels > 0) & ~kept_mask
    canvas[discarded] = (0.65 * canvas[discarded] + 0.35 * np.array([60, 60, 200])).astype(np.uint8)

    for row in crowns.itertuples():
        cv2.circle(canvas, (int(row.cx), int(row.cy)), 3, (0, 220, 255), -1)

    cv2.rectangle(canvas, (0, 0), (760, 34), (0, 0, 0), -1)
    cv2.putText(canvas, caption, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--frames", nargs="*", default=None, help="Pfade relativ zu --input; None = alle.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_segment")
    parser.add_argument("--depth-cache", type=Path, default=Path("/scratch/shared") / "nik" / "data" / "treeclf" / "depth_cache")

    parser.add_argument("--crown-px", type=float, default=100.0,
                        help="Erwarteter Kronendurchmesser in Pixeln. Steuert alle Skalen.")
    parser.add_argument("--detrend-factor", type=float, default=3.0,
                        help="Sigma des Trendfilters als Vielfaches von --crown-px.")
    parser.add_argument("--smooth-factor", type=float, default=0.06,
                        help="Sigma der Glaettung als Vielfaches von --crown-px.")
    parser.add_argument("--peak-prominence", type=float, default=0.08,
                        help="Wie weit sich ein Wipfel ueber seine Umgebung erheben muss, "
                             "als Anteil der 5-95-Perzentil-Spannweite des Ersatz-CHM.")
    parser.add_argument("--gap-percentile", type=float, default=10.0,
                        help="Perzentil der Oberflaeche, unterhalb dessen als Luecke/Boden verworfen wird.")
    parser.add_argument("--min-area-factor", type=float, default=0.15)
    parser.add_argument("--max-area-factor", type=float, default=4.0)
    parser.add_argument("--min-axis-ratio", type=float, default=0.35)

    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--save-chm", action="store_true", help="Ersatz-CHM zusaetzlich als Bild speichern.")
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
    print(f"Device: {device} | erwarteter Kronendurchmesser: {args.crown_px:.0f} px")

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
        labels, peaks = delineate(
            chm, args.crown_px, args.gap_percentile, args.smooth_factor, args.peak_prominence
        )
        crowns = crowns_to_frame(labels, chm, args.crown_px, args)

        # Wie viel des segmentierten Kronendachs ueberlebt den Filter?
        segmented = labels > 0
        kept = np.isin(labels, crowns["label"].to_numpy()) if len(crowns) else np.zeros_like(segmented)
        abdeckung = kept.sum() / max(1, segmented.sum())

        out_folder = args.out / frame_path.parent.name
        out_folder.mkdir(parents=True, exist_ok=True)

        caption = (
            f"crown_px {args.crown_px:.0f} | {len(peaks)} Wipfel -> {len(crowns)} Kronen | "
            f"Durchmesser med {crowns['durchmesser_px'].median():.0f} px | "
            f"Abdeckung {abdeckung:.0%}"
            if len(crowns)
            else f"{len(peaks)} Wipfel -> 0 Kronen nach Filter"
        )
        cv2.imwrite(
            str(out_folder / f"{frame_path.stem}_crowns.jpg"),
            draw_crowns(image_bgr, labels, crowns, caption),
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
        if args.save_chm:
            normalized = cv2.normalize(chm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imwrite(str(out_folder / f"{frame_path.stem}_chm.jpg"), cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO))

        if len(crowns):
            crowns["abdeckung"] = abdeckung
            crowns.insert(0, "frame", frame_path.name)
            crowns.insert(0, "folder", frame_path.parent.name)
            all_crowns.append(crowns)

        print(f"  {frame_path.parent.name}/{frame_path.name}: {caption}")

    if not all_crowns:
        print("Keine Kronen gefunden.")
        return

    combined = pd.concat(all_crowns, ignore_index=True)
    combined.to_csv(args.out / "all_crowns.csv", index=False)

    print(f"\n{len(combined)} Kronen -> {args.out}/all_crowns.csv")
    summary = combined.groupby("folder").agg(
        kronen=("frame", "count"),
        pro_frame=("frame", lambda s: round(len(s) / s.nunique(), 1)),
        durchmesser_px=("durchmesser_px", "median"),
        achsenverhaeltnis=("achsenverhaeltnis", "median"),
        abdeckung=("abdeckung", "mean"),
    ).round(2)
    print(summary.to_string())


if __name__ == "__main__":
    main()
