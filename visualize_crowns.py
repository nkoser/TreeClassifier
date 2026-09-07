"""Diagnoseansichten fuer die Kronenabgrenzung.

Die bisherige Darstellung hatte zwei Schwaechen: sie hat nicht erfasste Bereiche
abgedunkelt -- also genau die Information versteckt, die man zur Beurteilung
braucht -- und alle Kronen einheitlich gruen umrandet, sodass Falschtrennungen
nicht auffallen.

Diese Ansichten beheben beides und lesen die gespeicherte Labelkarte, laufen also
ohne erneute Modellinferenz:

  instanzen  Jede Krone in eigener Farbe halbtransparent gefuellt, Rand kraeftig.
             Zwei Farben auf einer optisch durchgehenden Krone = Falschtrennung.
             Das Bild bleibt ueberall in voller Helligkeit.
  luecken    Nicht erfasste Flaeche wird schraffiert statt abgedunkelt -- die
             Textur bleibt sichtbar, man kann beurteilen wie viel Struktur dort
             noch ist und ob es echte Baeume oder Schatten sind.
  relief     Kronengrenzen auf der Reliefschattierung des Ersatz-CHM. Zeigt, ob
             eine Grenze einem echten Hoehenruecken folgt oder mitten durch eine
             einzelne Kuppel schneidet.
  tiefe      Die Tiefenkarte selbst, farbcodiert und ohne Trendabzug -- die
             Vorstufe zu 'relief'. Zeigt, was Depth Pro geliefert hat, bevor
             irgendetwas daraus abgeleitet wurde.
  trennungen Jede Grenze zwischen zwei Kronen wird danach eingefaerbt, wie tief
             der Sattel zwischen ihren Wipfeln liegt. Rot = flacher Sattel, die
             beiden gehoeren vermutlich zusammen. Gruen = tiefe Kerbe, die
             Trennung ist durch das Relief gedeckt.

Beispiel:
    python visualize_crowns.py --frames dense/frame_000073.jpg --views instanzen luecken relief
"""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.measure import regionprops

from infer_species import REPO_ROOT
from segment_trees import build_pseudo_chm
from depth_probe import hillshade, normalize


def instance_colors(n: int) -> np.ndarray:
    """Gut unterscheidbare Farben ueber den Goldenen Winkel im Farbkreis."""
    colors = np.zeros((n + 1, 3), dtype=np.uint8)
    for i in range(1, n + 1):
        hue = (i * 0.61803398875) % 1.0
        # Saettigung/Helligkeit leicht variieren, damit Nachbarn mit aehnlichem
        # Farbton sich trotzdem unterscheiden.
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75 + 0.2 * ((i % 3) / 2), 0.85 + 0.15 * ((i % 2)))
        colors[i] = (int(b * 255), int(g * 255), int(r * 255))
    return colors


def boundary_mask(labels: np.ndarray) -> np.ndarray:
    """Pixel, an denen zwei verschiedene Instanzen aneinanderstossen."""
    # cv2-Morphologie kennt kein int32, uint16 reicht fuer die Instanzzahl.
    work = labels.astype(np.uint16)
    kernel = np.ones((3, 3), np.uint8)
    return (cv2.dilate(work, kernel) != cv2.erode(work, kernel)) & (labels > 0)


def hatch_pattern(shape: tuple[int, int], spacing: int = 9) -> np.ndarray:
    """Diagonale Schraffur -- markiert Flaechen, ohne sie zu verdecken."""
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return ((xx + yy) % spacing) < 2


def view_instanzen(image_bgr: np.ndarray, labels: np.ndarray, alpha: float) -> np.ndarray:
    colors = instance_colors(int(labels.max()))
    tint = colors[labels]
    covered = labels > 0

    canvas = image_bgr.copy()
    canvas[covered] = (
        (1 - alpha) * canvas[covered].astype(np.float32) + alpha * tint[covered].astype(np.float32)
    ).astype(np.uint8)
    borders = boundary_mask(labels)
    canvas[borders] = colors[labels[borders]]
    return canvas


def view_luecken(image_bgr: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Nicht erfasste Flaeche schraffieren, Kronen nur duenn umranden."""
    canvas = image_bgr.copy()
    missing = labels == 0

    hatch = hatch_pattern(labels.shape) & missing
    canvas[hatch] = (0.45 * canvas[hatch].astype(np.float32) + 0.55 * np.array([80, 60, 255])).astype(np.uint8)

    canvas[boundary_mask(labels)] = (255, 255, 255)
    return canvas


def view_relief(chm: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Kronengrenzen ueber der Reliefschattierung."""
    shaded = (hillshade(normalize(chm)) * 255).astype(np.uint8)
    canvas = cv2.cvtColor(shaded, cv2.COLOR_GRAY2BGR)

    colors = instance_colors(int(labels.max()))
    borders = boundary_mask(labels)
    canvas[borders] = colors[labels[borders]]
    return canvas


def view_tiefe(height_field: np.ndarray) -> np.ndarray:
    """Die Hoehenflaeche selbst, farbcodiert -- ohne Trendabzug, ohne Grenzen.

    'relief' zeigt das Ersatz-CHM, also die Tiefe nach Inversion und Abzug des
    grossskaligen Trends. Diese Ansicht zeigt die Vorstufe davon: was das
    Tiefenmodell tatsaechlich ausgegeben hat. Nuetzlich, um zu unterscheiden, ob
    ein fehlendes Kronenrelief schon in der Schaetzung fehlt oder erst beim
    Detrending verlorengeht.

    Gespreizt wird ueber das 2./98. Perzentil (in `normalize`), weil eine
    einzelne Luecke bis zum Boden sonst den ganzen Farbbereich zusammendrueckt.
    """
    return cv2.applyColorMap((normalize(height_field) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def split_prominence(labels: np.ndarray, chm: np.ndarray) -> tuple[dict[tuple[int, int], float], np.ndarray]:
    """Bewertet jede Trennung zwischen zwei Nachbarkronen.

    Zwei Kronen sind zu Recht getrennt, wenn zwischen ihren Wipfeln eine echte
    Kerbe liegt. Kennzahl ist deshalb die Sattelprominenz: wie weit faellt die
    Oberflaeche vom niedrigeren der beiden Wipfel bis zum hoechsten Punkt der
    gemeinsamen Grenze ab, relativ zur Spannweite des Bildes. Ein flacher Sattel
    heisst: die Grenze laeuft ueber eine durchgehende Kuppel -- vermutlich wurde
    ein Baum zerschnitten.
    """
    smoothed = cv2.GaussianBlur(chm, (0, 0), 6.0)
    low, high = np.percentile(smoothed[labels > 0], [5, 95]) if (labels > 0).any() else (0.0, 1.0)
    span = max(1e-6, high - low)

    # Wipfelhoehe je Instanz.
    peaks: dict[int, float] = {}
    for region in regionprops(labels.astype(np.int32), intensity_image=smoothed):
        peaks[region.label] = float(region.intensity_max)

    # Benachbarte Labelpaare ueber verschobene Vergleiche einsammeln.
    saddles: dict[tuple[int, int], float] = {}
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = labels[max(0, -dy) : labels.shape[0] - max(0, dy), max(0, -dx) : labels.shape[1] - max(0, dx)]
        b = labels[max(0, dy) : labels.shape[0] - max(0, -dy), max(0, dx) : labels.shape[1] - max(0, -dx)]
        h = smoothed[max(0, -dy) : smoothed.shape[0] - max(0, dy), max(0, -dx) : smoothed.shape[1] - max(0, dx)]

        touching = (a != b) & (a > 0) & (b > 0)
        if not touching.any():
            continue
        for la, lb, height in zip(a[touching], b[touching], h[touching]):
            key = (int(min(la, lb)), int(max(la, lb)))
            if height > saddles.get(key, -np.inf):
                saddles[key] = float(height)

    return (
        {
            key: (min(peaks.get(key[0], 0.0), peaks.get(key[1], 0.0)) - saddle) / span
            for key, saddle in saddles.items()
        },
        smoothed,
    )


def view_trennungen(image_bgr: np.ndarray, labels: np.ndarray, chm: np.ndarray, threshold: float) -> tuple[np.ndarray, dict]:
    prominences, _ = split_prominence(labels, chm)
    canvas = (image_bgr * 0.55).astype(np.uint8)

    work = labels.astype(np.uint16)
    kernel = np.ones((3, 3), np.uint8)
    dilated, eroded = cv2.dilate(work, kernel), cv2.erode(work, kernel)

    for (la, lb), value in prominences.items():
        pair_edge = ((dilated == lb) & (eroded == la)) | ((dilated == la) & (eroded == lb))
        if not pair_edge.any():
            continue
        # rot (flacher Sattel, verdaechtig) -> gruen (tiefe Kerbe, gedeckt)
        t = float(np.clip(value / (2 * threshold), 0, 1))
        canvas[pair_edge] = (int(60 * t), int(60 + 170 * t), int(255 - 195 * t))

    return canvas, prominences


def annotate(canvas: np.ndarray, caption: str) -> np.ndarray:
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 980), 34), (0, 0, 0), -1)
    cv2.putText(canvas, caption, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--segments", type=Path, default=REPO_ROOT / "results_hybrid",
                        help="Verzeichnis mit den *_labels.png aus segment_hybrid.py.")
    parser.add_argument("--frames", nargs="*", default=None, help="Pfade relativ zu --input; None = alle gefundenen.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_views")
    parser.add_argument("--views", nargs="*", default=["instanzen", "luecken", "tiefe", "relief", "trennungen"],
                        choices=["instanzen", "luecken", "tiefe", "relief", "trennungen"])
    parser.add_argument("--split-threshold", type=float, default=0.06,
                        help="Sattelprominenz, unterhalb derer eine Trennung als verdaechtig gilt.")
    parser.add_argument("--alpha", type=float, default=0.35, help="Deckkraft der Instanzfarben.")
    parser.add_argument("--depth-cache", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/depth_cache"))
    parser.add_argument("--parallax-cache", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/parallax_cache"))
    parser.add_argument("--surface", choices=("depth", "parallax"), default="depth",
                        help="Bezugsflaeche fuer Relief und Trennungsbewertung.")
    parser.add_argument("--crown-px", type=float, default=100.0)
    parser.add_argument("--detrend-factor", type=float, default=3.0)
    return parser.parse_args()


def collect(args) -> list[tuple[Path, Path]]:
    """Paare aus (Originalframe, Labelkarte)."""
    pairs = []
    label_paths = (
        [args.segments / Path(r).parent.name / f"{Path(r).stem}_labels.png" for r in args.frames]
        if args.frames
        else sorted(args.segments.glob("*/*_labels.png"))
    )
    for label_path in label_paths:
        if not label_path.exists():
            print(f"  keine Labelkarte: {label_path}")
            continue
        stem = label_path.name.replace("_labels.png", "")
        candidates = list((args.input / label_path.parent.name).glob(f"{stem}.*"))
        if not candidates:
            print(f"  kein Originalframe zu {label_path}")
            continue
        pairs.append((candidates[0], label_path))
    return pairs


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for frame_path, label_path in collect(args):
        image_bgr = cv2.imread(str(frame_path))
        labels = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED).astype(np.int32)

        covered = float((labels > 0).mean())
        n = int(labels.max())
        out_folder = args.out / frame_path.parent.name
        out_folder.mkdir(parents=True, exist_ok=True)

        if "instanzen" in args.views:
            canvas = view_instanzen(image_bgr, labels, args.alpha)
            annotate(canvas, f"{n} Instanzen | jede Krone eigene Farbe -- zwei Farben auf einer Krone = Falschtrennung")
            cv2.imwrite(str(out_folder / f"{frame_path.stem}_instanzen.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])

        if "luecken" in args.views:
            canvas = view_luecken(image_bgr, labels)
            annotate(canvas, f"Abdeckung {covered:.0%} | schraffiert = nicht erfasst, Struktur bleibt sichtbar")
            cv2.imwrite(str(out_folder / f"{frame_path.stem}_luecken.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])

        if {"tiefe", "relief", "trennungen"} & set(args.views):
            key = f"{frame_path.parent.name}__{frame_path.stem}.npy"
            surface_path = (args.parallax_cache if args.surface == "parallax" else args.depth_cache) / key
            if not surface_path.exists():
                print(f"  keine {args.surface}-Karte: {surface_path}")
            else:
                raw = np.load(surface_path).astype(np.float32)
                if args.surface == "parallax":
                    # Parallaxe ist bereits Hoehe -- nur Trend abziehen, nicht invertieren.
                    hoehe = raw
                    chm = raw - cv2.GaussianBlur(raw, (0, 0), max(1.0, args.crown_px * args.detrend_factor))
                else:
                    # Depth Pro gibt Entfernung aus: naeher an der Kamera = hoeher.
                    hoehe = -raw
                    chm = build_pseudo_chm(raw, args.crown_px, args.detrend_factor)

                if "tiefe" in args.views:
                    quelle = "Parallaxe (gemessen)" if args.surface == "parallax" else "Depth Pro (geschaetzt)"
                    canvas = view_tiefe(hoehe)
                    annotate(canvas, f"{quelle}, roh und ohne Trendabzug | rot = hoch/nah, blau = tief/fern")
                    cv2.imwrite(str(out_folder / f"{frame_path.stem}_tiefe.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])

                if "relief" in args.views:
                    canvas = view_relief(chm, labels)
                    annotate(canvas, "Grenzen auf Relief | folgt die Linie einem Hoehenruecken oder schneidet sie eine Kuppel?")
                    cv2.imwrite(str(out_folder / f"{frame_path.stem}_relief.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])

                if "trennungen" in args.views:
                    canvas, prominences = view_trennungen(image_bgr, labels, chm, args.split_threshold)
                    verdaechtig = sum(1 for v in prominences.values() if v < args.split_threshold)
                    annotate(
                        canvas,
                        f"{verdaechtig} von {len(prominences)} Trennungen verdaechtig (flacher Sattel) | "
                        f"rot = gehoert vermutlich zusammen, gruen = Trennung gedeckt",
                    )
                    cv2.imwrite(str(out_folder / f"{frame_path.stem}_trennungen.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])
                    print(f"    Trennungen: {verdaechtig}/{len(prominences)} verdaechtig")

        print(f"  {frame_path.parent.name}/{frame_path.name}: {n} Instanzen, Abdeckung {covered:.0%}")

    print(f"\nAnsichten in {args.out}")


if __name__ == "__main__":
    main()
