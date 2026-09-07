"""Echte Parallaxe aus aufeinanderfolgenden Videoframes statt geschaetzter Tiefe.

Bisher kam die Hoeheninformation aus einem monokularen Tiefenmodell -- also aus
einer Schaetzung, die auf einem Einzelbild nichts messen kann und in
kontrastarmen Bereichen glatte Flaechen erfindet. Das war durchgehend die
schwaechste Stelle der Pipeline.

Die Frames eines Ordners stammen aber aus demselben Video, wenige Zehntel- bis
Sekunden auseinander. Die Drohne hat sich dazwischen bewegt, und damit enthalten
zwei Frames echte Parallaxe: hohe Objekte verschieben sich staerker als der
Boden.

Verfahren:
  1. SIFT-Korrespondenzen zwischen zwei Frames.
  2. Homographie per RANSAC. Sie beschreibt die Abbildung einer *Ebene* -- bei
     Nadiraufnahmen im Wesentlichen den Bodenbereich -- und schluckt zugleich
     Rotation und Zoom der Kamera.
  3. Dichter optischer Fluss zwischen Frame A und dem homographie-entzerrten
     Frame B.
  4. Was jetzt an Restfluss bleibt, ist die Parallaxe. Ihr Betrag waechst mit der
     Hoehe ueber der angepassten Ebene -- das ist ein gemessenes Ersatz-CHM.

Die Skala ist unbekannt (ohne Kamerakalibrierung), aber fuer Wipfelsuche und
Watershed reicht relative Hoehe voellig -- genau wie beim monokularen Ersatz-CHM,
nur eben gemessen statt geraten.

Beispiel:
    python stereo_probe.py --folders 80m dense1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from depth_probe import hillshade, normalize
from infer_species import IMAGE_SUFFIXES, REPO_ROOT


def match_frames(gray_a: np.ndarray, gray_b: np.ndarray, max_features: int):
    """SIFT-Korrespondenzen mit Ratio-Test."""
    sift = cv2.SIFT_create(nfeatures=max_features)
    kp_a, desc_a = sift.detectAndCompute(gray_a, None)
    kp_b, desc_b = sift.detectAndCompute(gray_b, None)
    if desc_a is None or desc_b is None or len(kp_a) < 20 or len(kp_b) < 20:
        return np.empty((0, 2)), np.empty((0, 2))

    matcher = cv2.BFMatcher()
    pairs = matcher.knnMatch(desc_a, desc_b, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 10:
        return np.empty((0, 2)), np.empty((0, 2))

    return (
        np.float32([kp_a[m.queryIdx].pt for m in good]),
        np.float32([kp_b[m.trainIdx].pt for m in good]),
    )


def parallax_map(image_a: np.ndarray, image_b: np.ndarray, args) -> tuple[np.ndarray, dict] | None:
    """Restfluss nach Homographie-Entzerrung = Parallaxe."""
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

    points_a, points_b = match_frames(gray_a, gray_b, args.max_features)
    if len(points_a) < 20:
        return None

    homography, inliers = cv2.findHomography(points_b, points_a, cv2.RANSAC, args.ransac_thresh)
    if homography is None:
        return None

    displacement = np.linalg.norm(points_a - points_b, axis=1)
    stats = {
        "korrespondenzen": len(points_a),
        "inlier": int(inliers.sum()),
        "verschiebung_median_px": float(np.median(displacement)),
        "verschiebung_p90_px": float(np.percentile(displacement, 90)),
    }

    warped = cv2.warpPerspective(gray_b, homography, (gray_a.shape[1], gray_a.shape[0]))

    # Optischer Fluss auf dem entzerrten Paar: der globale Anteil ist raus, was
    # bleibt ist hoehenbedingt. Der Restfluss betraegt nur wenige Pixel, deshalb
    # ist Subpixelgenauigkeit entscheidend -- Farneback mit grossem Fenster
    # verschmiert genau die Kronendetails, auf die es ankommt.
    if args.flow == "dis":
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        dis.setFinestScale(args.finest_scale)
        dis.setPatchSize(args.patch_size)
        dis.setUseSpatialPropagation(True)
        flow = dis.calc(gray_a, warped, None)
    else:
        flow = cv2.calcOpticalFlowFarneback(
            gray_a, warped, None,
            pyr_scale=0.5, levels=args.levels, winsize=args.winsize,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
    residual = np.linalg.norm(flow, axis=2)

    # Bereiche ohne Ueberlappung (schwarz nach der Warpung) ausblenden.
    valid = warped > 0
    residual = np.where(valid, residual, 0.0)

    stats["restfluss_median_px"] = float(np.median(residual[valid])) if valid.any() else 0.0
    stats["restfluss_p95_px"] = float(np.percentile(residual[valid], 95)) if valid.any() else 0.0
    stats["ueberlappung"] = float(valid.mean())
    return residual, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--folders", nargs="*", default=None, help="Ordner; None = alle.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_stereo")
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--ransac-thresh", type=float, default=3.0)
    parser.add_argument("--flow", choices=("dis", "farneback"), default="dis",
                        help="DIS ist subpixelgenauer und loest Kronendetail auf.")
    parser.add_argument("--finest-scale", type=int, default=0, help="0 = feinste Stufe, mehr Detail.")
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--winsize", type=int, default=41, help="Farneback-Fenster; gross = glatter.")
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--smooth", type=float, default=2.5, help="Glaettung der Parallaxenkarte.")
    parser.add_argument("--pair-stride", type=int, default=1,
                        help="Abstand der Paare in der Frameliste. Groesser = laengere Basislinie, "
                             "also staerkere Parallaxe, aber weniger Ueberlappung.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    folders = (
        [args.input / f for f in args.folders]
        if args.folders
        else sorted(p for p in args.input.iterdir() if p.is_dir())
    )

    for folder in folders:
        frames = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if len(frames) < 2:
            print(f"{folder.name}: zu wenige Frames")
            continue

        out_folder = args.out / folder.name
        out_folder.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {folder.name} ===")

        pairs = list(zip(frames, frames[args.pair_stride :]))
        for first, second in pairs:
            image_a, image_b = cv2.imread(str(first)), cv2.imread(str(second))
            if image_a is None or image_b is None or image_a.shape != image_b.shape:
                print(f"  {first.stem} -> {second.stem}: Groessen passen nicht")
                continue

            result = parallax_map(image_a, image_b, args)
            if result is None:
                print(f"  {first.stem} -> {second.stem}: zu wenige Korrespondenzen")
                continue

            residual, stats = result
            smoothed = cv2.GaussianBlur(residual, (0, 0), args.smooth)
            surface = normalize(smoothed)

            stem = f"{first.stem}__{second.stem}"
            np.save(out_folder / f"{stem}_parallax.npy", smoothed)
            cv2.imwrite(str(out_folder / f"{stem}_parallax.jpg"),
                        cv2.applyColorMap((surface * 255).astype(np.uint8), cv2.COLORMAP_TURBO))
            cv2.imwrite(str(out_folder / f"{stem}_hillshade.jpg"),
                        (hillshade(surface) * 255).astype(np.uint8))

            print(
                f"  {first.stem} -> {second.stem}: "
                f"{stats['inlier']}/{stats['korrespondenzen']} Inlier, "
                f"Verschiebung {stats['verschiebung_median_px']:.0f} px (p90 {stats['verschiebung_p90_px']:.0f}), "
                f"Restfluss {stats['restfluss_median_px']:.2f} px (p95 {stats['restfluss_p95_px']:.2f}), "
                f"Ueberlappung {stats['ueberlappung']:.0%}"
            )

    print(f"\nParallaxenkarten in {args.out}")


if __name__ == "__main__":
    main()
