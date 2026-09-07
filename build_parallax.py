"""Gemessene Hoehenkarte je Frame aus der Parallaxe aller Partnerframes.

stereo_probe.py rechnet die Parallaxe eines einzelnen Framepaars. Fuer die
Pipeline braucht es aber je Frame *eine* Karte in dessen eigenen Koordinaten,
und die soll moeglichst rauscharm sein.

Deshalb hier: fuer jeden Frame A werden alle anderen Frames desselben Ordners als
Partner durchprobiert und die Ergebnisse gemittelt. Zwei Punkte sind dabei
wesentlich:

  Basislinie normieren  Der Restfluss waechst proportional zur Kamerabewegung.
                        Ohne Normierung dominiert das Paar mit der laengsten
                        Basislinie den Mittelwert. Geteilt wird deshalb durch die
                        mediane Verschiebung des Paars -- danach sind alle Karten
                        auf derselben (weiterhin unbekannten) Hoehenskala.
  Paare aussortieren    Wo die Drohne schwebte, gibt es keine Parallaxe. Paare
                        unterhalb einer Mindestverschiebung liefern nur Rauschen
                        und werden verworfen.

Ergebnis ist ein Cache im selben Format wie der Tiefencache, sodass die
bestehenden Skripte ihn direkt verwenden koennen.

Beispiel:
    python build_parallax.py --out /scratch/shared/nik/data/treeclf/parallax_cache
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from depth_probe import hillshade, normalize
from infer_species import IMAGE_SUFFIXES, REPO_ROOT
from stereo_probe import parallax_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--folders", nargs="*", default=None)
    parser.add_argument("--out", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/parallax_cache"))
    parser.add_argument("--preview", type=Path, default=REPO_ROOT / "results_parallax")

    parser.add_argument("--min-displacement", type=float, default=15.0,
                        help="Mindestverschiebung in px. Darunter schwebte die Drohne -- kein Signal.")
    parser.add_argument("--min-overlap", type=float, default=0.6)
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--ransac-thresh", type=float, default=3.0)
    parser.add_argument("--flow", choices=("dis", "farneback"), default="dis")
    parser.add_argument("--finest-scale", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--winsize", type=int, default=41)
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--smooth", type=float, default=2.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.preview.mkdir(parents=True, exist_ok=True)

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

        images = {p: cv2.imread(str(p)) for p in frames}
        preview_folder = args.preview / folder.name
        preview_folder.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {folder.name} ===")

        for target in frames:
            image_a = images[target]
            accumulated, used = [], []

            for partner in frames:
                if partner is target or images[partner] is None:
                    continue
                if image_a is None or images[partner].shape != image_a.shape:
                    continue

                result = parallax_map(image_a, images[partner], args)
                if result is None:
                    continue
                residual, stats = result
                if (
                    stats["verschiebung_median_px"] < args.min_displacement
                    or stats["ueberlappung"] < args.min_overlap
                ):
                    continue

                # Auf Basislinie 1 normieren, damit lange und kurze Paare gleich
                # gewichtet in den Mittelwert eingehen.
                accumulated.append(residual / stats["verschiebung_median_px"])
                used.append((partner.stem, stats["verschiebung_median_px"], stats["ueberlappung"]))

            if not accumulated:
                print(f"  {target.stem}: kein brauchbares Paar")
                continue

            merged = cv2.GaussianBlur(np.mean(accumulated, axis=0), (0, 0), args.smooth)
            np.save(args.out / f"{folder.name}__{target.stem}.npy", merged.astype(np.float32))

            surface = normalize(merged)
            cv2.imwrite(str(preview_folder / f"{target.stem}_parallax.jpg"),
                        cv2.applyColorMap((surface * 255).astype(np.uint8), cv2.COLORMAP_TURBO))
            cv2.imwrite(str(preview_folder / f"{target.stem}_hillshade.jpg"),
                        (hillshade(surface) * 255).astype(np.uint8))

            partners = ", ".join(f"{name} ({disp:.0f}px)" for name, disp, _ in used)
            print(f"  {target.stem}: {len(used)} Partner -> {partners}")

    print(f"\nParallaxen-Cache: {args.out}")
    print(f"Vorschau: {args.preview}")


if __name__ == "__main__":
    main()
