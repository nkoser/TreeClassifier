"""Gemessene Hoehenkarten aus einem Drohnenvideo statt geschaetzter Tiefe.

Zwei Dinge sind an der monokularen Tiefe gescheitert, beide gemessen: sie kann
verschmolzene Kronen nur begrenzt trennen (+0.030 F1), und sie unterscheidet
Rasen nicht von Kronendach -- der Filterversuch dagegen brachte nichts. Beides
sind Aufgaben fuer eine *gemessene* Hoehe, und ein Video liefert sie.

Die Drohne bewegt sich zwischen zwei Frames. Hohe Objekte verschieben sich dabei
staerker als der Boden, und dieser Restfluss nach Abzug der Bodenebene ist echte
Parallaxe -- ein gemessenes Ersatz-CHM, keine Schaetzung. Das Verfahren dafuer
steht bereits in `stereo_probe.parallax_map`; hier kommt nur die Auswahl der
Bildpaare dazu.

Unterschied zu `build_parallax.py`: das probiert alle Paare eines Ordners durch
und haelt alle Bilder im Speicher. Bei vier Frames je Ordner geht das, bei einem
Video mit 2224 Frames nicht -- quadratisch viele Paare. Hier bekommt jeder Frame
stattdessen feste Partner in definiertem zeitlichem Abstand.

Die Wahl der Abstaende ist der eigentliche Parameter: zu nah und es gibt keine
Basislinie (die Drohne stand oder bewegte sich kaum), zu weit und die Bilder
ueberlappen nicht mehr genug fuer eine gemeinsame Homographie. Deshalb mehrere
Abstaende gleichzeitig, jeder auf seine Basislinie normiert und gemittelt.

    python crownseg/video_parallax.py --video /cold/Mahfuz/DJI_...MP4 --count 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from depth_probe import hillshade, normalize  # noqa: E402
from stereo_probe import parallax_map  # noqa: E402


class Settings:
    """Was `parallax_map` an Parametern erwartet."""

    def __init__(self, args) -> None:
        self.max_features = args.max_features
        self.ransac_thresh = args.ransac_thresh
        self.flow = args.flow
        self.finest_scale = args.finest_scale
        self.patch_size = args.patch_size
        self.winsize = args.winsize
        self.levels = args.levels
        self.smooth = args.smooth


def destripe(surface: np.ndarray, window: int) -> np.ndarray:
    """Zeilenweisen Versatz entfernen -- standardmaessig aus, weil wirkungslos.

    Der optische Fluss erzeugt schmale waagerechte Baender, die in den Rohframes
    nicht vorhanden sind. Diese Funktion zieht den Ausreisser des Zeilenmedians
    gegen seinen gleitenden Median ab. Im A/B-Test auf derselben Karte aendert
    das nichts (36 auffaellige Zeilen mit und ohne, Amplitude 0.00366 gegen
    0.00368) -- die Baender sind also kein Versatz ganzer Zeilen.

    Wichtiger ist das Ergebnis der Amplitudenmessung: die Baender machen **2.9 %
    der Reliefspanne** aus, das Kronenrelief ist rund 35-mal staerker. Fuer
    Wipfelsuche und Watershed liegen sie im Rauschen. Dass sie in der
    Schattierung so kraeftig aussehen, liegt an der Darstellung -- ein Hillshade
    zeigt Ableitungen, und eine kleine Stoerung mit scharfer Kante erzeugt darin
    mehr Kontrast als eine grosse, weiche Kuppel. Die Funktion bleibt fuer den
    Fall stehen, dass ein anderes Video echte Zeilenversaetze zeigt.
    """
    rows = np.median(surface, axis=1)
    half = window // 2
    padded = np.pad(rows, half, mode="edge")
    baseline = np.array([np.median(padded[i : i + window]) for i in range(len(rows))])
    return (surface - (rows - baseline)[:, None]).astype(np.float32)


def extract(video: Path, out_dir: Path, start: int, stride: int, count: int) -> list[Path]:
    """Jeden n-ten Frame als JPEG ablegen -- der Rest der Pipeline liest Bilder."""
    out_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Video nicht lesbar: {video}")

    paths, index = [], 0
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    while len(paths) < count:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride == 0:
            path = out_dir / f"frame_{start + index:06d}.jpg"
            if not path.exists():
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            paths.append(path)
        index += 1
    capture.release()
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, default=Path("/cold/Mahfuz/DJI_20230506174726_0004_Z_80m.MP4"))
    parser.add_argument("--out", type=Path, default=Path("/scratch/shared/nik/data/treeclf/video"))
    parser.add_argument("--name", default=None, help="Ordnername; Vorgabe ist der Videoname.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=10, help="Jeder n-te Videoframe.")
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--offsets", type=int, nargs="*", default=[2, 4, 8],
                        help="Partnerabstaende in extrahierten Frames, jeweils vor und zurueck.")

    parser.add_argument("--min-displacement", type=float, default=8.0,
                        help="Paare mit weniger Kamerabewegung liefern nur Rauschen.")
    parser.add_argument("--min-overlap", type=float, default=0.5)
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--ransac-thresh", type=float, default=3.0)
    parser.add_argument("--flow", choices=("dis", "farneback"), default="dis")
    parser.add_argument("--finest-scale", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--winsize", type=int, default=41)
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--destripe", type=int, default=0,
                        help="Zeilenkorrektur; gemessen wirkungslos, siehe destripe(). 0 = aus.")
    parser.add_argument("--smooth", type=float, default=2.5)
    parser.add_argument("--detrend-sigma", type=float, default=180.0,
                        help="Nur fuer die Vorschau: Breite des abgezogenen Trends.")
    args = parser.parse_args()

    name = args.name or args.video.stem
    frames_dir = args.out / name / "frames"
    cache_dir = args.out / name / "parallax"
    preview_dir = args.out / name / "vorschau"
    cache_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    paths = extract(args.video, frames_dir, args.start, args.stride, args.count)
    print(f"{len(paths)} Frames aus {args.video.name} (jeder {args.stride}., ab {args.start})", flush=True)
    if len(paths) < max(args.offsets) + 1:
        print("Zu wenige Frames fuer die gewaehlten Abstaende.")
        return

    images = [cv2.imread(str(p)) for p in paths]
    settings = Settings(args)
    written = 0

    for index, target in enumerate(paths):
        # Pixelweise Summe und Zaehler statt einer Liste: ein Partnerframe deckt
        # das Zielbild nur teilweise ab, und `parallax_map` liefert ausserhalb
        # der Ueberlappung Null. Ein einfacher Mittelwert zieht den Wert dort
        # herunter, wo weniger Partner beitragen -- das erzeugt gerade
        # Nahtkanten quer durch die Karte, genau entlang der Bildraender der
        # gewarpten Partner.
        total = np.zeros(images[index].shape[:2], np.float32)
        counts = np.zeros(images[index].shape[:2], np.float32)
        used = []
        for offset in args.offsets:
            for other in (index - offset, index + offset):
                if not 0 <= other < len(paths):
                    continue
                result = parallax_map(images[index], images[other], settings)
                if result is None:
                    continue
                residual, stats = result
                if (stats["verschiebung_median_px"] < args.min_displacement
                        or stats["ueberlappung"] < args.min_overlap):
                    continue
                # Auf Basislinie 1 normieren, sonst dominiert das weiteste Paar.
                valid = residual > 0
                total += np.where(valid, residual / stats["verschiebung_median_px"], 0.0)
                counts += valid
                used.append(f"{other - index:+d}({stats['verschiebung_median_px']:.0f}px)")

        if not used or counts.max() == 0:
            print(f"  {target.stem}: kein brauchbares Paar", flush=True)
            continue

        merged = np.divide(total, counts, out=np.zeros_like(total), where=counts > 0)
        # Loecher ohne jeden Beitrag mit dem Bildmittel fuellen, damit die
        # spaetere Glaettung sie nicht in die Umgebung hineinzieht.
        if (counts == 0).any():
            merged[counts == 0] = float(merged[counts > 0].mean())
        if args.destripe > 1:
            merged = destripe(merged, args.destripe)
        merged = cv2.GaussianBlur(merged, (0, 0), args.smooth).astype(np.float32)
        np.save(cache_dir / f"{name}__{target.stem}.npy", merged)
        # Die Homographie passt eine Ebene an, die den Boden nur naeherungsweise
        # trifft -- uebrig bleibt ein grossflaechiger Verlauf ueber das Bild.
        # Fuer die Vorschau abgezogen; gespeichert wird die rohe Karte, damit
        # nachgelagerte Skripte selbst entscheiden koennen.
        trend = cv2.GaussianBlur(merged, (0, 0), args.detrend_sigma)
        surface = normalize(merged - trend)
        cv2.imwrite(str(preview_dir / f"{target.stem}_parallax.jpg"),
                    cv2.applyColorMap((surface * 255).astype(np.uint8), cv2.COLORMAP_TURBO))
        cv2.imwrite(str(preview_dir / f"{target.stem}_hillshade.jpg"), (hillshade(surface) * 255).astype(np.uint8))
        written += 1
        print(f"  {target.stem}: {len(used)} Paare {' '.join(used)}", flush=True)

    print(f"\n{written} Hoehenkarten -> {cache_dir}")
    print(f"Frames -> {frames_dir}")


if __name__ == "__main__":
    main()
