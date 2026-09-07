"""Write the depth and height maps of the fine-tuned model out as files.

Intended for further processing, not for looking at. Every map comes in three
versions, because the requirements differ:

    tiefe_m/   *.npy   float32, metres. Distance to the camera. Lossless.
    hoehe_m/   *.npy   float32, metres above ground -- altitude minus depth.
                       This is the number tree heights come from.
    tiefe_cm/  *.png   uint16, centimetres. For anything that cannot read npy;
                       1 cm resolution, up to 655 m.
    vorschau/  *.jpg   Height above ground, in colour, with a wedge and values.
                       For looking only -- use the npy for computation.

The height above ground depends on the assumed flight altitude, the depth does
not. Where the altitude is estimated, `karten.csv` says so -- the depth map is
unaffected by it.

    python depthft/karten_export.py --hfov-deg 48.0
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inferenz  # noqa: E402
from bilder import beschriften, farbskala, hoehenbild  # noqa: E402

BILDENDUNGEN = {".jpg", ".jpeg", ".png"}

LIESMICH = """Tiefen- und Hoehenkarten, Depth Pro feinabgestimmt auf FORTRESS
================================================================

Erzeugt mit depthft/karten_export.py aus dem TreeClassifier-Projekt.

Ordner
------
tiefe_m/    .npy, float32, Meter. Abstand von der Kamera zur Oberflaeche.
            Haengt NICHT von einer Annahme ueber die Flughoehe ab.
hoehe_m/    .npy, float32, Meter ueber Boden = Flughoehe - Tiefe.
            Haengt an der Flughoehe; wo die geschaetzt ist, steht es in karten.csv.
tiefe_cm/   .png, uint16, Zentimeter. Fuer Programme ohne npy-Unterstuetzung.
            Wert 0 heisst "kein Wert". Meter = Wert / 100.
vorschau/   .jpg, Hoehe ueber Boden, farbig, mit Farbkeil in Metern.
            Dunkelblau/lila = niedrig (Boden, Kronenluecken), gelb = hoch (Wipfel).
            ACHTUNG: jede Vorschau ist auf ihren EIGENEN Wertebereich gespreizt,
            der oben im Bild steht. Dieselbe Farbe bedeutet in zwei Frames also
            Verschiedenes -- zum Vergleichen die npy-Dateien nehmen.

Laden
-----
    import numpy as np
    tiefe = np.load("tiefe_m/80m_frame_000297.npy")     # Meter
    hoehe = np.load("hoehe_m/80m_frame_000297.npy")     # Meter ueber Boden

    import cv2
    tiefe = cv2.imread("tiefe_cm/80m_frame_000297.png", cv2.IMREAD_UNCHANGED) / 100.0

Kamera
------
Alle Karten sind mit einem horizontalen Bildwinkel von {hfov:.1f} Grad gerechnet.
Dieser Wert wurde rueckwaerts aus Frames mit bekannter Flughoehe bestimmt, nicht
aus EXIF gelesen. Er geht LINEAR in jede Tiefe ein: ist er falsch, sind alle
Werte um denselben Faktor falsch. Unsicherheit rund 45 bis 53 Grad.

Ohne bekannte Flughoehe bleibt die Kronenhoehe ablesbar, denn sie ist eine
Differenz und braucht keinen Bezugspunkt:

    Kronenhoehe = 95. Perzentil der Tiefe - 2. Perzentil der Tiefe

Guete
-----
Auf FORTRESS-Testgebieten, die im Training nicht vorkamen: mittlerer relativer
Fehler 12 %, mittlerer absoluter Fehler 7.4 m bei Flughoehen von 40 bis 120 m.
Pures Depth Pro liegt dort bei 98 %.

Datenherkunft: FORTRESS, Schiefer, Frey & Kattenborn 2022, CC BY 4.0.
"""


def kamera_pruefen(bild, pfad, erwartete_breite: int = 1920, erwartetes_verhaeltnis: float = 16 / 9,
                   toleranz: float = 0.02) -> str | None:
    """Warn when an image cannot come from the same camera.

    The supplied field of view holds for the drone frames, 1920x1080. A screenshot
    or a cropped image has a different framing and therefore a different field of
    view -- the depths would be wrong by an unknown factor without it showing in
    the result. The folder `urban` contained exactly such files.
    """
    h, w = bild.shape[:2]
    verhaeltnis = w / max(h, 1)
    if w == erwartete_breite and abs(verhaeltnis - erwartetes_verhaeltnis) < toleranz:
        return None
    return (f"{pfad.name}: {w}x{h} (Verhaeltnis {verhaeltnis:.2f}) statt "
            f"{erwartete_breite}x{int(erwartete_breite / erwartetes_verhaeltnis)} -- "
            f"der vorgegebene Bildwinkel gilt hier vermutlich nicht.")


def flughoehe_von(ordner: str, vorgaben: dict[str, float], rueckfall: float) -> tuple[float, str]:
    if ordner in vorgaben:
        return vorgaben[ordner], "geschaetzt"
    treffer = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*m?\s*", ordner, re.IGNORECASE)
    if treffer:
        return float(treffer.group(1).replace(",", ".")), "ordnername"
    return rueckfall, "rueckfall"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--ft", type=Path, default=Path("/scratch/shared/nik/runs/depthft/bestes"))
    parser.add_argument("--modellart", default="tiefe", choices=("tiefe", "hoehe"),
                        help="tiefe: via the depth, with a terrain model. hoehe: a checkpoint "
                             "that outputs metres directly -- locally somewhat more accurate, "
                             "but it does not distinguish between stands (r = -0.16), because "
                             "it lacks the scale reference.")
    parser.add_argument("--out", type=Path,
                        default=Path("/home/nik/workspace/TreeClassifier/results_depthft_karten"))
    parser.add_argument("--hfov-deg", type=float, default=48.0)
    parser.add_argument("--altitude", type=float, default=100.0)
    parser.add_argument("--altitudes", nargs="*", metavar="ORDNER=HOEHE",
                        default=["dense=51", "dense1=69", "mixed=92", "mixed1=103",
                                 "pines=60", "urban=120"])
    parser.add_argument("--zip", action="store_true", help="Also place a zip next to it.")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    vorgaben = {e.split("=")[0]: float(e.split("=")[1]) for e in args.altitudes}
    k = inferenz.k_von_fov(args.hfov_deg)

    for unter in ("tiefe_m", "hoehe_m", "tiefe_cm", "vorschau"):
        (args.out / unter).mkdir(parents=True, exist_ok=True)

    ordner = sorted(p for p in args.input.iterdir() if p.is_dir())
    frames = [(o.name, f) for o in ordner
              for f in sorted(o.iterdir()) if f.suffix.lower() in BILDENDUNGEN]
    print(f"{len(frames)} Frames | Bildwinkel {args.hfov_deg} Grad, k = {k:.4f}", flush=True)

    model = inferenz.lade(str(args.ft), device, fov_head=False)
    zeilen = []
    warnungen: list[str] = []
    for ordnername, pfad in frames:
        bild = cv2.cvtColor(cv2.imread(str(pfad)), cv2.COLOR_BGR2RGB)
        hinweis = kamera_pruefen(bild, pfad)
        if hinweis:
            warnungen.append(hinweis)
            print(f"  ACHTUNG {hinweis}", flush=True)
        H, quelle = flughoehe_von(ordnername, vorgaben, args.altitude)
        tiefe, hoehe = inferenz.karten_von(model, bild, art=args.modellart, k=k,
                                           flughoehe_m=H, device=device)
        if hoehe is None:
            hoehe = (H - tiefe).astype(np.float32)
        name = f"{ordnername}_{pfad.stem}"

        np.save(args.out / "tiefe_m" / f"{name}.npy", tiefe)
        np.save(args.out / "hoehe_m" / f"{name}.npy", hoehe)
        # 0 stays reserved for "no value", hence starting at 1 cm.
        cm = np.clip(np.round(tiefe * 100.0), 1, 65535).astype(np.uint16)
        cv2.imwrite(str(args.out / "tiefe_cm" / f"{name}.png"), cm)
        # The wedge and the values belong in the image: without them you can see
        # where it is high but not how high -- and because every preview is
        # stretched to its own value range, the same colour means different things
        # in two frames.
        unten, oben = float(np.percentile(hoehe, 2)), float(np.percentile(hoehe, 98))
        vorschau = beschriften(hoehenbild(hoehe, oben, unten), name,
                               f"Hoehe ueber Boden {unten:.1f} - {oben:.1f} m | "
                               f"Flughoehe {H:.0f} m ({quelle})")
        keil = farbskala(78, vorschau.shape[0], oben)
        cv2.imwrite(str(args.out / "vorschau" / f"{name}.jpg"),
                    np.hstack([vorschau, keil]), [cv2.IMWRITE_JPEG_QUALITY, 90])

        zeilen.append({
            "datei": name, "ordner": ordnername, "frame": pfad.name,
            "flughoehe_m": H, "flughoehe_quelle": quelle, "hfov_grad": args.hfov_deg,
            "modellart": args.modellart,
            "breite_px": bild.shape[1], "hoehe_px": bild.shape[0],
            "gsd_cm": inferenz.gsd_von_flughoehe(H, args.hfov_deg, bild.shape[1]) * 100,
            "tiefe_min_m": float(tiefe.min()), "tiefe_max_m": float(tiefe.max()),
            "tiefe_p95_m": float(np.percentile(tiefe, 95)),
            "kronenhoehe_m": float(np.percentile(tiefe, 95) - np.percentile(tiefe, 2)),
            "hoehe_p95_m": float(np.percentile(hoehe, 95)),
        })
        print(f"  {name:32s} Tiefe {tiefe.min():5.1f}-{tiefe.max():5.1f} m | "
              f"Kronenhoehe {zeilen[-1]['kronenhoehe_m']:5.1f} m", flush=True)

    tabelle = pd.DataFrame(zeilen)
    tabelle.to_csv(args.out / "karten.csv", index=False)
    (args.out / "LIESMICH.txt").write_text(LIESMICH.format(hfov=args.hfov_deg))

    groesse = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file()) / 1e6
    print(f"\n{len(tabelle)} Karten -> {args.out}  ({groesse:.0f} MB)")
    if warnungen:
        print(f"\n{len(warnungen)} Datei(en) passen nicht zur angenommenen Kamera:")
        for h in warnungen:
            print(f"  {h}")
        print("  Deren Hoehen sind um einen unbekannten Faktor falsch.")
    print(tabelle.groupby("ordner")[["flughoehe_m", "kronenhoehe_m", "hoehe_p95_m"]]
          .mean().to_string(float_format=lambda v: f"{v:8.1f}"))

    if args.zip:
        import shutil
        archiv = shutil.make_archive(str(args.out), "zip", root_dir=args.out.parent,
                                     base_dir=args.out.name)
        print(f"\n{archiv}  ({Path(archiv).stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
