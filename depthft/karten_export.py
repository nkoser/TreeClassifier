"""Die Tiefen- und Hoehenkarten des feinabgestimmten Modells als Dateien ablegen.

Gedacht zum Weiterverarbeiten, nicht zum Anschauen. Jede Karte kommt in drei
Fassungen, weil die Ansprueche verschieden sind:

    tiefe_m/   *.npy   float32, Meter. Abstand zur Kamera. Verlustfrei.
    hoehe_m/   *.npy   float32, Meter ueber Boden -- Flughoehe minus Tiefe.
                       Das ist die Zahl, aus der Baumhoehen werden.
    tiefe_cm/  *.png   uint16, Zentimeter. Fuer alles, was kein npy liest;
                       1 cm Aufloesung, bis 655 m.
    vorschau/  *.jpg   Hoehe ueber Boden, farbig, mit Farbkeil und Werteangabe.
                       Nur zum Draufschauen -- zum Rechnen die npy nehmen.

Die Hoehe ueber Boden haengt an der angenommenen Flughoehe, die Tiefe nicht.
Wo die Flughoehe geschaetzt ist, steht das in `karten.csv` -- die Tiefenkarte
bleibt davon unberuehrt.

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
    """Warnen, wenn ein Bild nicht aus derselben Kamera stammen kann.

    Der vorgegebene Bildwinkel gilt fuer die Frames der Drohne, 1920x1080. Ein
    Bildschirmfoto oder ein zugeschnittenes Bild hat einen anderen Ausschnitt und
    damit einen anderen Bildwinkel -- die Tiefen waeren um einen unbekannten
    Faktor falsch, ohne dass man es dem Ergebnis ansieht. Im Ordner `urban`
    lagen genau solche Dateien.
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
                        help="tiefe: ueber die Tiefe, mit Gelaendemodell. hoehe: ein Checkpoint, "
                             "der Meter unmittelbar ausgibt -- lokal etwas genauer, unterscheidet "
                             "aber nicht zwischen Bestaenden (r = -0.16), weil ihm die "
                             "Massstabsreferenz fehlt.")
    parser.add_argument("--out", type=Path,
                        default=Path("/home/nik/workspace/TreeClassifier/results_depthft_karten"))
    parser.add_argument("--hfov-deg", type=float, default=48.0)
    parser.add_argument("--altitude", type=float, default=100.0)
    parser.add_argument("--altitudes", nargs="*", metavar="ORDNER=HOEHE",
                        default=["dense=51", "dense1=69", "mixed=92", "mixed1=103",
                                 "pines=60", "urban=120"])
    parser.add_argument("--zip", action="store_true", help="Zusaetzlich ein zip danebenlegen.")
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
        # 0 bleibt als "kein Wert" frei, deshalb erst ab 1 cm.
        cm = np.clip(np.round(tiefe * 100.0), 1, 65535).astype(np.uint16)
        cv2.imwrite(str(args.out / "tiefe_cm" / f"{name}.png"), cm)
        # Farbkeil und Werteangabe gehoeren ins Bild: ohne sie sieht man zwar,
        # wo es hoch ist, aber nicht wie hoch -- und weil jede Vorschau auf
        # ihren eigenen Wertebereich gespreizt ist, bedeutet dieselbe Farbe in
        # zwei Frames Verschiedenes.
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
