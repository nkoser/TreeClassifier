"""Beide Modelle auf unsere eigenen Drohnenframes -- pur gegen feinabgestimmt.

Hier gibt es keine Wahrheit. Trotzdem laesst sich messen, und zwar an etwas,
das gar keine Hoehenkarte braucht: **die Tiefe zum Boden ist die Flughoehe**.

    Flughoehe geschaetzt = 95. Perzentil der Tiefe
    Kronenhoehe          = 95. Perzentil minus 2. Perzentil der Tiefe

Die zweite Zahl ist die wichtigere, weil sie ohne jede Annahme auskommt: die
Spanne zwischen Boden und Wipfel ist die Baumhoehe, ganz gleich wie hoch die
Drohne wirklich hing. Ein Modell, das den Bestand auf 6 m Hoehe zusammendrueckt,
faellt hier sofort auf -- auch dann, wenn seine Tiefenkarte huebsch aussieht.

Ist die Flughoehe bekannt (aus dem Ordnernamen wie `80m`, oder ueber
`--altitudes`), kommen zwei weitere Pruefungen dazu: das Bodenniveau muss bei
0 m liegen, und kein Bildpunkt darf unter dem Boden sitzen.

    python depthft/apply_frames.py --ft /scratch/shared/nik/runs/depthft/bestes
    python depthft/apply_frames.py --altitudes pines=35 dense=60 urban=50
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
from bilder import beschriften, einfache_abbildung, grauwert, hillshade  # noqa: E402

BILDENDUNGEN = {".jpg", ".jpeg", ".png"}


def flughoehe_von(ordner: str, vorgaben: dict[str, float], rueckfall: float) -> tuple[float, str]:
    """Reihenfolge: --altitudes, dann Zahl im Ordnernamen, dann Rueckfallwert.

    Der Ordnername zaehlt nur, wenn er **ganz** aus der Zahl besteht, optional
    mit angehaengtem `m`: `80m` und `100` sind Hoehenangaben, `mixed1` und
    `dense1` sind es nicht -- dort ist die 1 eine laufende Nummer. Eine Regex,
    die irgendwo im Namen sucht, macht daraus eine Flughoehe von einem Meter.
    """
    if ordner in vorgaben:
        return vorgaben[ordner], "vorgabe"
    treffer = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*m?\s*", ordner, re.IGNORECASE)
    if treffer:
        return float(treffer.group(1).replace(",", ".")), "ordnername"
    return rueckfall, "rueckfall"


def vorgaben_lesen(eintraege: list[str] | None) -> dict[str, float]:
    werte = {}
    for eintrag in eintraege or []:
        if "=" not in eintrag:
            raise SystemExit(f"--altitudes erwartet ORDNER=HOEHE, bekam: {eintrag!r}")
        name, wert = eintrag.split("=", 1)
        werte[name] = float(wert)
    return werte


def pruefen(d: np.ndarray, flughoehe: float) -> dict[str, float]:
    """Kennzahlen aus der Tiefenkarte allein.

    `boden` und `wipfel` sind Perzentile statt Extremwerte: ein einzelnes
    fehlgeschaetztes Pixel -- eine Spiegelung, ein Bildrand -- wuerde sonst die
    ganze Auswertung bestimmen.
    """
    boden, wipfel = float(np.percentile(d, 95)), float(np.percentile(d, 2))
    hoehe = flughoehe - d
    return {
        "flughoehe_geschaetzt_m": boden,
        "kronenhoehe_m": boden - wipfel,
        "tiefe_median_m": float(np.median(d)),
        "bodenfehler_m": boden - flughoehe,
        "hoehe_p50_m": float(np.percentile(hoehe, 50)),
        "hoehe_p95_m": float(np.percentile(hoehe, 95)),
        "anteil_unter_boden": float(np.mean(hoehe < -1.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--ft", type=Path, default=Path("/scratch/shared/nik/runs/depthft/bestes"))
    parser.add_argument("--pur", default="apple/DepthPro-hf")
    parser.add_argument("--out", type=Path,
                        default=Path("/home/nik/workspace/TreeClassifier/results_depthft_frames"))
    parser.add_argument("--folders", nargs="*", default=None, help="Vorgabe: alle Unterordner.")
    parser.add_argument("--hfov-deg", type=float, default=73.7,
                        help="Horizontaler Bildwinkel unserer Kamera. Schaetzwert, "
                             "solange keine EXIF-Angabe vorliegt.")
    parser.add_argument("--altitude", type=float, default=100.0, help="Rueckfall, wenn nichts anderes greift.")
    parser.add_argument("--altitudes", nargs="*", metavar="ORDNER=HOEHE")
    parser.add_argument("--fovkopf", action="store_true",
                        help="Zusaetzlich mit geschaetztem statt vorgegebenem Bildwinkel rechnen.")
    parser.add_argument("--max-kante", type=int, default=0,
                        help="Frames vorher verkleinern; 0 laesst sie in Originalgroesse.")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    vorgaben = vorgaben_lesen(args.altitudes)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "vergleich").mkdir(exist_ok=True)
    (args.out / "hoehenkarten").mkdir(exist_ok=True)
    (args.out / "relief").mkdir(exist_ok=True)

    ordner = sorted(p for p in args.input.iterdir() if p.is_dir())
    if args.folders:
        ordner = [p for p in ordner if p.name in set(args.folders)]
    frames = [(p.parent.name, p) for p in
              sorted(f for o in ordner for f in o.iterdir() if f.suffix.lower() in BILDENDUNGEN)]
    if not frames:
        raise SystemExit(f"Keine Frames unter {args.input}")

    k_vorgabe = inferenz.k_von_fov(args.hfov_deg)
    print(f"Device: {device} | {len(frames)} Frames aus {len(ordner)} Ordnern")
    print(f"Bildwinkel {args.hfov_deg} Grad -> k = {k_vorgabe:.4f}\n", flush=True)

    modelle = [("pur", args.pur)]
    if args.ft.exists():
        modelle.append(("feinabgestimmt", str(args.ft)))
    else:
        print(f"{args.ft} fehlt -- nur das pure Modell laeuft.\n", flush=True)

    zeilen: list[dict] = []
    karten: dict[tuple[str, str], dict] = {}

    for name, quelle in modelle:
        print(f"--- {name}: {quelle} ---", flush=True)
        model = inferenz.lade(quelle, device, fov_head=args.fovkopf)

        for ordnername, pfad in frames:
            bild = cv2.cvtColor(cv2.imread(str(pfad)), cv2.COLOR_BGR2RGB)
            if args.max_kante and max(bild.shape[:2]) > args.max_kante:
                faktor = args.max_kante / max(bild.shape[:2])
                bild = cv2.resize(bild, None, fx=faktor, fy=faktor, interpolation=cv2.INTER_AREA)
            H, quelle_H = flughoehe_von(ordnername, vorgaben, args.altitude)

            D, fov = inferenz.roh(model, bild, device=device)
            varianten = {f"{name}_kamera": (k_vorgabe / D, args.hfov_deg)}
            if fov is not None:
                varianten[f"{name}_fovkopf"] = (inferenz.k_von_fov(fov) / D, fov)

            for variante, (d, fov_benutzt) in varianten.items():
                werte = pruefen(d, H)
                werte.update(variante=variante, ordner=ordnername, frame=pfad.name,
                             flughoehe_m=H, flughoehe_quelle=quelle_H, fov_grad=fov_benutzt,
                             gsd_cm=inferenz.gsd_von_flughoehe(H, fov_benutzt, bild.shape[1]) * 100)
                zeilen.append(werte)

            schluessel = (ordnername, pfad.name)
            eintrag = karten.setdefault(schluessel, {"rgb": bild, "flughoehe": H, "quelle": quelle_H})
            eintrag[name] = H - varianten[f"{name}_kamera"][0]

        del model
        torch.cuda.empty_cache()

    tabelle = pd.DataFrame(zeilen)
    tabelle.to_csv(args.out / "frames.csv", index=False)

    spalten = ["flughoehe_geschaetzt_m", "kronenhoehe_m", "bodenfehler_m", "hoehe_p95_m", "anteil_unter_boden"]
    print("\n=== ueber alle Frames ===")
    print(tabelle.groupby("variante")[spalten].mean().to_string(float_format=lambda v: f"{v:9.2f}"))
    print("\n--- je Ordner (Kronenhoehe in m, ohne Annahme ueber die Flughoehe) ---")
    kreuz = tabelle.pivot_table(index="ordner", columns="variante", values="kronenhoehe_m", aggfunc="mean")
    kreuz.insert(0, "flughoehe_m", tabelle.groupby("ordner")["flughoehe_m"].first())
    kreuz.insert(1, "quelle", tabelle.groupby("ordner")["flughoehe_quelle"].first())
    print(kreuz.to_string(float_format=lambda v: f"{v:8.1f}"))
    kreuz.to_csv(args.out / "kronenhoehe_je_ordner.csv")

    # Wie stark haben sich die Karten geaendert: bleibt die Struktur, verschiebt
    # sich nur der Massstab? Dann waere das Feinabstimmen eine reine Kalibrierung.
    if len(modelle) == 2:
        aehnlich = []
        for eintrag in karten.values():
            a, b = eintrag["pur"].ravel(), eintrag["feinabgestimmt"].ravel()
            schritt = max(1, a.size // 200000)
            aehnlich.append(float(np.corrcoef(a[::schritt], b[::schritt])[0, 1]))
        print(f"\nKorrelation der Hoehenkarten pur/feinabgestimmt: "
              f"Mittel {np.mean(aehnlich):.3f}, Minimum {np.min(aehnlich):.3f}")

    # Jede Karte auf ihren eigenen Wertebereich gespreizt, der Massstab als
    # Balken darunter. Eine gemeinsame Skala waere sachlich richtig, macht die
    # Kachel des puren Modells aber zu einer einfarbigen Flaeche -- es liegt um
    # Faktor 50 daneben, also faellt alles jenseits des Skalenendes zusammen.
    ROT, BLAU = (110, 110, 245), (245, 190, 110)
    farbe_von = {"pur": ROT, "feinabgestimmt": BLAU}
    for (ordnername, dateiname), eintrag in karten.items():
        H = eintrag["flughoehe"]
        vorhanden = [n for n, _ in modelle if n in eintrag]
        abbildung = einfache_abbildung(
            cv2.cvtColor(eintrag["rgb"], cv2.COLOR_RGB2BGR),
            [(n, eintrag[n], farbe_von.get(n, BLAU)) for n in vorhanden],
            f"{ordnername}/{dateiname}",
            f"Flughoehe {H:.0f} m ({eintrag['quelle']})")
        cv2.imwrite(str(args.out / "vergleich" / f"{ordnername}_{Path(dateiname).stem}.jpg"),
                    abbildung, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Das Relief getrennt: es zeigt die Form unabhaengig vom Massstab und
        # gehoert deshalb nicht in dieselbe Abbildung wie die Hoehenwerte.
        relief = [beschriften(grauwert(hillshade(eintrag[n])), f"{n}, Relief") for n in vorhanden]
        if relief:
            cv2.imwrite(str(args.out / "relief" / f"{ordnername}_{Path(dateiname).stem}.jpg"),
                        np.hstack(relief), [cv2.IMWRITE_JPEG_QUALITY, 88])
        for n in vorhanden:
            np.save(args.out / "hoehenkarten" / f"{ordnername}_{Path(dateiname).stem}_{n}.npy",
                    eintrag[n].astype(np.float16))

    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
