"""Den Bildwinkel unserer Kamera rueckwaerts aus den Frames bestimmen.

Die 73.7 Grad im Projekt sind ein Vorgabewert im Code, keine Kameraangabe. Der
Wert geht **linear** in jede Tiefe ein: `d = k / D` mit `k = 0.5 / tan(HFOV/2)`.
Ist er falsch, ist jede Hoehe um denselben Faktor falsch -- und man sieht es der
Tiefenkarte nicht an.

Ohne EXIF laesst er sich trotzdem eingrenzen, wenn die Flughoehe bekannt ist.
Bei Nadirblick ist die tiefste Stelle im Bild ungefaehr der Boden, also

    k = boden_faktor * H / p95(1 / D)

`boden_faktor` ist noetig, weil im geschlossenen Kronendach eben **kein** Boden
zu sehen ist -- die tiefste sichtbare Stelle liegt darueber. Aus den
FORTRESS-Hoehenmodellen gemessen (rein geometrisch, ohne jedes Modell) liegt sie
im Median bei 0.917 der Flughoehe, mit einem 5.-bis-95.-Perzentil von 0.82 bis
0.99. Ohne diese Korrektur faellt der Bildwinkel zu klein aus.

Das ist kein Ersatz fuer eine echte Kalibrierung, denn es haengt daran, dass das
Tiefenmodell stimmt. Es hat aber eine eingebaute Probe: **verschiedene
Flughoehen muessen denselben Bildwinkel ergeben.** Tun sie das, stuetzen sich
zwei unabhaengige Messungen gegenseitig. Tun sie es nicht, stimmt entweder das
Modell nicht oder die angegebenen Flughoehen.

Mit dem so bestimmten Wert werden anschliessend die uebrigen Ordner
durchgerechnet, deren Flughoehe niemand kennt.

    python depthft/kalibrieren.py --bekannt 80m=80 100=100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inferenz  # noqa: E402

BILDENDUNGEN = {".jpg", ".jpeg", ".png"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--ft", type=Path, default=Path("/scratch/shared/nik/runs/depthft/bestes"))
    parser.add_argument("--pur", default="apple/DepthPro-hf")
    parser.add_argument("--out", type=Path,
                        default=Path("/home/nik/workspace/TreeClassifier/results_depthft_frames"))
    parser.add_argument("--bekannt", nargs="*", default=["80m=80", "100=100"],
                        metavar="ORDNER=HOEHE", help="Ordner, deren Flughoehe feststeht.")
    parser.add_argument("--boden-perzentil", type=float, default=95.0,
                        help="Welche Stelle im Bild als Boden gilt.")
    parser.add_argument("--boden-faktor", type=float, default=0.917,
                        help="Tiefe der tiefsten sichtbaren Stelle im Verhaeltnis zur "
                             "Flughoehe. Aus den FORTRESS-Hoehenmodellen gemessen; 1.0 "
                             "hiesse, der Boden waere ueberall zu sehen.")
    parser.add_argument("--modell-skalenfehler", type=float, default=0.945,
                        help="Wie das Modell auf den Testgebieten im Median danebenliegt. "
                             "Wird herausgerechnet, damit der eigene Bias nicht als "
                             "Kameraeigenschaft erscheint.")
    parser.add_argument("--modelle", nargs="*", default=["feinabgestimmt", "pur"])
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    bekannt = {}
    for eintrag in args.bekannt:
        name, wert = eintrag.split("=", 1)
        bekannt[name] = float(wert)
    args.out.mkdir(parents=True, exist_ok=True)

    ordner = sorted(p for p in args.input.iterdir() if p.is_dir())
    frames = [(o.name, f) for o in ordner
              for f in sorted(o.iterdir()) if f.suffix.lower() in BILDENDUNGEN]
    print(f"{len(frames)} Frames aus {len(ordner)} Ordnern | bekannt: "
          f"{', '.join(f'{n}={h:.0f} m' for n, h in bekannt.items())}\n", flush=True)

    quellen = {"feinabgestimmt": str(args.ft), "pur": args.pur}
    zeilen = []
    for name in args.modelle:
        if name == "feinabgestimmt" and not args.ft.exists():
            continue
        print(f"--- {name} ---", flush=True)
        model = inferenz.lade(quellen[name], device, fov_head=False)
        for ordnername, pfad in frames:
            bild = cv2.cvtColor(cv2.imread(str(pfad)), cv2.COLOR_BGR2RGB)
            D, _ = inferenz.roh(model, bild, device=device)
            # d = k/D, also p95(d) = k * p95(1/D). Die Kamerakonstante faellt
            # damit aus einer einzigen Messung heraus, sobald H bekannt ist.
            kehrwert = float(np.percentile(1.0 / D, args.boden_perzentil))
            spanne = kehrwert - float(np.percentile(1.0 / D, 2.0))
            zeilen.append({"modell": name, "ordner": ordnername, "frame": pfad.name,
                           "kehrwert_p95": kehrwert, "kehrwert_spanne": spanne,
                           "flughoehe_bekannt": bekannt.get(ordnername, np.nan)})
        del model
        torch.cuda.empty_cache()

    tabelle = pd.DataFrame(zeilen)
    # Zwei Korrekturen, beide unabhaengig gemessen: die tiefste sichtbare Stelle
    # ist nicht der Boden, und das Modell hat einen bekannten Restbias.
    korrektur = args.boden_faktor / max(args.modell_skalenfehler, 1e-6)
    tabelle["k_geschaetzt"] = korrektur * tabelle["flughoehe_bekannt"] / tabelle["kehrwert_p95"]
    tabelle["fov_geschaetzt"] = tabelle["k_geschaetzt"].apply(
        lambda k: inferenz.fov_von_k(k) if np.isfinite(k) and k > 0 else np.nan)
    tabelle.to_csv(args.out / "kalibrierung.csv", index=False)

    print(f"\nKorrekturen: Bodenfaktor {args.boden_faktor:.3f}, "
          f"Modellbias {args.modell_skalenfehler:.3f} -> zusammen {korrektur:.3f}")
    print("\n=== Bildwinkel rueckwaerts, je Ordner mit bekannter Flughoehe ===")
    teil = tabelle[tabelle["flughoehe_bekannt"].notna()]
    je_ordner = teil.groupby(["modell", "ordner"]).agg(
        n=("frame", "size"), H=("flughoehe_bekannt", "first"),
        k=("k_geschaetzt", "median"), fov=("fov_geschaetzt", "median"),
        fov_streuung=("fov_geschaetzt", "std"))
    print(je_ordner.to_string(float_format=lambda v: f"{v:8.2f}"))

    print("\n=== Probe: ergeben verschiedene Flughoehen denselben Bildwinkel? ===")
    for modell in teil["modell"].unique():
        werte = je_ordner.loc[modell, "fov"]
        spanne = float(werte.max() - werte.min())
        urteil = ("stimmig" if spanne < 5 else "grenzwertig" if spanne < 12 else "widerspruechlich")
        print(f"  {modell:15s} {', '.join(f'{o}: {v:.1f}°' for o, v in werte.items())} "
              f"| Spanne {spanne:.1f}° -> {urteil}")

    empfohlen = float(teil[teil["modell"] == args.modelle[0]]["fov_geschaetzt"].median())
    k_empf = inferenz.k_von_fov(empfohlen)
    print(f"\nEmpfohlener Bildwinkel ({args.modelle[0]}): {empfohlen:.1f} Grad, k = {k_empf:.4f}")
    print(f"Bisher angenommen: 73.7 Grad, k = {inferenz.k_von_fov(73.7):.4f} "
          f"-- Tiefen waeren um Faktor {k_empf/inferenz.k_von_fov(73.7):.2f} zu korrigieren.")
    # Das 5.-bis-95.-Perzentil des Bodenfaktors als Unsicherheitsband: es ist die
    # groesste bekannte Fehlerquelle dieser Rueckrechnung.
    for name, faktor in (("Boden gut sichtbar", 0.99), ("Kronendach dicht", 0.82)):
        k_alt = k_empf * faktor / args.boden_faktor
        print(f"  waere {name:22s} ({faktor:.2f}): {inferenz.fov_von_k(k_alt):5.1f} Grad")
    print("Belastbar wird der Wert erst durch EXIF oder eine echte Kalibrierung "
          "(z.B. AnyCam auf den Originalvideos) -- diese Rueckrechnung setzt voraus, "
          "dass das Tiefenmodell stimmt.")

    print("\n=== Damit alle Ordner durchgerechnet ===")
    tabelle["flughoehe_neu"] = k_empf * tabelle["kehrwert_p95"]
    tabelle["kronenhoehe_neu"] = k_empf * tabelle["kehrwert_spanne"]
    uebersicht = tabelle[tabelle["modell"] == args.modelle[0]].groupby("ordner").agg(
        n=("frame", "size"), flughoehe_m=("flughoehe_neu", "mean"),
        kronenhoehe_m=("kronenhoehe_neu", "mean"), kronenhoehe_streuung=("kronenhoehe_neu", "std"))
    uebersicht["bekannt"] = [bekannt.get(o, np.nan) for o in uebersicht.index]
    print(uebersicht.to_string(float_format=lambda v: f"{v:9.1f}"))
    uebersicht.to_csv(args.out / "flughoehen_geschaetzt.csv")
    tabelle.to_csv(args.out / "kalibrierung.csv", index=False)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
