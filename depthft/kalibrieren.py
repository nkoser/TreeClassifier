"""Determine the field of view of our camera backwards from the frames.

The 73.7 degrees in the project are a default in the code, not a camera
specification. The value enters every depth **linearly**: `d = k / D` with
`k = 0.5 / tan(HFOV/2)`. If it is wrong, every height is wrong by the same
factor -- and you cannot see it in the depth map.

Without EXIF it can still be bounded if the flight altitude is known. For a nadir
view the deepest point in the image is roughly the ground, so

    k = boden_faktor * H / p95(1 / D)

`boden_faktor` is necessary because in a closed canopy there is precisely **no**
ground visible -- the deepest visible point lies above it. Measured from the
FORTRESS height models (purely geometrically, without any model) it lies at 0.917
of the flight altitude at the median, with a 5th-to-95th percentile of 0.82 to
0.99. Without this correction the field of view comes out too small.

This is no substitute for a real calibration, because it depends on the depth
model being right. But it has a built-in check: **different flight altitudes have
to give the same field of view.** If they do, two independent measurements
support each other. If they do not, either the model or the stated altitudes are
wrong.

The value determined this way is then used to compute the remaining folders,
whose flight altitude nobody knows.

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
                        metavar="FOLDER=ALTITUDE", help="Folders whose altitude is known.")
    parser.add_argument("--boden-perzentil", type=float, default=95.0,
                        help="Which point in the image counts as the ground.")
    parser.add_argument("--boden-faktor", type=float, default=0.917,
                        help="Depth of the deepest visible point relative to the flight "
                             "altitude. Measured from the FORTRESS height models; 1.0 "
                             "would mean the ground is visible everywhere.")
    parser.add_argument("--modell-skalenfehler", type=float, default=0.945,
                        help="How far the model is off at the median on the test sites. "
                             "It is factored out so that our own bias does not appear as "
                             "a camera property.")
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
            # d = k/D, hence p95(d) = k * p95(1/D). The camera constant therefore
            # falls out of a single measurement as soon as H is known.
            kehrwert = float(np.percentile(1.0 / D, args.boden_perzentil))
            spanne = kehrwert - float(np.percentile(1.0 / D, 2.0))
            zeilen.append({"modell": name, "ordner": ordnername, "frame": pfad.name,
                           "kehrwert_p95": kehrwert, "kehrwert_spanne": spanne,
                           "flughoehe_bekannt": bekannt.get(ordnername, np.nan)})
        del model
        torch.cuda.empty_cache()

    tabelle = pd.DataFrame(zeilen)
    # Two corrections, both measured independently: the deepest visible point is
    # not the ground, and the model has a known residual bias.
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
    # The 5th-to-95th percentile of the ground factor as an uncertainty band: it
    # is the largest known source of error in this back-calculation.
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
