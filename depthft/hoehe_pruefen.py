"""The whole chain against the truth: depth, terrain model, height above ground.

On our own frames it cannot be checked whether the terrain model is right --
there is no truth. On the FORTRESS test sites it can: there the nDSM carries the
height above ground for every pixel.

Three routes to height are compared:

  from the terrain model  `Z = ground(estimated) - d`. The route the point
                          clouds take. Needs no flight altitude.
  from the altitude       `Z = H - d`. Presumes H is known, and does not carry
                          terrain slope.
  depth alone             `d` against the true depth -- separates errors of the
                          depth model from errors of the terrain model.
  direct height model     A second model predicting the height directly instead
                          of computing it from the depth (`--hoehenmodell`).
                          Needs neither field of view nor ground reference.

In addition the ground factor is swept. It compensates for the fact that the
deepest **visible** point in a closed stand lies above the real ground; how large
it has to be is directly measurable here rather than estimated.

    python depthft/hoehe_pruefen.py --n 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inferenz  # noqa: E402
from dataset import NadirFrames  # noqa: E402
from punktwolke import bodenmodell  # noqa: E402


def fehler(vorher: np.ndarray, wahr: np.ndarray, maske: np.ndarray) -> dict[str, float]:
    a, b = vorher[maske], wahr[maske]
    if a.size == 0:
        return {}
    return {"mae_m": float(np.mean(np.abs(a - b))), "bias_m": float(np.mean(a - b)),
            "rmse_m": float(np.sqrt(np.mean((a - b) ** 2))),
            "korrelation": float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan,
            # Recorded per crop, so as to distinguish later whether the model
            # differentiates *within* an image or also *between* stands. A model
            # that always guesses the training mean can look good within an image
            # and be blind between stands.
            "p95_vorher": float(np.percentile(a, 95)), "p95_wahr": float(np.percentile(b, 95))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("/scratch/shared/nik/data/fortress/depthft"))
    parser.add_argument("--ft", type=Path, default=Path("/scratch/shared/nik/runs/depthft/bestes"))
    parser.add_argument("--out", type=Path,
                        default=Path("/home/nik/workspace/TreeClassifier/results_depthft"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=40, help="Crops per site.")
    parser.add_argument("--kachel-m", type=float, nargs="*", default=[10.0, 15.0, 25.0])
    parser.add_argument("--boden-faktoren", type=float, nargs="*",
                        default=[1.0, 0.96, 0.917, 0.88])
    parser.add_argument("--perzentil", type=float, default=97.0)
    parser.add_argument("--hoehenmodell", type=Path, default=None,
                        help="Checkpoint from finetune_hoehe.py; predicts metres directly.")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    daten = NadirFrames(args.data, args.split, crop_px=1536, pro_gebiet=args.n,
                        augment=False, cache=2, seed=args.seed, fov_min=60.0, fov_max=85.0)
    print(f"{len(daten)} Ausschnitte aus {len(daten.sites)} Gebieten", flush=True)
    model = inferenz.lade(str(args.ft), device, fov_head=False)
    hoehenmodell = None
    if args.hoehenmodell and args.hoehenmodell.exists():
        hoehenmodell = inferenz.lade(str(args.hoehenmodell), device, fov_head=False)
        print(f"Dazu das direkte Hoehenmodell: {args.hoehenmodell}", flush=True)

    zeilen = []
    for i in range(len(daten)):
        probe = daten[i]
        bild = probe["bild"].numpy().transpose(1, 2, 0)
        d_gt = probe["tiefe"].numpy()
        h_gt = probe["hoehe"].numpy()
        maske = probe["maske"].numpy()
        k, H, gsd = float(probe["k"]), float(probe["flughoehe"]), float(probe["gsd"])
        if maske.sum() < 1000:
            continue

        D, _ = inferenz.roh(model, bild, device=device)
        d = k / D

        grund = {"ausschnitt": i, "gebiet": probe["site"], "flughoehe_m": H}
        if hoehenmodell is not None:
            # The output is directly in metres; no k, no ground reference.
            h_direkt, _ = inferenz.roh(hoehenmodell, bild, device=device)
            zeilen.append({**grund, "weg": "direktes Hoehenmodell", "kachel_m": np.nan,
                           "boden_faktor": np.nan, **fehler(h_direkt, h_gt, maske)})
        zeilen.append({**grund, "weg": "tiefe", "kachel_m": np.nan, "boden_faktor": np.nan,
                       **fehler(d, d_gt, maske)})
        zeilen.append({**grund, "weg": "aus der Flughoehe", "kachel_m": np.nan, "boden_faktor": np.nan,
                       **fehler(H - d, h_gt, maske)})
        for kachel in args.kachel_m:
            for faktor in args.boden_faktoren:
                boden = bodenmodell(d, gsd, kachel, args.perzentil, faktor)
                z = boden - d
                # Shape from the terrain model, absolute position from the flight
                # altitude: the terrain model gets the slope right but knows no
                # scale; the altitude supplies exactly that and nothing else.
                boden_k = boden - np.median(boden) + H
                zk = boden_k - d
                werte_k = fehler(zk, h_gt, maske)
                werte_k["unter_null"] = float(np.mean(zk[maske] < -1.0))
                zeilen.append({**grund, "weg": "Gelaendeform + Flughoehe",
                               "kachel_m": kachel, "boden_faktor": faktor, **werte_k})
                werte = fehler(z, h_gt, maske)
                werte["unter_null"] = float(np.mean(z[maske] < -1.0))
                zeilen.append({**grund, "weg": "aus dem Gelaendemodell",
                               "kachel_m": kachel, "boden_faktor": faktor, **werte})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(daten)}", flush=True)

    tabelle = pd.DataFrame(zeilen)
    tabelle.to_csv(args.out / f"hoehe_pruefung_{args.split}.csv", index=False)

    print(f"\n=== Tiefe und Hoehe aus bekannter Flughoehe ({args.split}) ===")
    einfach = tabelle[tabelle["weg"] != "aus dem Gelaendemodell"]
    print(einfach.groupby("weg")[["mae_m", "bias_m", "rmse_m", "korrelation"]].mean()
          .to_string(float_format=lambda v: f"{v:8.2f}"))

    print("\n=== Unterscheidet das Modell zwischen Bestaenden? ===")
    print("Korrelation der Bestandshoehe (95. Perzentil je Ausschnitt) ueber alle Ausschnitte.")
    print("Hoch heisst: hohe Bestaende werden als hoch erkannt. Niedrig heisst: das Modell")
    print("raet ueberall dasselbe, egal wie hoch der Bestand wirklich ist.\n")
    for weg, teil in tabelle.groupby("weg"):
        eins = teil.drop_duplicates(subset=["ausschnitt"]) if weg != "aus dem Gelaendemodell" else \
            teil[(teil["kachel_m"] == teil["kachel_m"].iloc[0]) &
                 (teil["boden_faktor"] == teil["boden_faktor"].iloc[0])]
        a, b = eins["p95_vorher"].to_numpy(), eins["p95_wahr"].to_numpy()
        gueltig = np.isfinite(a) & np.isfinite(b)
        r = float(np.corrcoef(a[gueltig], b[gueltig])[0, 1]) if gueltig.sum() > 2 else np.nan
        print(f"  {weg:26s} r = {r:5.2f} | vorhergesagt {a[gueltig].mean():5.1f} +- "
              f"{a[gueltig].std():4.1f} m | wahr {b[gueltig].mean():5.1f} +- {b[gueltig].std():4.1f} m")

    print("\n=== Hoehe aus dem Gelaendemodell, je Einstellung ===")
    gm = tabelle[tabelle["weg"] == "aus dem Gelaendemodell"]
    uebersicht = gm.groupby(["kachel_m", "boden_faktor"])[
        ["mae_m", "bias_m", "rmse_m", "korrelation", "unter_null"]].mean()
    print(uebersicht.to_string(float_format=lambda v: f"{v:8.2f}"))

    beste = uebersicht["mae_m"].idxmin()
    print(f"\nGeringster Fehler bei Kachel {beste[0]:.0f} m und Bodenfaktor {beste[1]:.3f}: "
          f"MAE {uebersicht.loc[beste, 'mae_m']:.2f} m, Bias {uebersicht.loc[beste, 'bias_m']:+.2f} m")
    print("Zum Vergleich die Hoehe aus bekannter Flughoehe: MAE "
          f"{einfach[einfach['weg'] == 'aus der Flughoehe']['mae_m'].mean():.2f} m")


if __name__ == "__main__":
    main()
