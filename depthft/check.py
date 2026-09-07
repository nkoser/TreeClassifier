"""Before training: does the truth match the image at all?

The most expensive error in this chain would be an offset between orthomosaic
and height model -- the georeferencing is off, the crowns in the nDSM sit two
metres beside those in the image, and the training patiently learns nonsense.
You see that at once when you put both side by side.

Three things are checked:

  geometry     Do ground sampling, ground width and depth range agree with what
               must follow from flight altitude and field of view?
  coverage     Does the crown relief of the nDSM lie on the crowns in the image?
               The comparison strip shows image, height and relief side by side.
  baseline     What does pure Depth Pro deliver on these crops? Those numbers are
               the bar everything is later compared against.

    python depthft/check.py --n 8
    python depthft/check.py --n 8 --kein-modell     # data only, no GPU
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bilder import beschriften, farbskala, grauwert, hillshade, hoehenbild  # noqa: E402
from dataset import NadirFrames, fov_von_k  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("/scratch/shared/nik/data/fortress/depthft"))
    parser.add_argument("--out", type=Path,
                        default=Path("/home/nik/workspace/TreeClassifier/results_depthft/check"))
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--n", type=int, default=8, help="Total number of crops.")
    parser.add_argument("--crop-px", type=int, default=1536)
    parser.add_argument("--seitenverhaeltnis", type=float, default=16 / 9)
    parser.add_argument("--pur", default="apple/DepthPro-hf")
    parser.add_argument("--hfov-deg", type=float, default=73.7)
    parser.add_argument("--hoehe-min", type=float, default=25.0)
    parser.add_argument("--hoehe-max", type=float, default=120.0)
    parser.add_argument("--abstand-min", type=float, default=20.0)
    parser.add_argument("--kein-modell", action="store_true", help="Check the data only.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    daten = NadirFrames(args.data, args.split, crop_px=args.crop_px, pro_gebiet=1,
                        seitenverhaeltnis=args.seitenverhaeltnis,
                        hoehe_min=args.hoehe_min, hoehe_max=args.hoehe_max,
                        abstand_min=args.abstand_min,
                        augment=False, seed=args.seed, cache=1,
                        fov_min=args.hfov_deg, fov_max=args.hfov_deg)
    anzahl = min(args.n, len(daten))
    print(f"{len(daten.sites)} Gebiete im Split {args.split}, {anzahl} Ausschnitte geprueft\n", flush=True)

    model, device = None, None
    if not args.kein_modell:
        import torch
        import inferenz
        device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
        model = inferenz.lade(args.pur, device, fov_head=True)
        print(f"Pures Depth Pro auf {device}\n", flush=True)

    kopf = (f"{'Gebiet':9s} {'H':>6s} {'FOV':>6s} {'GSD':>7s} {'Breite':>8s} "
            f"{'gueltig':>8s} {'d_gt':>14s} {'Hoehe p95':>10s}")
    if model is not None:
        kopf += f" | {'d_pur':>13s} {'AbsRel':>7s} {'Skala':>6s} {'FOV_Kopf':>9s}"
    print(kopf)
    print("-" * len(kopf))

    absrel, skalen = [], []
    for i in range(anzahl):
        probe = daten[i]
        bild = probe["bild"].numpy().transpose(1, 2, 0)
        d_gt, hoehe, maske = probe["tiefe"].numpy(), probe["hoehe"].numpy(), probe["maske"].numpy()
        k, H, gsd = float(probe["k"]), float(probe["flughoehe"]), float(probe["gsd"])
        breite_m = args.crop_px * gsd

        zeile = (f"{probe['site']:9s} {H:5.0f}m {fov_von_k(k):5.1f}° {gsd*100:6.2f}cm "
                 f"{breite_m:7.1f}m {maske.mean()*100:7.0f}% "
                 f"{d_gt[maske].min():5.1f}-{d_gt[maske].max():5.1f}m "
                 f"{np.percentile(hoehe[maske], 95):9.1f}m")

        d_pur = None
        if model is not None:
            D, fov_kopf = inferenz.roh(model, bild, device=device)
            d_pur = k / D            # with the known camera, not the estimated angle
            p, g = d_pur[maske], d_gt[maske]
            fehler = float(np.mean(np.abs(p - g) / g))
            skala = float(np.median(p) / np.median(g))
            absrel.append(fehler)
            skalen.append(skala)
            zeile += (f" | {p.min():5.1f}-{p.max():5.1f}m {fehler:7.3f} {skala:6.2f} "
                      f"{fov_kopf:8.1f}°")
        print(zeile, flush=True)

        obergrenze = float(max(np.percentile(hoehe[maske], 99), 5.0))
        kacheln = [
            beschriften(cv2.cvtColor(bild, cv2.COLOR_RGB2BGR), f"{probe['site']}",
                        f"{H:.0f} m, {gsd*100:.1f} cm/px, {breite_m:.0f} m breit"),
            beschriften(hoehenbild(np.where(maske, hoehe, np.nan), obergrenze), "nDSM (Wahrheit)",
                        f"p95 {np.percentile(hoehe[maske], 95):.1f} m, "
                        f"{(1-maske.mean())*100:.0f} % ohne Wahrheit"),
            beschriften(grauwert(hillshade(hoehe)), "nDSM, Relief"),
        ]
        if d_pur is not None:
            kacheln.append(beschriften(hoehenbild(H - d_pur, obergrenze), "pur Depth Pro",
                                       f"Skala {skalen[-1]:.2f}x"))
        streifen = np.hstack(kacheln)
        cv2.imwrite(str(args.out / f"{i:02d}_{probe['site']}.jpg"),
                    np.hstack([streifen, farbskala(70, streifen.shape[0], obergrenze)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

    if absrel:
        print(f"\nPures Depth Pro: AbsRel {np.mean(absrel):.3f} | "
              f"Skalenfehler {np.mean(skalen):.2f}x (1.00 waere richtig)")
        print("Ein Skalenfehler deutlich neben 1.00 heisst: die Struktur mag stimmen, "
              "der Massstab nicht.")
    print(f"\nVergleichsstreifen -> {args.out}")
    print("Bitte ansehen: Kronen im Relief muessen auf den Kronen im Bild liegen. "
          "Ein Versatz waere ein Fehler in der Georeferenzierung, kein Modellproblem.")


if __name__ == "__main__":
    main()
