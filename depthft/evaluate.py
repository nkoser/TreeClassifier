"""Pure vs. fine-tuned Depth Pro -- on FORTRESS sites that have ground truth.

The test sites were never seen during training (the split is in `index.json` and
is set by `prepare.py`). The same virtual nadir frames are cut from them as in
training, only without augmentation and with a fixed draw -- every variant sees
exactly the same crops.

Four variants are compared, and the third is the most revealing:

  pure + FOV head        Depth Pro as you take it off the shelf.
  pure + known camera    The same run, but with `k` supplied instead of
                         estimated. Separates FOV errors from depth errors.
  pure + scale match     Additionally scaled globally so that the median is
                         exactly right. **Not an applicable method but an
                         oracle**: the factor comes from the truth. This row is
                         the ceiling of what a pure scale correction could ever
                         reach -- and thereby measures how good the *relative*
                         structure is.
  pure + height anchor   Scaled until the deepest point in the image (95th
                         percentile) matches the known flight altitude. That is
                         applicable, because a drone knows its altitude from
                         barometer and GPS. It does presume that ground is
                         visible at all -- in a closed canopy the deepest visible
                         point is not the ground, and the anchor slips.
  fine-tuned             The result of this project: the scale comes from the
                         image itself, without an anchor and without an oracle.
  fine-tuned + anchor    Both together.

With `--videolook` the same crops run through blur, noise and JPEG artefacts.
That is not a cosmetic detail but the actual question: our frames are single
video images, while the orthomosaic is a razor-sharp product computed from many
captures. In a nadir forest capture the level of detail is the only cue to the
flight altitude -- if the model sees more detail, it infers finer ground sampling
and hence a lower altitude. Both numbers side by side show how strongly that
comes through.

    python depthft/evaluate.py --ft /scratch/shared/nik/runs/depthft/bestes
    python depthft/evaluate.py --videolook          # at the sharpness of our frames
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inferenz  # noqa: E402
from bilder import beschriften, hoehenbild  # noqa: E402
from dataset import NadirFrames  # noqa: E402

HOEHENKLASSEN = [(0, 40), (40, 70), (70, 200)]


def messen(d_pred: np.ndarray, d_gt: np.ndarray, maske: np.ndarray) -> dict[str, float]:
    p, g = d_pred[maske], d_gt[maske]
    if p.size == 0:
        return {}
    verhaeltnis = np.maximum(p / g, g / p)
    return {
        "absrel": float(np.mean(np.abs(p - g) / g)),
        "mae_m": float(np.mean(np.abs(p - g))),
        "rmse_m": float(np.sqrt(np.mean((p - g) ** 2))),
        "bias_m": float(np.mean(p - g)),
        "delta125": float(np.mean(verhaeltnis < 1.25)),
        "skalenfehler": float(np.median(p) / np.median(g)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("/scratch/shared/nik/data/fortress/depthft"))
    parser.add_argument("--ft", type=Path, default=Path("/scratch/shared/nik/runs/depthft/bestes"),
                        help="The fine-tuned checkpoint.")
    parser.add_argument("--pur", default="apple/DepthPro-hf")
    parser.add_argument("--out", type=Path, default=Path("/home/nik/workspace/TreeClassifier/results_depthft"))
    parser.add_argument("--split", default="test", choices=("test", "val", "train"))
    parser.add_argument("--pro-gebiet", type=int, default=40)
    parser.add_argument("--crop-px", type=int, default=1536)
    parser.add_argument("--seitenverhaeltnis", type=float, default=16 / 9)
    parser.add_argument("--hoehe-min", type=float, default=25.0)
    parser.add_argument("--hoehe-max", type=float, default=120.0)
    parser.add_argument("--abstand-min", type=float, default=20.0)
    parser.add_argument("--fov-min", type=float, default=60.0,
                        help="Narrower than in training for evaluation: around our camera.")
    parser.add_argument("--fov-max", type=float, default=85.0)
    parser.add_argument("--strahl-tiefe", action="store_true",
                        help="Has to match the training -- recorded in bestes/depthft.json.")
    parser.add_argument("--videolook", action="store_true",
                        help="Bring the crops to the sharpness of our video frames.")
    parser.add_argument("--beispiele", type=int, default=6)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "beispiele").mkdir(exist_ok=True)

    daten = NadirFrames(args.data, args.split, crop_px=args.crop_px,
                        seitenverhaeltnis=args.seitenverhaeltnis,
                        hoehe_min=args.hoehe_min, hoehe_max=args.hoehe_max,
                        abstand_min=args.abstand_min,
                        fov_min=args.fov_min, fov_max=args.fov_max,
                        pro_gebiet=args.pro_gebiet, augment=False, spiegeln=False,
                        domaene=args.videolook, cache=2, seed=args.seed,
                        strahl_tiefe=args.strahl_tiefe)
    print(f"{len(daten)} Ausschnitte aus {len(daten.sites)} Gebieten: {', '.join(daten.sites)}", flush=True)

    # Generate the crops once and keep them -- that way every variant is
    # guaranteed the same images, even if the dataset changes.
    proben = [daten[i] for i in range(len(daten))]
    zeigen = set(np.linspace(0, len(proben) - 1, min(args.beispiele, len(proben))).astype(int).tolist())

    zeilen: list[dict] = []
    karten: dict[int, dict[str, np.ndarray]] = {i: {} for i in zeigen}

    for name, quelle, mit_fov in (("pur", args.pur, True), ("feinabgestimmt", str(args.ft), False)):
        if name == "feinabgestimmt" and not args.ft.exists():
            print(f"\n{args.ft} fehlt -- nur das pure Modell wird ausgewertet.", flush=True)
            continue
        print(f"\n--- {name}: {quelle} ---", flush=True)
        model = inferenz.lade(quelle, device, fov_head=mit_fov)

        for i, probe in enumerate(proben):
            bild = probe["bild"].numpy().transpose(1, 2, 0)
            d_gt = probe["tiefe"].numpy()
            maske = probe["maske"].numpy()
            k_wahr, H = float(probe["k"]), float(probe["flughoehe"])

            D, fov = inferenz.roh(model, bild, device=device)
            varianten = {f"{name}_kamera": k_wahr / D}
            if fov is not None:
                varianten[f"{name}_fovkopf"] = inferenz.k_von_fov(fov) / D
            d = varianten[f"{name}_kamera"]
            # Anchor from the known flight altitude -- no truth needed, so applicable.
            boden = float(np.percentile(d[maske], 95))
            varianten[f"{name}_hoehenanker"] = d * (H / max(boden, 1e-6))
            if name == "pur":
                # Oracle: it knows the true median. To be read as a ceiling only.
                faktor = float(np.median(d_gt[maske]) / max(np.median(d[maske]), 1e-6))
                varianten["pur_skalenangleich"] = d * faktor

            for variante, d_pred in varianten.items():
                werte = messen(d_pred, d_gt, maske)
                if not werte:
                    continue
                werte.update(variante=variante, ausschnitt=i, gebiet=probe["site"],
                             flughoehe_m=H, fov_grad=inferenz.fov_von_k(k_wahr),
                             gsd_cm=float(probe["gsd"]) * 100,
                             fov_geschaetzt=fov if fov is not None else np.nan,
                             hoehe_p95_gt=float(np.percentile(probe["hoehe"].numpy()[maske], 95)))
                zeilen.append(werte)

            if i in zeigen:
                karten[i]["rgb"] = bild
                karten[i]["gt"] = probe["hoehe"].numpy()
                karten[i][f"{name}_h"] = H - varianten[f"{name}_kamera"]
                karten[i]["info"] = f"{probe['site']} | {H:.0f} m | GSD {float(probe['gsd'])*100:.1f} cm"
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(proben)}", flush=True)

        del model
        torch.cuda.empty_cache()

    tabelle = pd.DataFrame(zeilen)
    kennung = f"{args.split}{'_videolook' if args.videolook else ''}"
    tabelle.to_csv(args.out / f"metriken_{kennung}.csv", index=False)

    spalten = ["absrel", "mae_m", "rmse_m", "bias_m", "delta125", "skalenfehler"]
    # Crops of the same site are strongly correlated. So the site, not the crop,
    # is the statistical unit: average within a site first, then weight all sites
    # equally.
    je_gebiet = tabelle.groupby(["variante", "gebiet"])[spalten].mean()
    zusammen = je_gebiet.groupby("variante").mean().sort_values("absrel")
    streuung = je_gebiet.groupby("variante").std()
    je_gebiet.to_csv(args.out / f"metriken_{kennung}_je_gebiet.csv")
    print(f"\n=== {kennung}, {len(proben)} Ausschnitte ===")
    print(zusammen.to_string(float_format=lambda v: f"{v:8.3f}"))

    tabelle["hoehenklasse"] = pd.cut(tabelle["flughoehe_m"], [g[0] for g in HOEHENKLASSEN] + [HOEHENKLASSEN[-1][1]],
                                     labels=[f"{a}-{b} m" for a, b in HOEHENKLASSEN])
    nach_hoehe = tabelle.groupby(["variante", "hoehenklasse"], observed=True)[["absrel", "mae_m"]].mean()
    print("\n--- nach Flughoehe ---")
    print(nach_hoehe.to_string(float_format=lambda v: f"{v:8.3f}"))
    nach_hoehe.to_csv(args.out / f"metriken_{kennung}_nach_hoehe.csv")

    if "fov_geschaetzt" in tabelle and tabelle["fov_geschaetzt"].notna().any():
        teil = tabelle[tabelle["variante"] == "pur_kamera"]
        print(f"\nBildwinkel: wahr {teil['fov_grad'].mean():.1f} Grad, "
              f"vom Kopf geschaetzt {teil['fov_geschaetzt'].mean():.1f} Grad")

    for i, eintrag in karten.items():
        if "gt" not in eintrag:
            continue
        obergrenze = float(np.percentile(eintrag["gt"], 99)) or 1.0
        kacheln = [beschriften(cv2.cvtColor(eintrag["rgb"], cv2.COLOR_RGB2BGR), eintrag["info"]),
                   beschriften(hoehenbild(eintrag["gt"], obergrenze), "Wahrheit (nDSM)")]
        for name, titel in (("pur_h", "pur"), ("feinabgestimmt_h", "feinabgestimmt")):
            if name in eintrag:
                kacheln.append(beschriften(hoehenbild(eintrag[name], obergrenze),
                                           f"{titel} (0-{obergrenze:.0f} m)"))
        cv2.imwrite(str(args.out / "beispiele" / f"{kennung}_{i:03d}.jpg"), np.hstack(kacheln),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

    (args.out / f"zusammenfassung_{kennung}.json").write_text(
        json.dumps({"ausschnitte": len(proben), "gebiete": daten.sites,
                    "statistische_einheit": "gebiet",
                    "varianten": zusammen.to_dict(orient="index"),
                    "streuung_zwischen_gebieten": streuung.to_dict(orient="index")}, indent=2))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
