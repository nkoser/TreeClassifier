"""Check the species classifier on its own domain.

On our frames DINOvTree gives implausible answers -- a pine stand is classified
as yellow birch, so not even the right broad group. Two explanations are
possible, and they lead to completely different consequences:

  transfer       The checkpoint knows 14 Canadian classes and fails on Central
                 European stands. Then we need our own labels.
  wiring         Scale, crop, channel order or normalisation are wrong in our
                 call. Then it is a bug in the code.

Quebec zone 3 separates the two: the same domain the checkpoint was trained on,
with species labels on every crown. Classification here runs on the **ground
truth** rather than on predictions, so that the segmentation does not co-determine
the result -- what is measured is stage 2 alone.

Caveat: whether zone 3 was in the training split of DINOvTree is unknown. A good
value would then be flattered. A bad value, by contrast, is meaningful, because a
domain that was trained on should not fail.

    python crownseg/classify_check.py --zone zone3
"""

from __future__ import annotations

import argparse
import collections
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infer_species import (  # noqa: E402
    QUEBEC_TREES_EXCLUDE, REPO_ROOT, build_dinovtree, classify_crops,
    crop_centered, load_class_names, resolve_device, short_name, to_model_input,
)
from quebec import wkb_rings  # noqa: E402

# Quebec abbreviation -> scientific name as it appears in the label set.
KUERZEL = {
    "ABBA": "Abies balsamea", "ACPE": "Acer pensylvanicum", "ACRU": "Acer rubrum",
    "ACSA": "Acer saccharum", "BEAL": "Betula alleghaniensis", "BEPA": "Betula papyrifera",
    "FAGR": "Fagus grandifolia", "LALA": "Larix laricina", "PIST": "Pinus strobus",
    "THOC": "Thuja occidentalis", "TSCA": "Tsuga canadensis",
    "PIGL": "Picea", "PIMA": "Picea", "PIRU": "Picea", "Picea": "Picea",
    "POGR": "Populus", "POTR": "Populus", "Populus": "Populus", "Mort": "dead",
}


def main() -> None:
    import cv2
    import rasterio
    from rasterio.windows import Window

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path,
                        default=Path("/scratch/shared/nik/data/quebec_trees/quebec_trees_dataset_2021-09-02"))
    parser.add_argument("--date", default="2021-09-02")
    parser.add_argument("--zone", default="zone3")
    parser.add_argument("--ckpt", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/checkpoints/dinovtreeb_quebectrees.pth"))
    parser.add_argument("--categories", type=Path,
                        default=REPO_ROOT / "third_party" / "quebec_trees_categories.json")
    parser.add_argument("--footprint-m", type=float, default=9.73)
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_arten_crownseg" / "quebec_check.csv")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()

    device = resolve_device(args.device)
    class_names = load_class_names(args.categories, QUEBEC_TREES_EXCLUDE)
    model = build_dinovtree(args.ckpt, n_classes=len(class_names), max_height=30.0, device=device)
    model.eval()
    print(f"Device: {device} | {len(class_names)} Klassen")
    print("Klassen:", ", ".join(short_name(c) for c in class_names), "\n", flush=True)

    table = f"Z{args.zone[-1]}_polygons"
    con = sqlite3.connect(args.root / f"{table}.gpkg")
    eintraege = con.execute(f"SELECT Shape, Label FROM {table}").fetchall()
    cog = next((args.root / args.date / args.zone).glob("*-cog.tif"))

    wahr, vorhergesagt, sicherheit = [], [], []
    with rasterio.open(cog) as src:
        inverse = ~src.transform
        gsd = abs(src.transform.a)
        size_px = max(16, int(round(args.footprint_m / gsd)))
        print(f"{cog.name}: GSD {gsd*100:.2f} cm | Ausschnitt {size_px} px", flush=True)

        rng = np.random.default_rng(0)
        auswahl = rng.permutation(len(eintraege))[: args.limit]
        puffer, labels = [], []
        for i in auswahl:
            geometry, label = eintraege[i]
            ziel = KUERZEL.get(label)
            if ziel is None:
                continue
            rings = wkb_rings(geometry)
            if not rings:
                continue
            cols, rows = inverse * (rings[0][:, 0], rings[0][:, 1])
            cx, cy = float(cols.mean()), float(rows.mean())
            x0, y0 = int(cx - size_px // 2), int(cy - size_px // 2)
            if x0 < 0 or y0 < 0 or x0 + size_px >= src.width or y0 + size_px >= src.height:
                continue
            patch = src.read((1, 2, 3), window=Window(x0, y0, size_px, size_px))
            patch = np.ascontiguousarray(np.transpose(patch, (1, 2, 0)))
            if (patch.max(axis=2) == 0).mean() > 0.05:
                continue
            puffer.append(to_model_input(patch))
            labels.append(ziel)

    crops = np.stack(puffer)
    print(f"{len(crops)} Kronen mit Artlabel", flush=True)
    probabilities, _ = classify_crops(model, crops, args.batch_size, device)
    predicted = [short_name(class_names[k]) for k in probabilities.argmax(axis=1)]

    # Bring both sides to the same short form. The truth is spelled out
    # ("Abies balsamea"), the prediction abbreviated ("A. balsamea") -- compared
    # as strings they would never match.
    def kurz(name: str) -> str:
        teile = name.split()
        return f"{teile[0][0]}. {teile[-1]}" if len(teile) > 1 else name

    frame = pd.DataFrame({"wahr": [kurz(v) for v in labels], "vorhergesagt": predicted,
                          "p": probabilities.max(axis=1)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    treffer = (frame["wahr"] == frame["vorhergesagt"]).mean()
    print(f"\n=== Genauigkeit auf {args.zone}: {treffer:.1%} ===")
    print(f"Zufallsniveau bei {frame['wahr'].nunique()} vorkommenden Klassen: "
          f"{1/frame['wahr'].nunique():.1%}, haeufigste Klasse: "
          f"{frame['wahr'].value_counts(normalize=True).iloc[0]:.1%}\n")
    print("Wahre Klasse -> haeufigste Vorhersage (Anzahl, Trefferquote):")
    for wahr_name, gruppe in frame.groupby("wahr"):
        top = gruppe["vorhergesagt"].value_counts()
        richtig = (gruppe["vorhergesagt"] == wahr_name).mean()
        print(f"  {wahr_name:24s} n={len(gruppe):4d}  {richtig:5.1%} richtig  "
              f"| meist: {top.index[0]} ({top.iloc[0]})")


if __name__ == "__main__":
    main()
