"""Apply the FORTRESS head to our own crown instances.

`classify.py` queries the Canadian checkpoint; on a pine stand it has to pick
one of the 14 Quebec classes and answers yellow birch. Here the same frozen
backbone is attached to the head from `train_head.py`, which knows eight Central
European classes -- among them *Pinus sylvestris*.

The setup follows `classify.py`, so the results stay comparable: the centroid of
the mask as the centre, the crop as 2.4 times the crown diameter, the scale per
folder from `scale_probe.py`. `fortress.py` used the same conventions when
producing the training crops.

    python crownseg/apply_head.py --labels results_frames_sam3_multiscale --scales pines=1.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classify import BAM_GSD_M, instances_from_labels  # noqa: E402
from cluster_crowns import extract_features  # noqa: E402
from infer_species import (  # noqa: E402
    QUEBEC_TREES_EXCLUDE, REPO_ROOT, build_dinovtree, crop_centered,
    load_class_names, resolve_device, to_model_input,
)

# Conifers green, broadleaves orange, non-tree grey -- that way the image shows
# without a legend whether the assignment matches the stand at all.
FARBEN = {
    "Picea abies": (90, 200, 90), "Abies alba": (150, 210, 110),
    "Pinus sylvestris": (60, 230, 230), "Pseudotsuga menziesii": (110, 160, 60),
    "Fagus sylvatica": (60, 140, 240), "Acer pseudoplatanus": (80, 90, 230),
    "deadwood": (130, 130, 130), "forest floor": (90, 90, 90),
}


def kurz(name: str) -> str:
    teile = name.split()
    return f"{teile[0][0]}. {teile[1]}" if len(teile) > 1 and teile[0][0].isupper() else name


def zeichne(bild_bgr, labels, frame) -> np.ndarray:
    canvas = bild_bgr.copy()
    farbflaeche = canvas.copy()
    kernel = np.ones((3, 3), np.uint8)
    for row in frame.itertuples():
        farbflaeche[labels == row.instanz] = FARBEN.get(row.art, (200, 200, 200))
    canvas = cv2.addWeighted(canvas, 0.72, farbflaeche, 0.28, 0)
    canvas[(cv2.dilate(labels, kernel) != cv2.erode(labels, kernel)) & (labels > 0)] = (255, 255, 255)
    for row in frame.itertuples():
        text = f"{kurz(row.art)} {row.p:.2f}"
        for farbe, dicke in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(canvas, text, (int(row.cx) - 32, int(row.cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, farbe, dicke, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_arten_fortress")
    parser.add_argument("--backbone", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/checkpoints/dinovtreeb_quebectrees.pth"))
    parser.add_argument("--kopf", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/checkpoints/kopf_fortress.pth"))
    parser.add_argument("--categories", type=Path,
                        default=REPO_ROOT / "third_party" / "quebec_trees_categories.json")
    parser.add_argument("--scales", nargs="*", default=[], help="FOLDER=FACTOR from scale_probe.py.")
    parser.add_argument("--footprint-factor", type=float, default=2.4)
    parser.add_argument("--min-area", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()
    scales = {p.split("=")[0]: float(p.split("=")[1]) for p in args.scales}

    device = resolve_device(args.device)
    stand = torch.load(args.kopf, map_location=device, weights_only=False)
    klassen = stand["klassen"]
    mittel, streuung = stand["mittel"].to(device), stand["streuung"].to(device)
    kopf = nn.Sequential(nn.Linear(len(mittel), stand["args"]["hidden"]), nn.GELU(),
                         nn.Dropout(0.3), nn.Linear(stand["args"]["hidden"], len(klassen))).to(device)
    kopf.load_state_dict(stand["kopf"])
    kopf.eval()
    backbone = build_dinovtree(args.backbone,
                               n_classes=len(load_class_names(args.categories, QUEBEC_TREES_EXCLUDE)),
                               max_height=30.0, device=device)
    backbone.eval()
    print(f"{len(klassen)} Klassen | ausgewogene Testguete {stand['ausgewogen']:.1%} "
          f"auf {len(stand['test_gebiete'])} zurueckgehaltenen Gebieten\n", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    alle = []
    for pfad in sorted(args.labels.rglob("*_labels.png")):
        ordner, stem = pfad.relative_to(args.labels).parent.name, pfad.name[: -len("_labels.png")]
        treffer = list((args.images / ordner).glob(f"{stem}.*"))
        bild = cv2.imread(str(treffer[0])) if treffer else None
        if bild is None:
            continue
        labels = cv2.imread(str(pfad), cv2.IMREAD_UNCHANGED)
        if labels.shape[:2] != bild.shape[:2]:
            labels = cv2.resize(labels, bild.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        instanzen = instances_from_labels(labels, args.min_area)
        if not instanzen:
            continue

        gsd_m = scales.get(ordner, 1.0) * BAM_GSD_M
        rgb = cv2.cvtColor(bild, cv2.COLOR_BGR2RGB)
        crops = []
        for i in instanzen:
            durchmesser = max(i["x1"] - i["x0"], i["y1"] - i["y0"])
            groesse = max(32, int(round(durchmesser * args.footprint_factor)))
            i["ausschnitt_px"] = groesse
            crops.append(to_model_input(crop_centered(rgb, i["cx"], i["cy"], groesse)))

        merkmale, _ = extract_features(backbone, np.stack(crops), args.batch_size, device, "backbone")
        with torch.no_grad():
            X = (torch.from_numpy(merkmale.copy()).float().to(device) - mittel) / streuung
            p = kopf(X).softmax(1).cpu().numpy()

        frame = pd.DataFrame(instanzen)
        rang = np.argsort(-p, axis=1)
        frame["art"] = [klassen[k] for k in rang[:, 0]]
        frame["p"] = p[np.arange(len(frame)), rang[:, 0]]
        frame["art_2"] = [klassen[k] for k in rang[:, 1]]
        frame["p_2"] = p[np.arange(len(frame)), rang[:, 1]]
        frame.insert(0, "frame", stem)
        frame.insert(0, "ordner", ordner)

        ziel = args.out / ordner
        ziel.mkdir(parents=True, exist_ok=True)
        frame.to_csv(ziel / f"{stem}_arten.csv", index=False)
        cv2.imwrite(str(ziel / f"{stem}_arten.jpg"), zeichne(bild, labels, frame),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        alle.append(frame)
        top = frame["art"].value_counts().head(3)
        print(f"  {ordner}/{stem}: {len(frame)} Kronen | " +
              ", ".join(f"{kurz(k)} {v}" for k, v in top.items()), flush=True)

    tabelle = pd.concat(alle, ignore_index=True)
    tabelle.to_csv(args.out / "alle_kronen.csv", index=False)
    print(f"\n=== {len(tabelle)} Kronen -> {args.out} ===")
    for ordner, teil in tabelle.groupby("ordner"):
        anteil = teil["art"].value_counts(normalize=True)
        print(f"{ordner:12s} {len(teil):5d} Kronen | mittlere Sicherheit {teil['p'].mean():.2f} | " +
              ", ".join(f"{kurz(k)} {v:.0%}" for k, v in anteil.head(3).items()))


if __name__ == "__main__":
    main()
