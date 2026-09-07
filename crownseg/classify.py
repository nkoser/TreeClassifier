"""Tree species per crown instance -- stage 2 on the instances from `crownseg`.

`infer_species.py` used to get its instances from DeepForest. Measured on
BAMFORESTS its successor, the trained EoMT, is far ahead of it, and above all it
delivers masks instead of boxes. That changes two things:

  centre        The centroid of the mask hits the tree better than the centre of
                a box -- for a leaning or half-occluded crown the box centre
                often lands beside it.
  selection     Instances come with a confidence and a shape; fragments can be
                sorted out beforehand instead of handed to the classifier.

The crop still follows the training of the checkpoint: DINOvTree-B has only ever
seen crops of 9.73 m edge length (512 px at 1.9 cm/px, without resampling). The
image scale has to be known for that -- here it comes from the per-folder
measurement (`scale_probe.py`), not from an assumed flight altitude.

**How to read the results:** the checkpoint knows 14 classes from Quebec. For
Central European stands that is usable at genus level, not at species level --
European beech does not exist in this label set.

    python crownseg/classify.py --labels results_frames_eomt_v2 --scales pines=1.2
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infer_species import (  # noqa: E402
    QUEBEC_TREES_EXCLUDE, REPO_ROOT, build_dinovtree, classify_crops,
    crop_centered, load_class_names, resolve_device, short_name, to_model_input,
)

BAM_GSD_M = 0.0170


def instances_from_labels(labels: np.ndarray, min_area: int) -> list[dict]:
    out = []
    for value in np.unique(labels):
        if value == 0:
            continue
        mask = labels == value
        area = int(mask.sum())
        if area < min_area:
            continue
        ys, xs = np.nonzero(mask)
        out.append({
            "instanz": int(value),
            "cx": float(xs.mean()), "cy": float(ys.mean()),   # centroid, not box centre
            "x0": int(xs.min()), "y0": int(ys.min()),
            "x1": int(xs.max()) + 1, "y1": int(ys.max()) + 1,
            "flaeche_px": area,
        })
    return out


def draw(image_bgr: np.ndarray, labels: np.ndarray, frame: pd.DataFrame,
         class_names: list[str]) -> np.ndarray:
    canvas = image_bgr.copy()
    kernel = np.ones((3, 3), np.uint8)
    borders = (cv2.dilate(labels, kernel) != cv2.erode(labels, kernel)) & (labels > 0)
    canvas[borders] = (80, 230, 120)
    for row in frame.itertuples():
        name = short_name(class_names[row.klasse])
        cv2.putText(canvas, f"{name} {row.p:.2f}", (int(row.cx) - 30, int(row.cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, f"{name} {row.p:.2f}", (int(row.cx) - 30, int(row.cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, required=True, help="Folder with <folder>/<stem>_labels.png")
    parser.add_argument("--images", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_arten_crownseg")
    parser.add_argument("--ckpt", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/checkpoints/dinovtreeb_quebectrees.pth"))
    parser.add_argument("--categories", type=Path,
                        default=REPO_ROOT / "third_party" / "quebec_trees_categories.json")
    parser.add_argument("--scales", nargs="*", default=[],
                        help="FOLDER=FACTOR from scale_probe.py; GSD = factor x 1.70 cm.")
    parser.add_argument("--gsd-cm", type=float, default=None, help="Fixed GSD instead of --scales.")
    parser.add_argument("--footprint-factor", type=float, default=2.4,
                        help="Crop as a multiple of the crown diameter. The optimum measured "
                             "on Quebec (86.2 %% against 79.8 %% at factor 5).")
    parser.add_argument("--footprint-m", type=float, default=None,
                        help="Fixed crop instead of --footprint-factor; 9.73 m matches the "
                             "training but only suits crowns of about 4 m.")
    parser.add_argument("--min-area", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()
    scales = {p.split("=")[0]: float(p.split("=")[1]) for p in args.scales}

    device = resolve_device(args.device)
    class_names = load_class_names(args.categories, QUEBEC_TREES_EXCLUDE)
    model = build_dinovtree(args.ckpt, n_classes=len(class_names), max_height=30.0, device=device)
    model.eval()
    print(f"Device: {device} | {len(class_names)} Klassen | Ausschnitt {args.footprint_m} m", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(args.labels.rglob("*_labels.png")):
        relative = path.relative_to(args.labels)
        folder, stem = relative.parent.name, path.name[: -len("_labels.png")]
        matches = list((args.images / folder).glob(f"{stem}.*"))
        image = cv2.imread(str(matches[0])) if matches else None
        if image is None:
            continue
        labels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if labels.shape[:2] != image.shape[:2]:
            labels = cv2.resize(labels, image.shape[1::-1], interpolation=cv2.INTER_NEAREST)

        gsd_m = (args.gsd_cm / 100.0) if args.gsd_cm else scales.get(folder, 1.0) * BAM_GSD_M
        instances = instances_from_labels(labels, args.min_area)
        if not instances:
            continue

        # The crop per crown comes from its own diameter. A fixed crop shows
        # mostly neighbouring trees for small crowns: on Quebec the accuracy falls
        # from 86.2 % (crown fills 17 % of the area) to 69.2 % (1.6 %). Too tight
        # is worse as well -- a bit of surroundings contributes.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crops = []
        for i in instances:
            durchmesser = max(i["x1"] - i["x0"], i["y1"] - i["y0"])
            size_px = (max(16, int(round(args.footprint_m / gsd_m))) if args.footprint_m
                       else max(32, int(round(durchmesser * args.footprint_factor))))
            i["ausschnitt_px"] = size_px
            crops.append(to_model_input(crop_centered(rgb, i["cx"], i["cy"], size_px)))
        crops = np.stack(crops)
        probabilities, heights = classify_crops(model, crops, args.batch_size, device)

        frame = pd.DataFrame(instances)
        order = np.argsort(-probabilities, axis=1)
        frame["klasse"] = order[:, 0]
        frame["art"] = [class_names[k] for k in frame["klasse"]]
        frame["p"] = probabilities[np.arange(len(frame)), order[:, 0]]
        frame["art_2"] = [class_names[k] for k in order[:, 1]]
        frame["p_2"] = probabilities[np.arange(len(frame)), order[:, 1]]
        frame["hoehe_m"] = heights
        frame["entropie"] = -(probabilities * np.log(probabilities + 1e-9)).sum(axis=1)
        frame.insert(0, "frame", stem)
        frame.insert(0, "ordner", folder)
        frame["gsd_cm"] = gsd_m * 100

        out_folder = args.out / folder
        out_folder.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_folder / f"{stem}_arten.csv", index=False)
        cv2.imwrite(str(out_folder / f"{stem}_arten.jpg"),
                    draw(image, labels, frame, class_names), [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append(frame)
        top = frame["art"].value_counts().head(3)
        print(f"  {folder}/{stem}: {len(frame)} Kronen | GSD {gsd_m*100:.2f} cm | "
              f"Ausschnitt {frame['ausschnitt_px'].median():.0f} px "
              f"({frame['ausschnitt_px'].median()*gsd_m:.1f} m) | "
              + ", ".join(f"{short_name(k)} {v}" for k, v in top.items()), flush=True)

    if rows:
        table = pd.concat(rows, ignore_index=True)
        table.to_csv(args.out / "alle_kronen.csv", index=False)
        print(f"\n{len(table)} Kronen -> {args.out / 'alle_kronen.csv'}")
        print(table.groupby("ordner")["art"].value_counts().groupby("ordner").head(3).to_string())


if __name__ == "__main__":
    main()
