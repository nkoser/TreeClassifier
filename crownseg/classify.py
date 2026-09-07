"""Baumart je Kroneninstanz -- Stufe 2 auf den Instanzen aus `crownseg`.

`infer_species.py` bekam seine Instanzen bisher von DeepForest. Gemessen auf
BAMFORESTS liegt dessen Nachfolger, das trainierte EoMT, deutlich darueber, und
vor allem liefert es Masken statt Boxen. Das aendert zwei Dinge:

  Mittelpunkt   Der Schwerpunkt der Maske trifft den Baum besser als die Mitte
                einer Box -- bei einer schraeg gewachsenen oder halb verdeckten
                Krone liegt die Boxmitte oft daneben.
  Auswahl       Instanzen mit Konfidenz und Form; Bruchstuecke lassen sich vorher
                aussortieren, statt sie dem Klassifikator zu geben.

Der Ausschnitt folgt weiter dem Training des Checkpoints: DINOvTree-B hat
ausschliesslich Ausschnitte von 9.73 m Kantenlaenge gesehen (512 px bei
1.9 cm/px, ohne Resampling). Der Bildmassstab muss dafuer bekannt sein -- hier
kommt er aus der Messung je Ordner (`scale_probe.py`), nicht aus einer
angenommenen Flughoehe.

**Zur Einordnung der Ergebnisse:** der Checkpoint kennt 14 Klassen aus Quebec.
Fuer mitteleuropaeische Bestaende ist das auf Gattungsebene brauchbar, auf
Artebene nicht -- eine Rotbuche gibt es in diesem Label-Satz nicht.

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
            "cx": float(xs.mean()), "cy": float(ys.mean()),   # Schwerpunkt, nicht Boxmitte
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
    parser.add_argument("--labels", type=Path, required=True, help="Ordner mit <ordner>/<stem>_labels.png")
    parser.add_argument("--images", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_arten_crownseg")
    parser.add_argument("--ckpt", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/checkpoints/dinovtreeb_quebectrees.pth"))
    parser.add_argument("--categories", type=Path,
                        default=REPO_ROOT / "third_party" / "quebec_trees_categories.json")
    parser.add_argument("--scales", nargs="*", default=[],
                        help="ORDNER=FAKTOR aus scale_probe.py; GSD = Faktor x 1.70 cm.")
    parser.add_argument("--gsd-cm", type=float, default=None, help="Fester GSD statt --scales.")
    parser.add_argument("--footprint-factor", type=float, default=2.4,
                        help="Ausschnitt als Vielfaches des Kronendurchmessers. Auf Quebec "
                             "gemessenes Optimum (86.2 %% gegen 79.8 %% bei Faktor 5).")
    parser.add_argument("--footprint-m", type=float, default=None,
                        help="Fester Ausschnitt statt --footprint-factor; 9.73 m entspricht "
                             "dem Training, passt aber nur zu Kronen um 4 m.")
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

        # Ausschnitt je Krone aus ihrem eigenen Durchmesser. Ein fester
        # Ausschnitt zeigt bei kleinen Kronen ueberwiegend Nachbarbaeume: auf
        # Quebec faellt die Genauigkeit von 86.2 % (Krone fuellt 17 % der
        # Flaeche) auf 69.2 % (1.6 %). Zu eng ist ebenfalls schlechter --
        # etwas Umgebung traegt bei.
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
