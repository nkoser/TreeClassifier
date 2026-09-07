"""Artklassifikation auf segmentierten Kronen statt auf Detektor-Boxen.

Bindet die Segmentierung (segment_sam3.py / segment_hybrid.py / merge_crowns.py)
an DINOvTree an. Der entscheidende Gewinn gegenueber infer_species.py: die
Crop-Groesse kommt jetzt aus dem **gemessenen Kronendurchmesser** statt aus einer
Flughoehen-Schaetzung.

Im Training lag der Baum in einem 9.73-m-Fenster und fuellte davon grob ein
Drittel bis die Haelfte. Genau dieses Verhaeltnis stellt --crop-factor her: der
Ausschnitt ist ein Vielfaches des Kronendurchmessers, unabhaengig von Flughoehe,
Kamera und Massstab. Damit entfaellt die groesste Unsicherheit der bisherigen
Pipeline.

Beispiel:
    python classify_crowns.py --segments results_sam3/fix2 --out results_arten
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from infer_species import (
    IMAGE_SUFFIXES,
    PALETTE,
    QUEBEC_TREES_EXCLUDE,
    REPO_ROOT,
    build_dinovtree,
    classify_crops,
    crop_centered,
    load_class_names,
    resolve_device,
    short_name,
    to_model_input,
)


def crown_records(labels: np.ndarray) -> pd.DataFrame:
    """Schwerpunkt und Durchmesser je Instanz, direkt aus der Labelkarte."""
    from skimage.measure import regionprops

    records = []
    for region in regionprops(labels):
        cy, cx = region.centroid
        records.append(
            {
                "id": int(region.label),
                "cx": float(cx),
                "cy": float(cy),
                "area_px": int(region.area),
                "durchmesser_px": float(region.equivalent_diameter_area),
            }
        )
    return pd.DataFrame(records)


def draw_species(image_bgr: np.ndarray, labels: np.ndarray, crowns: pd.DataFrame, class_names: list[str],
                 alpha: float, caption: str) -> np.ndarray:
    """Kronenpolygone in der Farbe ihrer vorhergesagten Art."""
    color_of = {name: PALETTE[i % len(PALETTE)][::-1] for i, name in enumerate(class_names)}

    tint = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for row in crowns.itertuples():
        tint[labels == row.id] = color_of[row.species_top1]

    covered = labels > 0
    canvas = image_bgr.copy()
    canvas[covered] = ((1 - alpha) * canvas[covered] + alpha * tint[covered]).astype(np.uint8)

    work = labels.astype(np.uint16)
    kernel = np.ones((3, 3), np.uint8)
    borders = (cv2.dilate(work, kernel) != cv2.erode(work, kernel)) & covered
    canvas[borders] = tint[borders]

    for row in crowns.itertuples():
        label = f"{short_name(row.species_top1)} {row.prob_top1:.2f}"
        cv2.putText(canvas, label, (int(row.cx) - 30, int(row.cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (int(row.cx) - 30, int(row.cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 900), 34), (0, 0, 0), -1)
    cv2.putText(canvas, caption, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--segments", type=Path, default=REPO_ROOT / "results_sam3" / "fix2",
                        help="Verzeichnis mit den *_labels.png der Segmentierung.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_arten")
    parser.add_argument("--ckpt", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/checkpoints/dinovtreeb_quebectrees.pth"))
    parser.add_argument("--categories", type=Path, default=REPO_ROOT / "third_party" / "quebec_trees_categories.json")

    parser.add_argument("--crop-factor", type=float, default=2.5,
                        help="Ausschnittsgroesse als Vielfaches des Kronendurchmessers. Das Training "
                             "zeigte den Baum in etwa diesem Verhaeltnis zum Bildausschnitt.")
    parser.add_argument("--min-diameter-px", type=float, default=30.0,
                        help="Kleinere Kronen ueberspringen -- zu wenig Bildinformation.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    class_names = load_class_names(args.categories, QUEBEC_TREES_EXCLUDE)
    model = build_dinovtree(args.ckpt, n_classes=len(class_names), max_height=30.0, device=device)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | {len(class_names)} Klassen | Crop = {args.crop_factor}x Kronendurchmesser\n")

    all_results = []
    for label_path in sorted(args.segments.glob("*/*_labels.png")):
        folder = label_path.parent.name
        stem = label_path.name.replace("_labels.png", "")
        originals = [p for p in (args.input / folder).glob(f"{stem}.*") if p.suffix.lower() in IMAGE_SUFFIXES]
        if not originals:
            print(f"  kein Originalframe zu {folder}/{stem}")
            continue

        image_bgr = cv2.imread(str(originals[0]))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        labels = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED).astype(np.int32)

        crowns = crown_records(labels)
        skipped = int((crowns["durchmesser_px"] < args.min_diameter_px).sum())
        crowns = crowns[crowns["durchmesser_px"] >= args.min_diameter_px].reset_index(drop=True)
        if crowns.empty:
            print(f"  {folder}/{stem}: keine ausreichend grossen Kronen")
            continue

        crops = np.stack(
            [
                to_model_input(
                    crop_centered(image_rgb, row.cx, row.cy, max(16, int(round(args.crop_factor * row.durchmesser_px))))
                )
                for row in crowns.itertuples()
            ]
        )
        probabilities, heights = classify_crops(model, crops, args.batch_size, device)
        order = np.argsort(-probabilities, axis=1)
        rows = np.arange(len(order))

        crowns["crop_px"] = (args.crop_factor * crowns["durchmesser_px"]).round().astype(int)
        crowns["species_top1"] = [class_names[i] for i in order[:, 0]]
        crowns["prob_top1"] = probabilities[rows, order[:, 0]]
        crowns["species_top2"] = [class_names[i] for i in order[:, 1]]
        crowns["prob_top2"] = probabilities[rows, order[:, 1]]
        crowns["species_top3"] = [class_names[i] for i in order[:, 2]]
        crowns["prob_top3"] = probabilities[rows, order[:, 2]]
        crowns["height_m_pred"] = heights
        crowns["entropy"] = -(probabilities * np.log(probabilities + 1e-12)).sum(axis=1)

        out_folder = args.out / folder
        out_folder.mkdir(parents=True, exist_ok=True)
        crowns.insert(0, "frame", originals[0].name)
        crowns.insert(0, "folder", folder)
        crowns.to_csv(out_folder / f"{stem}_arten.csv", index=False)

        top = crowns["species_top1"].value_counts().head(3)
        caption = f"{len(crowns)} Kronen | " + ", ".join(f"{short_name(k)} {v}" for k, v in top.items())
        cv2.imwrite(
            str(out_folder / f"{stem}_arten.jpg"),
            draw_species(image_bgr, labels, crowns, class_names, args.alpha, caption),
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )

        all_results.append(crowns)
        print(
            f"  {folder}/{stem}: {len(crowns)} Kronen"
            f"{f' ({skipped} zu klein)' if skipped else ''} | "
            f"Crop med {crowns['crop_px'].median():.0f} px | {caption.split('| ')[1]}"
        )

    if not all_results:
        print("Keine Ergebnisse.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(args.out / "all_species.csv", index=False)

    print(f"\n{len(combined)} Kronen klassifiziert -> {args.out}/all_species.csv")
    print(f"mittlere Top-1-Wahrscheinlichkeit: {combined['prob_top1'].mean():.3f} "
          f"| mittlere Entropie: {combined['entropy'].mean():.3f}")
    pivot = (
        combined.pivot_table(index="folder", columns="species_top1", values="id", aggfunc="count")
        .fillna(0)
        .astype(int)
    )
    pivot.to_csv(args.out / "summary_by_folder.csv")
    print(pivot.to_string())


if __name__ == "__main__":
    main()
