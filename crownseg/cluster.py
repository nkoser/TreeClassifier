"""Kronen nach Merkmalsvektoren gruppieren -- ohne Artlabels.

Der DINOvTree-Kopf kennt 14 kanadische Klassen. Auf mitteleuropaeischen
Bestaenden gibt es fuer die meisten Baeume keine richtige Antwort: ein
Kiefernbestand wird zu Gelb-Birke, weil *Pinus sylvestris* im Label-Satz fehlt.
Die Merkmale selbst sind davon unberuehrt -- sie beschreiben das Aussehen der
Krone, nicht ihren Namen.

Dass das traegt, ist gemessen. Auf Quebec Zone 3, geclustert ohne die Labels und
danach verglichen, erreicht das Verfahren denselben Adjusted Rand Index wie der
beschriftete Klassifikator (0.764 gegen 0.762) und 86 bis 89 % Reinheit. Der
praktische Gewinn: zwoelf bis zwanzig Gruppen benennen statt Tausende Baeume.

Unterschied zu `cluster_crowns.py` im Wurzelverzeichnis: der Ausschnitt wird je
Krone aus ihrem eigenen Durchmesser bestimmt (Faktor 2.4, auf Quebec gemessenes
Optimum) statt fest vorgegeben. Ein fester Ausschnitt zeigt bei kleinen Kronen
ueberwiegend Nachbarbaeume -- auf Quebec faellt die Reinheit von 86.1 % auf
71.2 %, wenn die Krone nur 1.6 % statt 17 % der Flaeche fuellt.

    python crownseg/cluster.py --labels results_sam3/multiskala --clusters 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classify import instances_from_labels  # noqa: E402
from cluster_crowns import extract_features  # noqa: E402
from infer_species import (  # noqa: E402
    QUEBEC_TREES_EXCLUDE, REPO_ROOT, build_dinovtree, crop_centered,
    load_class_names, resolve_device, short_name, to_model_input,
)

THUMB = 112
BAM_GSD_M = 0.0170


def contact_sheet(thumbs: list[np.ndarray], columns: int, title: str) -> np.ndarray:
    rows = max(1, (len(thumbs) + columns - 1) // columns)
    sheet = np.zeros((rows * THUMB + 30, columns * THUMB, 3), dtype=np.uint8)
    for index, thumb in enumerate(thumbs):
        r, c = divmod(index, columns)
        sheet[30 + r * THUMB : 30 + (r + 1) * THUMB, c * THUMB : (c + 1) * THUMB] = thumb
    cv2.putText(sheet, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return sheet


def center_per_group(features: np.ndarray, groups: pd.Series) -> np.ndarray:
    """Ordnerweise zentrieren -- einfache Korrektur der Aufnahmebedingung.

    Jeder Flug hat eigene Beleuchtung, Belichtung und Kompression, und dieser
    Versatz ist groesser als der Unterschied zwischen zwei Baumarten. Ohne die
    Korrektur clustern die Merkmale nach Ordner statt nach Art.
    """
    centered = features.copy()
    for value in groups.unique():
        mask = (groups == value).to_numpy()
        centered[mask] -= centered[mask].mean(axis=0, keepdims=True)
    return centered


def main() -> None:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_cluster_crownseg")
    parser.add_argument("--ckpt", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/checkpoints/dinovtreeb_quebectrees.pth"))
    parser.add_argument("--categories", type=Path,
                        default=REPO_ROOT / "third_party" / "quebec_trees_categories.json")
    parser.add_argument("--features", choices=("head", "backbone"), default="head")
    parser.add_argument("--clusters", type=int, default=12)
    parser.add_argument("--pca", type=int, default=50)
    parser.add_argument("--footprint-factor", type=float, default=2.4)
    parser.add_argument("--scales", nargs="*", default=[])
    parser.add_argument("--min-area", type=int, default=200)
    parser.add_argument("--min-diameter-px", type=float, default=25.0)
    parser.add_argument("--no-center", action="store_true", help="Ordnerweises Zentrieren abschalten.")
    parser.add_argument("--sheet-columns", type=int, default=12)
    parser.add_argument("--sheet-samples", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()
    scales = {p.split("=")[0]: float(p.split("=")[1]) for p in args.scales}

    device = resolve_device(args.device)
    class_names = load_class_names(args.categories, QUEBEC_TREES_EXCLUDE)
    model = build_dinovtree(args.ckpt, n_classes=len(class_names), max_height=30.0, device=device)
    model.eval()

    crops, thumbs, records = [], [], []
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
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gsd_m = scales.get(folder, 1.0) * BAM_GSD_M

        for instance in instances_from_labels(labels, args.min_area):
            durchmesser = max(instance["x1"] - instance["x0"], instance["y1"] - instance["y0"])
            if durchmesser < args.min_diameter_px:
                continue
            size_px = max(32, int(round(durchmesser * args.footprint_factor)))
            patch = crop_centered(rgb, instance["cx"], instance["cy"], size_px)
            crops.append(to_model_input(patch))
            thumbs.append(cv2.resize(patch[:, :, ::-1], (THUMB, THUMB), interpolation=cv2.INTER_AREA))
            records.append({"ordner": folder, "frame": stem, "instanz": instance["instanz"],
                            "cx": instance["cx"], "cy": instance["cy"],
                            "durchmesser_m": durchmesser * gsd_m, "ausschnitt_px": size_px})

    frame = pd.DataFrame(records)
    print(f"{len(frame)} Kronen aus {frame['frame'].nunique()} Frames | "
          f"Ausschnitt Median {frame['ausschnitt_px'].median():.0f} px | Merkmale: {args.features}",
          flush=True)

    features, probabilities = extract_features(model, np.stack(crops), args.batch_size, device, args.features)
    if not args.no_center:
        features = center_per_group(features, frame["ordner"])

    normalized = features / np.maximum(1e-9, np.linalg.norm(features, axis=1, keepdims=True))
    reduced = PCA(n_components=min(args.pca, len(frame) - 1), random_state=0).fit_transform(normalized)
    frame["cluster"] = KMeans(n_clusters=args.clusters, n_init=10, random_state=0).fit_predict(reduced)
    frame["dinovtree"] = [short_name(class_names[k]) for k in probabilities.argmax(axis=1)]

    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "clusters.csv", index=False)
    np.save(args.out / "features.npy", features)

    print("\nCluster x Ordner:")
    print(pd.crosstab(frame["cluster"], frame["ordner"]).to_string())

    sheets = args.out / "kontaktboegen"
    sheets.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for value, gruppe in frame.groupby("cluster"):
        auswahl = rng.choice(gruppe.index.to_numpy(),
                             size=min(args.sheet_samples, len(gruppe)), replace=False)
        top = gruppe["dinovtree"].value_counts().head(2)
        ordner = gruppe["ordner"].value_counts().head(2)
        titel = (f"cluster_{value:02d} | {len(gruppe)} Kronen | "
                 f"Median-Durchmesser {gruppe['durchmesser_m'].median():.1f} m | "
                 f"Ordner: {', '.join(f'{k} {v}' for k, v in ordner.items())} | "
                 f"DINOvTree: {', '.join(f'{k} {v}' for k, v in top.items())}")
        cv2.imwrite(str(sheets / f"cluster_{value:02d}.jpg"),
                    contact_sheet([thumbs[i] for i in auswahl], args.sheet_columns, titel),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"\n{args.clusters} Kontaktboegen -> {sheets}")


if __name__ == "__main__":
    main()
