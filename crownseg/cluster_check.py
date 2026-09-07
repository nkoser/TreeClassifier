"""Trennen die Merkmalscluster tatsaechlich Baumarten?

`cluster_crowns.py` gruppiert Kronen nach ihren Merkmalsvektoren, weil der
kanadische Klassenkopf fuer mitteleuropaeische Bestaende keine richtige Antwort
kennt. Die Idee ist plausibel -- die Merkmale beschreiben das Aussehen der Krone,
nicht ihren Namen -- aber sie wurde nie geprueft. Beurteilt wurde nach
Kontaktboegen, also nach Augenschein.

Quebec Zone 3 erlaubt die Pruefung: dort steht an jeder Krone die Art. Geclustert
wird ohne diese Labels, verglichen wird danach. Gemessen mit

  ARI   Adjusted Rand Index -- Uebereinstimmung der Gruppierung, zufallskorrigiert.
        0 heisst wie zufaellig, 1 heisst identisch.
  NMI   Normalized Mutual Information -- wie viel die Cluster ueber die Art verraten.
  Reinheit  Anteil der Kronen, die zur Mehrheitsart ihres Clusters gehoeren.

Zum Vergleich laeuft dieselbe Messung auf dem Klassenkopf, der auf dieser
Domaene 90.1 % erreicht. Das ist die Obergrenze: was die Merkmale hergeben, wenn
jemand sie beschriftet hat.

    python crownseg/cluster_check.py --clusters 14
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cluster_crowns import extract_features  # noqa: E402
from infer_species import (  # noqa: E402
    QUEBEC_TREES_EXCLUDE, REPO_ROOT, build_dinovtree, load_class_names,
    resolve_device, to_model_input,
)
from classify_check import KUERZEL  # noqa: E402
from quebec import wkb_rings  # noqa: E402


def main() -> None:
    import rasterio
    from rasterio.windows import Window
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path,
                        default=Path("/scratch/shared/nik/data/quebec_trees/quebec_trees_dataset_2021-09-02"))
    parser.add_argument("--date", default="2021-09-02")
    parser.add_argument("--zone", default="zone3")
    parser.add_argument("--ckpt", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/checkpoints/dinovtreeb_quebectrees.pth"))
    parser.add_argument("--categories", type=Path,
                        default=REPO_ROOT / "third_party" / "quebec_trees_categories.json")
    parser.add_argument("--features", choices=("head", "backbone"), default="head")
    parser.add_argument("--clusters", type=int, nargs="*", default=[8, 12, 14, 20])
    parser.add_argument("--pca", type=int, default=50)
    parser.add_argument("--footprint-m", type=float, default=9.73)
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_arten_crownseg" / "cluster_check.csv")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()

    device = resolve_device(args.device)
    class_names = load_class_names(args.categories, QUEBEC_TREES_EXCLUDE)
    model = build_dinovtree(args.ckpt, n_classes=len(class_names), max_height=30.0, device=device)
    model.eval()

    table = f"Z{args.zone[-1]}_polygons"
    con = sqlite3.connect(args.root / f"{table}.gpkg")
    eintraege = con.execute(f"SELECT Shape, Label FROM {table}").fetchall()
    cog = next((args.root / args.date / args.zone).glob("*-cog.tif"))

    crops, arten = [], []
    with rasterio.open(cog) as src:
        inverse = ~src.transform
        size_px = max(16, int(round(args.footprint_m / abs(src.transform.a))))
        rng = np.random.default_rng(0)
        for i in rng.permutation(len(eintraege))[: args.limit]:
            geometry, label = eintraege[i]
            ziel = KUERZEL.get(label)
            if ziel is None:
                continue
            rings = wkb_rings(geometry)
            if not rings:
                continue
            cols, rows = inverse * (rings[0][:, 0], rings[0][:, 1])
            x0, y0 = int(cols.mean() - size_px // 2), int(rows.mean() - size_px // 2)
            if x0 < 0 or y0 < 0 or x0 + size_px >= src.width or y0 + size_px >= src.height:
                continue
            patch = np.ascontiguousarray(np.transpose(
                src.read((1, 2, 3), window=Window(x0, y0, size_px, size_px)), (1, 2, 0)))
            if (patch.max(axis=2) == 0).mean() > 0.05:
                continue
            crops.append(to_model_input(patch))
            arten.append(ziel)

    crops = np.stack(crops)
    print(f"{len(crops)} Kronen, {len(set(arten))} Arten | Merkmale: {args.features}", flush=True)
    features, probabilities = extract_features(model, crops, args.batch_size, device, args.features)
    wahr = pd.Series(arten)

    normalized = features / np.maximum(1e-9, np.linalg.norm(features, axis=1, keepdims=True))
    reduced = PCA(n_components=min(args.pca, len(crops) - 1, features.shape[1]),
                  random_state=0).fit_transform(normalized)

    print(f"\n=== Clustern ohne Labels, verglichen mit der Wahrheit ({args.zone}) ===")
    print(f"{'k':>4} {'ARI':>7} {'NMI':>7} {'Reinheit':>9}")
    rows = []
    for k in args.clusters:
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(reduced)
        reinheit = pd.DataFrame({"c": labels, "a": wahr}).groupby("c")["a"] \
            .agg(lambda s: s.value_counts().iloc[0]).sum() / len(wahr)
        ari = adjusted_rand_score(wahr, labels)
        nmi = normalized_mutual_info_score(wahr, labels)
        rows.append({"k": k, "ari": ari, "nmi": nmi, "reinheit": reinheit})
        print(f"{k:4d} {ari:7.3f} {nmi:7.3f} {reinheit:9.1%}")

    # Bezugswerte: der beschriftete Kopf, und eine zufaellige Gruppierung.
    kopf = np.array([class_names[i].split()[0][0] + ". " + class_names[i].split()[-1]
                     if len(class_names[i].split()) > 1 else class_names[i]
                     for i in probabilities.argmax(axis=1)])
    zufall = np.random.default_rng(0).integers(0, 14, len(wahr))
    print(f"\nZum Vergleich:")
    print(f"  Klassenkopf (beschriftet)  ARI {adjusted_rand_score(wahr, kopf):.3f}  "
          f"NMI {normalized_mutual_info_score(wahr, kopf):.3f}")
    print(f"  zufaellige Gruppen         ARI {adjusted_rand_score(wahr, zufall):.3f}  "
          f"NMI {normalized_mutual_info_score(wahr, zufall):.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\n{args.out}")


if __name__ == "__main__":
    main()
