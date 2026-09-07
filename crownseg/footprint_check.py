"""How large does the crop around a crown have to be?

DINOvTree was trained on crops of 9.73 m edge length, and that number had been
adopted unquestioned. It only suits the crown size it was meant for, though:

    BAMFORESTS Hain   6.66 m crown in 9.73 m  ->  47 % of the area
    Quebec            4.09 m crown in 9.73 m  ->  18 %
    our own pines     2.40 m crown in 9.73 m  ->   6 %

At six percent the crop describes the stand and not the tree in question -- which
explains why a whole folder gets classified uniformly as one species and why the
feature clusters group by stand texture.

What is measured here is how species accuracy and cluster purity depend on the
crown-to-crop ratio. On Quebec, where both are checkable: the crop size is
derived per crown from its own diameter rather than fixed in advance.

    python crownseg/footprint_check.py --faktoren 1.5 2.4 3.5 5.0
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classify_check import KUERZEL  # noqa: E402
from cluster_crowns import extract_features  # noqa: E402
from infer_species import (  # noqa: E402
    QUEBEC_TREES_EXCLUDE, REPO_ROOT, build_dinovtree, load_class_names,
    resolve_device, short_name, to_model_input,
)
from quebec import wkb_rings  # noqa: E402


def kurz(name: str) -> str:
    teile = name.split()
    return f"{teile[0][0]}. {teile[-1]}" if len(teile) > 1 else name


def passt(wahr: str, vorhergesagt: str) -> bool:
    tw, tv = wahr.split(), vorhergesagt.split()
    if tw[0][0].upper() != tv[0][0].upper():
        return False
    if len(tw) == 1 or tw[-1] in ("A.Dietr.", "L."):
        return True
    return tw[-1] == tv[-1]


def main() -> None:
    import rasterio
    from rasterio.windows import Window
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path,
                        default=Path("/scratch/shared/nik/data/quebec_trees/quebec_trees_dataset_2021-09-02"))
    parser.add_argument("--date", default="2021-09-02")
    parser.add_argument("--zone", default="zone3")
    parser.add_argument("--ckpt", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/checkpoints/dinovtreeb_quebectrees.pth"))
    parser.add_argument("--categories", type=Path,
                        default=REPO_ROOT / "third_party" / "quebec_trees_categories.json")
    parser.add_argument("--faktoren", type=float, nargs="*", default=[1.5, 2.4, 3.5, 5.0, 8.0],
                        help="Crop as a multiple of the crown diameter.")
    parser.add_argument("--clusters", type=int, default=14)
    parser.add_argument("--limit", type=int, default=900)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_arten_crownseg" / "ausschnitt.csv")
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

    # Collect all crowns once, with centre and diameter.
    kronen = []
    with rasterio.open(cog) as src:
        inverse = ~src.transform
        gsd = abs(src.transform.a)
        rng = np.random.default_rng(0)
        for i in rng.permutation(len(eintraege)):
            if len(kronen) >= args.limit:
                break
            geometry, label = eintraege[i]
            ziel = KUERZEL.get(label)
            rings = wkb_rings(geometry)
            if ziel is None or not rings:
                continue
            cols, rows = inverse * (rings[0][:, 0], rings[0][:, 1])
            durchmesser = max(cols.max() - cols.min(), rows.max() - rows.min())
            kronen.append((float(cols.mean()), float(rows.mean()), float(durchmesser), ziel))

        print(f"{len(kronen)} Kronen | GSD {gsd*100:.2f} cm | "
              f"Median-Durchmesser {np.median([k[2] for k in kronen])*gsd:.2f} m\n", flush=True)

        print(f"{'Faktor':>7} {'Ausschnitt':>11} {'Kronenanteil':>13} {'Genauigkeit':>12} {'ARI':>7} {'Reinheit':>9}")
        rows = []
        for faktor in args.faktoren:
            crops, arten = [], []
            for cx, cy, durchmesser, ziel in kronen:
                size_px = max(32, int(round(durchmesser * faktor)))
                x0, y0 = int(cx - size_px // 2), int(cy - size_px // 2)
                if x0 < 0 or y0 < 0 or x0 + size_px >= src.width or y0 + size_px >= src.height:
                    continue
                patch = np.ascontiguousarray(np.transpose(
                    src.read((1, 2, 3), window=Window(x0, y0, size_px, size_px)), (1, 2, 0)))
                if (patch.max(axis=2) == 0).mean() > 0.05:
                    continue
                crops.append(to_model_input(patch))
                arten.append(ziel)

            features, probabilities = extract_features(model, np.stack(crops), args.batch_size, device, "head")
            wahr = pd.Series([kurz(a) for a in arten])
            # short_name, not kurz(): the class names carry author citations
            # ("Abies balsamea (L.) Mill."), and kurz() takes the last word.
            vorhergesagt = [short_name(class_names[k]) for k in probabilities.argmax(axis=1)]
            genauigkeit = np.mean([passt(w, v) for w, v in zip(arten, vorhergesagt)])

            normalized = features / np.maximum(1e-9, np.linalg.norm(features, axis=1, keepdims=True))
            reduced = PCA(n_components=min(50, len(crops) - 1), random_state=0).fit_transform(normalized)
            labels = KMeans(n_clusters=args.clusters, n_init=10, random_state=0).fit_predict(reduced)
            reinheit = pd.DataFrame({"c": labels, "a": wahr}).groupby("c")["a"] \
                .agg(lambda s: s.value_counts().iloc[0]).sum() / len(wahr)
            ari = adjusted_rand_score(wahr, labels)

            mittel_m = np.median([k[2] for k in kronen]) * faktor * gsd
            anteil = 1.0 / faktor**2
            rows.append({"faktor": faktor, "ausschnitt_m": mittel_m, "kronenanteil": anteil,
                         "genauigkeit": genauigkeit, "ari": ari, "reinheit": reinheit})
            print(f"{faktor:7.1f} {mittel_m:10.1f} m {anteil:12.1%} {genauigkeit:12.1%} "
                  f"{ari:7.3f} {reinheit:9.1%}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nZum Vergleich: fester Ausschnitt 9.73 m entspricht bei Quebec Faktor "
          f"{9.73/(np.median([k[2] for k in kronen])*gsd):.1f}")


if __name__ == "__main__":
    main()
