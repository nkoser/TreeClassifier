"""Cluster crowns by feature vector instead of by the Canadian classes.

The checkpoint can only output 14 classes from Quebec -- for Central European
stands there is no correct answer at all for most trees. The features from which
the head forms its decision are untouched by that, however: they describe the
appearance of the crown, not its Canadian name. Trees of the same species should
lie close together there, even when the model has no name for them.

What is tapped is the 1536-dimensional vector immediately before the last linear
layer -- exactly what the classifier bases its decision on (backbone [CLS]
concatenated with the cross-attention token, then LayerNorm). Optionally the
backbone [CLS] alone instead (--features backbone), which knows nothing about
the species task.

The output is a contact sheet per cluster: one grid of real crown crops per
group. That makes it possible to check in minutes whether the groups separate
anything biological -- and to name them with a few dozen clicks.

Example:
    python cluster_crowns.py --segments results_merged/s0.10_c16 --clusters 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from infer_species import (
    IMAGE_SUFFIXES,
    QUEBEC_TREES_EXCLUDE,
    REPO_ROOT,
    build_dinovtree,
    crop_centered,
    load_class_names,
    resolve_device,
    short_name,
    to_model_input,
)
from classify_crowns import crown_records

THUMB = 112


@torch.no_grad()
def extract_features(model, crops: np.ndarray, batch_size: int, device, source: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (features [N, D], class probabilities [N, C]).

    The hook taps the input of the last linear layer -- the representation the
    species decision actually rests on.
    """
    captured: list[torch.Tensor] = []

    def hook(_module, inputs, _output):
        captured.append(inputs[0].detach().flatten(1).cpu())

    handle = model.task_heads.classifier.register_forward_hook(hook)
    try:
        features, probabilities = [], []
        for start in range(0, len(crops), batch_size):
            batch = torch.from_numpy(crops[start : start + batch_size]).to(device)
            captured.clear()

            if source == "backbone":
                patch_tokens, cls_token = model.backbone(batch)
                features.append(cls_token.detach().cpu().numpy())
                logits, _ = model.task_heads((patch_tokens, cls_token))
            else:
                logits, _ = model(batch, {})
                features.append(captured[0].numpy())

            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    finally:
        handle.remove()

    return np.concatenate(features), np.concatenate(probabilities)


def center_per_group(features: np.ndarray, groups: pd.Series) -> np.ndarray:
    """Centre per group -- a simple batch correction.

    Without it the features cluster by capture condition: every flight has its own
    illumination, exposure and compression, and that offset is larger than the
    difference between two tree species. Subtracting the per-folder mean leaves
    only the variation *within* one flight -- and that is the part in which the
    species information can sit.
    """
    centered = features.copy()
    for value in groups.unique():
        mask = (groups == value).to_numpy()
        centered[mask] -= centered[mask].mean(axis=0, keepdims=True)
    return centered


def cluster(features: np.ndarray, args) -> tuple[np.ndarray, dict]:
    """L2 normalisation, PCA, then k-means or HDBSCAN."""
    from sklearn.cluster import HDBSCAN, KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    normalized = features / np.maximum(1e-9, np.linalg.norm(features, axis=1, keepdims=True))
    components = min(args.pca, normalized.shape[0] - 1, normalized.shape[1])
    reduced = PCA(n_components=components, random_state=0).fit_transform(normalized)

    info: dict = {"pca_components": components}
    if args.method == "hdbscan":
        labels = HDBSCAN(min_cluster_size=args.min_cluster_size).fit_predict(reduced)
        info["rauschen"] = int((labels == -1).sum())
        return labels, info

    if args.clusters:
        candidates = [args.clusters]
    else:
        candidates = list(range(args.k_min, args.k_max + 1))

    best_labels, best_score, scores = None, -np.inf, {}
    for k in candidates:
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(reduced)
        # Silhouette on a sample -- with thousands of points it is expensive otherwise.
        score = silhouette_score(reduced, labels, sample_size=min(3000, len(reduced)), random_state=0)
        scores[k] = round(float(score), 4)
        if score > best_score:
            best_labels, best_score = labels, score

    info["silhouette"] = scores
    info["gewaehlt"] = int(len(np.unique(best_labels)))
    return best_labels, info


def contact_sheet(thumbs: list[np.ndarray], columns: int, title: str) -> np.ndarray:
    """Grid of crown crops, with a caption line along the top."""
    rows = int(np.ceil(len(thumbs) / columns))
    sheet = np.zeros((rows * THUMB + 30, columns * THUMB, 3), dtype=np.uint8)
    for index, thumb in enumerate(thumbs):
        r, c = divmod(index, columns)
        sheet[30 + r * THUMB : 30 + (r + 1) * THUMB, c * THUMB : (c + 1) * THUMB] = thumb
    cv2.putText(sheet, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--segments", type=Path, default=REPO_ROOT / "results_merged" / "s0.10_c16")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_cluster")
    parser.add_argument("--ckpt", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/checkpoints/dinovtreeb_quebectrees.pth"))
    parser.add_argument("--categories", type=Path, default=REPO_ROOT / "third_party" / "quebec_trees_categories.json")

    parser.add_argument("--features", choices=("head", "backbone"), default="head",
                        help="head: vector before the class layer. backbone: plain DINOv3 [CLS].")
    parser.add_argument("--method", choices=("kmeans", "hdbscan"), default="kmeans")
    parser.add_argument("--clusters", type=int, default=None, help="Fixed cluster count; otherwise chosen by silhouette.")
    parser.add_argument("--k-min", type=int, default=4)
    parser.add_argument("--k-max", type=int, default=16)
    parser.add_argument("--min-cluster-size", type=int, default=25, help="HDBSCAN only.")
    parser.add_argument("--pca", type=int, default=50)
    parser.add_argument("--center-per-folder", action="store_true",
                        help="Centre the features per folder, to remove the batch effect of the capture.")
    parser.add_argument("--only-folder", default=None, help="Cluster one folder only (one flight, one illumination).")

    parser.add_argument("--crop-factor", type=float, default=2.5)
    parser.add_argument("--crop-px", type=int, default=None,
                        help="Fixed crop size instead of --crop-factor. Every crop then undergoes "
                             "the same scaling -- otherwise the features cluster by sharpness.")
    parser.add_argument("--min-diameter-px", type=float, default=30.0)
    parser.add_argument("--require-full-crop", action="store_true",
                        help="Skip crowns whose crop extends beyond the image border. Prevents the "
                             "mirrored border continuation from shaping the features.")
    parser.add_argument("--sheet-columns", type=int, default=12)
    parser.add_argument("--sheet-samples", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    class_names = load_class_names(args.categories, QUEBEC_TREES_EXCLUDE)
    model = build_dinovtree(args.ckpt, n_classes=len(class_names), max_height=30.0, device=device)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | Merkmale: {args.features} | Verfahren: {args.method}\n")

    records, crops, thumbs = [], [], []
    for label_path in sorted(args.segments.glob("*/*_labels.png")):
        folder = label_path.parent.name
        if args.only_folder and folder != args.only_folder:
            continue
        stem = label_path.name.replace("_labels.png", "")
        originals = [p for p in (args.input / folder).glob(f"{stem}.*") if p.suffix.lower() in IMAGE_SUFFIXES]
        if not originals:
            continue

        image_rgb = cv2.cvtColor(cv2.imread(str(originals[0])), cv2.COLOR_BGR2RGB)
        labels = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED).astype(np.int32)

        crowns = crown_records(labels)
        crowns = crowns[crowns["durchmesser_px"] >= args.min_diameter_px]
        height, width = image_rgb.shape[:2]
        for row in crowns.itertuples():
            size = args.crop_px or max(16, int(round(args.crop_factor * row.durchmesser_px)))
            if args.require_full_crop:
                half = size // 2
                if not (half <= row.cx <= width - half and half <= row.cy <= height - half):
                    continue
            patch = crop_centered(image_rgb, row.cx, row.cy, size)
            crops.append(to_model_input(patch))
            thumbs.append(cv2.resize(patch, (THUMB, THUMB), interpolation=cv2.INTER_AREA)[:, :, ::-1])
            records.append({"folder": folder, "frame": originals[0].name, "id": row.id,
                            "cx": row.cx, "cy": row.cy, "durchmesser_px": row.durchmesser_px})

        print(f"  {folder}/{stem}: {len(crowns)} Kronen")

    if not records:
        print("Keine Kronen gefunden.")
        return

    crops = np.stack(crops)
    frame = pd.DataFrame(records)
    print(f"\n{len(frame)} Kronen -> Merkmale berechnen ...")

    features, probabilities = extract_features(model, crops, args.batch_size, device, args.features)
    print(f"Merkmalsdimension: {features.shape[1]}")

    if args.center_per_folder and frame["folder"].nunique() > 1:
        features_for_clustering = center_per_group(features, frame["folder"])
        print("Merkmale je Ordner zentriert (Batch-Korrektur)")
    else:
        features_for_clustering = features

    labels, info = cluster(features_for_clustering, args)
    frame["cluster"] = labels
    frame["dinovtree_klasse"] = [class_names[i] for i in probabilities.argmax(axis=1)]
    frame["dinovtree_prob"] = probabilities.max(axis=1)
    frame.to_csv(args.out / "clusters.csv", index=False)
    np.save(args.out / "features.npy", features)

    print(f"\nCluster-Info: {info}")

    sheet_dir = args.out / "kontaktboegen"
    sheet_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for cluster_id in sorted(set(labels)):
        members = np.flatnonzero(labels == cluster_id)
        sample = rng.choice(members, size=min(args.sheet_samples, len(members)), replace=False)
        name = "rauschen" if cluster_id == -1 else f"cluster_{cluster_id:02d}"
        top = frame.loc[members, "dinovtree_klasse"].value_counts().head(2)
        title = (
            f"{name} | {len(members)} Kronen | DINOvTree sagt: "
            + ", ".join(f"{short_name(k)} {v}" for k, v in top.items())
        )
        cv2.imwrite(str(sheet_dir / f"{name}.jpg"),
                    contact_sheet([thumbs[i] for i in sample], args.sheet_columns, title),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"\nKontaktboegen -> {sheet_dir}")
    print("\nCluster x Ordner:")
    print(pd.crosstab(frame["cluster"], frame["folder"]).to_string())
    print("\nCluster x DINOvTree-Klasse (deckt sich das oder nicht?):")
    crosstab = pd.crosstab(frame["cluster"], frame["dinovtree_klasse"].map(short_name))
    print(crosstab.to_string())


if __name__ == "__main__":
    main()
