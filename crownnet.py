"""Stufe 2: Kroneninstanzen direkt aus einem Einzelbild lernen.

Trainiert auf BAMFORESTS (58 228 annotierte Kronen aus deutschem Wald) statt auf
Heuristiken. Damit entfaellt zur Anwendung alles, was mehrere Frames oder eine
geschaetzte Tiefe brauchte -- ein Bild rein, Kronen raus.

Bauform wie bei dichter Zellsegmentierung, weil das Problem dasselbe ist: viele
sich beruehrende, rundliche Objekte. Der Kopf sagt drei Karten vorher:

  inneres   Krone ohne Randsaum. Getrennte Zusammenhangskomponenten hier sind
            bereits die Instanzkeime -- die Trennung lernt also das Netz.
  rand      Kronenrand. Wird vom Inneren abgezogen und haelt Nachbarn auseinander.
  zentrum   Gauss um den Kronenschwerpunkt, stabilisiert das Training und liefert
            bei Bedarf zusaetzliche Keime.

Zur Anwendung: Keime = Zusammenhangskomponenten des Inneren, danach Watershed bis
zur Kronenmaske. Das Watershed fuellt hier nur noch auf, es entscheidet nichts --
anders als in segment_trees.py, wo es die Trennung selbst treffen musste.

Der Backbone (DINOv3 aus dem DINOvTree-Checkpoint) bleibt eingefroren, trainiert
werden nur die rund 2 M Parameter des Kopfes.

Beispiel:
    python crownnet.py --mode prepare
    python crownnet.py --mode train
    python crownnet.py --mode predict --frames-dir /cold/Mahfuz/chosen_frames
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from infer_species import (
    IMAGE_SUFFIXES,
    QUEBEC_TREES_EXCLUDE,
    REPO_ROOT,
    build_dinovtree,
    load_class_names,
    resolve_device,
)

PATCH = 16


# --------------------------------------------------------------------------- #
# Aufbereitung
# --------------------------------------------------------------------------- #


def rasterize_split(coco_path: Path, image_root: Path, out_dir: Path, scale: float) -> int:
    """COCO-Polygone in Instanz-Labelkarten umwandeln, Bilder auf Zielmassstab bringen."""
    data = json.loads(coco_path.read_text())
    by_image: dict[int, list] = {}
    for annotation in data["annotations"]:
        by_image.setdefault(annotation["image_id"], []).append(annotation)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for image_info in data["images"]:
        annotations = by_image.get(image_info["id"], [])
        if not annotations:
            continue

        matches = list(image_root.rglob(image_info["file_name"]))
        if not matches:
            continue

        image = cv2.imread(str(matches[0]), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        image = image[:, :, :3]

        labels = np.zeros(image.shape[:2], dtype=np.int32)
        for index, annotation in enumerate(annotations, start=1):
            for polygon in annotation["segmentation"]:
                points = np.array(polygon, dtype=np.float64).reshape(-1, 2).round().astype(np.int32)
                cv2.fillPoly(labels, [points], index)

        size = (int(round(image.shape[1] * scale)), int(round(image.shape[0] * scale)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        labels = cv2.resize(labels.astype(np.int32), size, interpolation=cv2.INTER_NEAREST)

        stem = Path(image_info["file_name"]).stem
        cv2.imwrite(str(out_dir / f"{stem}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out_dir / f"{stem}_labels.png"), labels.astype(np.uint16))
        written += 1

    return written


def targets_from_labels(labels: np.ndarray, boundary_px: int, sigma: float) -> np.ndarray:
    """Drei Zielkarten aus einer Instanz-Labelkarte."""
    kernel = np.ones((3, 3), np.uint8)
    work = labels.astype(np.uint16)

    crown = (labels > 0).astype(np.float32)
    # Randsaum: dort stossen zwei Instanzen aneinander oder die Krone endet.
    border = ((cv2.dilate(work, kernel, iterations=boundary_px)
               != cv2.erode(work, kernel, iterations=boundary_px)) & (labels > 0)).astype(np.float32)
    interior = np.clip(crown - border, 0, 1)

    centers = np.zeros_like(crown)
    for value in np.unique(labels):
        if value == 0:
            continue
        ys, xs = np.nonzero(labels == value)
        centers[int(ys.mean()), int(xs.mean())] = 1.0
    centers = cv2.GaussianBlur(centers, (0, 0), sigma)
    if centers.max() > 0:
        centers /= centers.max()

    return np.stack([interior, border, centers])


# --------------------------------------------------------------------------- #
# Modell
# --------------------------------------------------------------------------- #


class CrownHead(nn.Module):
    """Faltungsdecoder auf den Patch-Tokens plus hochaufgeloester Bildzweig.

    Die Tokens der letzten ViT-Schicht liegen auf einem 16-px-Raster und sind
    semantisch stark, aber raeumlich grob -- daraus allein entsteht ein
    weichgezeichnetes Blobfeld ohne geschlossene Kronenraender. Der zusaetzliche
    flache Zweig auf dem Originalbild liefert die scharfen Kanten, die Tokens den
    Kontext. Beides wird bei voller Aufloesung zusammengefuehrt.
    """

    def __init__(self, in_dim: int = 768, width: int = 256, out_channels: int = 3, stem: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem, 3, padding=1), nn.GroupNorm(4, stem), nn.GELU(),
            nn.Conv2d(stem, stem, 3, padding=1), nn.GroupNorm(4, stem), nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(width // 8 + stem, 64, 3, padding=1), nn.GroupNorm(4, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(4, 64), nn.GELU(),
            nn.Conv2d(64, out_channels, 1),
        )
        self.blocks = nn.Sequential(
            nn.Conv2d(in_dim, width, 3, padding=1), nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(width, width // 2, 3, padding=1), nn.GroupNorm(8, width // 2), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(width // 2, width // 4, 3, padding=1), nn.GroupNorm(8, width // 4), nn.GELU(),
            # Dritte Stufe: 16 px Patch -> 2 px Ausgaberaster. Ohne sie liegt das
            # Ausgaberaster bei 4 px und ein schmaler Randsaum ist nicht darstellbar.
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(width // 4, width // 8, 3, padding=1), nn.GroupNorm(4, width // 8), nn.GELU(),
        )

    def forward(self, tokens: torch.Tensor, image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        context = F.interpolate(self.blocks(tokens), size=size, mode="bilinear", align_corners=False)
        detail = self.stem(F.interpolate(image, size=size, mode="bilinear", align_corners=False))
        return self.fuse(torch.cat([context, detail], dim=1))


class TileDataset(torch.utils.data.Dataset):
    def __init__(self, tiles: list[Path], crop: int, length: int, args, augment: bool) -> None:
        self.tiles, self.crop, self.length, self.args, self.augment = tiles, crop, length, args, augment

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        rng = np.random.default_rng(index)
        tile = self.tiles[rng.integers(len(self.tiles))]
        image = cv2.cvtColor(cv2.imread(str(tile)), cv2.COLOR_BGR2RGB)
        labels = cv2.imread(str(tile.with_name(tile.stem + "_labels.png")), cv2.IMREAD_UNCHANGED).astype(np.int32)

        height, width = labels.shape
        y = int(rng.integers(0, max(1, height - self.crop)))
        x = int(rng.integers(0, max(1, width - self.crop)))
        image = image[y : y + self.crop, x : x + self.crop]
        labels = labels[y : y + self.crop, x : x + self.crop]

        if self.augment:
            k = int(rng.integers(4))
            image, labels = np.rot90(image, k, (0, 1)), np.rot90(labels, k, (0, 1))
            if rng.random() < 0.5:
                image, labels = image[:, ::-1], labels[:, ::-1]

        targets = targets_from_labels(np.ascontiguousarray(labels), self.args.boundary_px, self.args.sigma)
        return (
            torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)).astype(np.float32) / 255.0),
            torch.from_numpy(targets),
        )


def build_backbone(args, device):
    class_names = load_class_names(args.categories, QUEBEC_TREES_EXCLUDE)
    model = build_dinovtree(args.ckpt, n_classes=len(class_names), max_height=30.0, device=device)
    backbone = model.backbone
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    backbone.eval()
    return backbone


def dice_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Dice gegen das starke Klassenungleichgewicht, BCE fuer stabile Gradienten."""
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2))
    dice = 1 - (2 * intersection + 1) / (probability.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + 1)
    return F.binary_cross_entropy_with_logits(logits, target) + dice.mean()


def compute_loss(prediction: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return (
        dice_bce(prediction[:, 0], targets[:, 0])
        + dice_bce(prediction[:, 1], targets[:, 1])
        + 5.0 * F.mse_loss(torch.sigmoid(prediction[:, 2]), targets[:, 2])
    )


# --------------------------------------------------------------------------- #
# Instanzbildung
# --------------------------------------------------------------------------- #


def instances_from_maps(maps: np.ndarray, args) -> np.ndarray:
    """Keime aus dem Inneren, danach Watershed bis zur Kronenmaske."""
    from skimage.measure import label as cc_label
    from skimage.segmentation import watershed

    interior, border, centers = maps
    seeds = cc_label(interior > args.interior_thresh)

    # Zu kleine Keime verwerfen -- meist Reste am Kronenrand.
    counts = np.bincount(seeds.ravel())
    for value in np.flatnonzero(counts < args.min_seed_px):
        if value:
            seeds[seeds == value] = 0
    seeds = cc_label(seeds > 0)

    crown = (interior + border) > args.crown_thresh
    if seeds.max() == 0:
        return np.zeros_like(seeds)
    return watershed(border, seeds, mask=crown)


@torch.no_grad()
def predict_maps(backbone, head, image_rgb: np.ndarray, device, long_side: int) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    scale = long_side / max(height, width)
    new_w = max(PATCH, int(round(width * scale / PATCH)) * PATCH)
    new_h = max(PATCH, int(round(height * scale / PATCH)) * PATCH)

    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    batch = torch.from_numpy(resized.transpose(2, 0, 1).astype(np.float32) / 255.0)[None].to(device)
    tokens, _ = backbone(batch)
    maps = torch.sigmoid(head(tokens, batch, (new_h, new_w)))[0].cpu().numpy()
    return np.stack([cv2.resize(m, (width, height), interpolation=cv2.INTER_LINEAR) for m in maps])


# --------------------------------------------------------------------------- #


def run_prepare(args) -> None:
    root = args.bamforests
    splits = {
        "train": ("instances_tree_train2023.json", "train2023"),
        "val": ("instances_tree_eval2023.json", "val2023"),
        "test1": ("instances_tree_TestSet12023.json", "test2023"),
        "test2": ("instances_tree_TestSet22023.json", "test2023"),
    }
    for name, (coco, folder) in splits.items():
        written = rasterize_split(root / "annotations" / coco, root / folder, args.prepared / name, args.scale)
        print(f"  {name:6s}: {written} Kacheln -> {args.prepared / name}")
    print(f"\nMassstab {args.scale:.2f} angewendet (BAMFORESTS 1.70 cm/px -> {1.70/args.scale:.1f} cm/px)")


def run_training(args, device) -> None:
    train_tiles = sorted((args.prepared / "train").glob("*.jpg"))
    val_tiles = sorted((args.prepared / "val").glob("*.jpg"))
    if not train_tiles:
        print(f"Keine aufbereiteten Kacheln in {args.prepared} -- erst --mode prepare laufen lassen.")
        return
    print(f"Train: {len(train_tiles)} Kacheln | Val: {len(val_tiles)}\n")

    backbone = build_backbone(args, device)
    head = CrownHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    loaders = {
        "train": torch.utils.data.DataLoader(
            TileDataset(train_tiles, args.crop, args.steps_per_epoch * args.batch_size, args, augment=True),
            batch_size=args.batch_size, num_workers=args.workers, drop_last=True, persistent_workers=True),
        "val": torch.utils.data.DataLoader(
            TileDataset(val_tiles, args.crop, args.val_steps * args.batch_size, args, augment=False),
            batch_size=args.batch_size, num_workers=args.workers, persistent_workers=True),
    }

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        losses = {}
        for phase, loader in loaders.items():
            head.train(phase == "train")
            total, count = 0.0, 0
            for images, targets in loader:
                images, targets = images.to(device), targets.to(device)
                with torch.no_grad():
                    tokens, _ = backbone(images)
                with torch.set_grad_enabled(phase == "train"):
                    loss = compute_loss(head(tokens, images, targets.shape[-2:]), targets)
                    if phase == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                total += loss.item() * len(images)
                count += len(images)
            losses[phase] = total / max(1, count)

        scheduler.step()
        marker = ""
        if losses["val"] < best:
            best = losses["val"]
            torch.save({"head": head.state_dict(), "val_loss": best, "args": vars(args)}, args.checkpoint)
            marker = "  <- gespeichert"
        print(f"Epoche {epoch:3d}  train {losses['train']:.4f}  val {losses['val']:.4f}{marker}")

    print(f"\nBester Validierungsverlust: {best:.4f} -> {args.checkpoint}")


def run_prediction(args, device) -> None:
    backbone = build_backbone(args, device)
    head = CrownHead().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    head.load_state_dict(state["head"])
    head.eval()
    print(f"Kopf geladen (val {state['val_loss']:.4f})")

    args.out.mkdir(parents=True, exist_ok=True)
    folders = sorted(p for p in args.frames_dir.iterdir() if p.is_dir())

    for folder in folders:
        frames = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        out_folder = args.out / folder.name
        out_folder.mkdir(parents=True, exist_ok=True)

        for frame_path in frames:
            image_bgr = cv2.imread(str(frame_path))
            if image_bgr is None:
                continue
            long_side = int(max(image_bgr.shape[:2]) * args.predict_scale)
            maps = predict_maps(backbone, head, cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), device, long_side)
            labels = instances_from_maps(maps, args)
            cv2.imwrite(str(out_folder / f"{frame_path.stem}_labels.png"), labels.astype(np.uint16))

            overlay = image_bgr.copy()
            work = labels.astype(np.uint16)
            kernel = np.ones((3, 3), np.uint8)
            borders = (cv2.dilate(work, kernel) != cv2.erode(work, kernel)) & (labels > 0)
            overlay[borders] = (80, 230, 120)
            caption = f"{int(labels.max())} Kronen"
            cv2.rectangle(overlay, (0, 0), (360, 34), (0, 0, 0), -1)
            cv2.putText(overlay, caption, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(out_folder / f"{frame_path.stem}_crownnet.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"  {folder.name}/{frame_path.name}: {caption}")

    print(f"\nInstanzen: {args.out}")


def match_instances(predicted: np.ndarray, truth: np.ndarray, threshold: float) -> tuple[int, int, int, list[float]]:
    """Greedy-Zuordnung ueber IoU. Gibt (Treffer, Fehlalarme, Verfehlte, IoUs)."""
    pred_ids = [i for i in np.unique(predicted) if i > 0]
    true_ids = [i for i in np.unique(truth) if i > 0]
    if not pred_ids or not true_ids:
        return 0, len(pred_ids), len(true_ids), []

    # Ueberschneidungsmatrix ueber ein gemeinsames Histogramm -- deutlich
    # schneller als paarweise Maskenvergleiche.
    pred_index = {value: i for i, value in enumerate(pred_ids)}
    true_index = {value: i for i, value in enumerate(true_ids)}
    overlap = np.zeros((len(pred_ids), len(true_ids)), dtype=np.int64)

    both = (predicted > 0) & (truth > 0)
    for p_value, t_value in zip(predicted[both], truth[both]):
        overlap[pred_index[p_value], true_index[t_value]] += 1

    pred_area = np.array([(predicted == v).sum() for v in pred_ids])[:, None]
    true_area = np.array([(truth == v).sum() for v in true_ids])[None, :]
    iou = overlap / np.maximum(1, pred_area + true_area - overlap)

    matched_pred, matched_true, scores = set(), set(), []
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    for i, j in order:
        if iou[i, j] < threshold:
            break
        if i in matched_pred or j in matched_true:
            continue
        matched_pred.add(int(i))
        matched_true.add(int(j))
        scores.append(float(iou[i, j]))

    return len(scores), len(pred_ids) - len(scores), len(true_ids) - len(scores), scores


def run_evaluation(args, device) -> None:
    """Instanzgenauigkeit gegen die BAMFORESTS-Labels -- die erste harte Zahl.

    Ausgewertet wird nach Gebiet getrennt: `Hain` kommt in Training und
    Validierung nicht vor und ist damit der einzige echte Uebertragungstest.
    """
    import collections

    import pandas as pd

    tiles = sorted(
        p for split in args.eval_splits for p in (args.prepared / split).glob("*.jpg")
    )
    if not tiles:
        print(f"Keine Kacheln in {args.eval_splits}")
        return

    # Nach Gebiet schichten -- alphabetisch sortiert waeren sonst alle Kacheln
    # aus demselben Gebiet, und der Uebertragungstest waere keiner.
    by_area: dict[str, list] = {}
    for tile in tiles:
        by_area.setdefault(tile.stem.split("_")[0], []).append(tile)
    per_area = max(1, args.eval_tiles // max(1, len(by_area)))
    tiles = [t for area in sorted(by_area) for t in by_area[area][:per_area]]
    print("Kacheln je Gebiet: " + ", ".join(f"{a} {min(len(v), per_area)}" for a, v in sorted(by_area.items())))

    if args.labels_from:
        print(f"Bewerte vorhandene Labelkarten aus {args.labels_from}")
        backbone = head = None
    else:
        backbone = build_backbone(args, device)
        head = CrownHead().to(device)
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        head.load_state_dict(state["head"])
        head.eval()
        print(f"Kopf geladen (val {state['val_loss']:.4f})")

    stats = collections.defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "iou": []})
    for tile in tiles:
        truth = cv2.imread(str(tile.with_name(tile.stem + "_labels.png")), cv2.IMREAD_UNCHANGED).astype(np.int32)

        if args.labels_from:
            candidate = args.labels_from / f"{tile.stem}_labels.png"
            if not candidate.exists():
                continue
            predicted = cv2.imread(str(candidate), cv2.IMREAD_UNCHANGED).astype(np.int32)
        else:
            image = cv2.cvtColor(cv2.imread(str(tile)), cv2.COLOR_BGR2RGB)
            predicted = instances_from_maps(predict_maps(backbone, head, image, device, args.long_side), args)

        tp, fp, fn, ious = match_instances(predicted, truth, args.iou_thresh)
        area = tile.stem.split("_")[0]
        stats[area]["tp"] += tp
        stats[area]["fp"] += fp
        stats[area]["fn"] += fn
        stats[area]["iou"].extend(ious)

    rows = []
    for area, value in sorted(stats.items()):
        precision = value["tp"] / max(1, value["tp"] + value["fp"])
        recall = value["tp"] / max(1, value["tp"] + value["fn"])
        rows.append({
            "Gebiet": area,
            "Kronen(GT)": value["tp"] + value["fn"],
            "Praezision": round(precision, 3),
            "Trefferquote": round(recall, 3),
            "F1": round(2 * precision * recall / max(1e-9, precision + recall), 3),
            "mittlere_IoU": round(float(np.mean(value["iou"])) if value["iou"] else 0.0, 3),
        })
    print(f"\n=== Instanzgenauigkeit bei IoU >= {args.iou_thresh} ===")
    print(pd.DataFrame(rows).to_string(index=False))


def run_inspect(args, device) -> None:
    """Ziel- und Vorhersagekarten nebeneinander -- zeigt, ob das Netz oder die
    Instanzbildung das Problem ist."""
    backbone = build_backbone(args, device)
    head = CrownHead().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    head.load_state_dict(state["head"])
    head.eval()

    tiles = sorted((args.prepared / "val").glob("*.jpg"))[: args.inspect_tiles]
    args.out.mkdir(parents=True, exist_ok=True)

    for tile in tiles:
        image = cv2.cvtColor(cv2.imread(str(tile)), cv2.COLOR_BGR2RGB)
        truth = cv2.imread(str(tile.with_name(tile.stem + "_labels.png")), cv2.IMREAD_UNCHANGED).astype(np.int32)
        targets = targets_from_labels(truth, args.boundary_px, args.sigma)
        maps = predict_maps(backbone, head, image, device, max(image.shape[:2]))
        instances = instances_from_maps(maps, args)

        def panel(a, title):
            u = (np.clip(a, 0, 1) * 255).astype(np.uint8)
            u = cv2.cvtColor(u, cv2.COLOR_GRAY2BGR) if u.ndim == 2 else u
            cv2.putText(u, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            return u

        top = np.hstack([cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                         panel(targets[0], "Ziel inneres"), panel(targets[1], "Ziel rand")])
        bottom = np.hstack([panel((instances > 0).astype(np.float32), f"Instanzen: {instances.max()}"),
                            panel(maps[0], "Vorhersage inneres"), panel(maps[1], "Vorhersage rand")])
        cv2.imwrite(str(args.out / f"{tile.stem}_inspect.jpg"), np.vstack([top, bottom]),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  {tile.stem}: GT {truth.max()} Kronen, vorhergesagt {instances.max()} | "
              f"inneres mean {maps[0].mean():.3f} max {maps[0].max():.3f} | "
              f"rand mean {maps[1].mean():.3f} max {maps[1].max():.3f} | "
              f"Ziel inneres mean {targets[0].mean():.3f}")
    print(f"\nPanels: {args.out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("prepare", "train", "predict", "eval", "inspect"), default="train")
    parser.add_argument("--bamforests", type=Path,
                        default=Path("/scratch/shared/nik/data/bamforests/coco2048"))
    parser.add_argument("--prepared", type=Path,
                        default=Path("/scratch/shared/nik/data/bamforests/prepared"))
    parser.add_argument("--frames-dir", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_crownnet")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/checkpoints/crownnet.pth"))
    parser.add_argument("--ckpt", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/checkpoints/dinovtreeb_quebectrees.pth"))
    parser.add_argument("--categories", type=Path, default=REPO_ROOT / "third_party" / "quebec_trees_categories.json")

    parser.add_argument("--scale", type=float, default=0.34,
                        help="BAMFORESTS auf den Massstab der eigenen Frames bringen.")
    parser.add_argument("--boundary-px", type=int, default=3, help="Breite des Randsaums.")
    parser.add_argument("--sigma", type=float, default=6.0, help="Streuung der Zentrums-Gausskurven.")

    parser.add_argument("--crop", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps-per-epoch", type=int, default=128)
    parser.add_argument("--val-steps", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=8)

    parser.add_argument("--interior-thresh", type=float, default=0.5)
    parser.add_argument("--crown-thresh", type=float, default=0.5)
    parser.add_argument("--min-seed-px", type=int, default=60)
    parser.add_argument("--long-side", type=int, default=1920)
    parser.add_argument("--predict-scale", type=float, default=1.0,
                        help="Eigene Frames hochskalieren, damit eine Krone genauso viele Patches "
                             "belegt wie im Training. Massgeblich ist die Patchzahl, nicht die Pixelzahl.")
    parser.add_argument("--eval-splits", nargs="*", default=["test1", "test2"])
    parser.add_argument("--eval-tiles", type=int, default=200, help="Obergrenze, haelt die Auswertung kurz.")
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--inspect-tiles", type=int, default=3)
    parser.add_argument("--labels-from", type=Path, default=None,
                        help="Statt selbst vorherzusagen: fertige Labelkarten bewerten (Vergleich mit SAM 3).")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | Modus: {args.mode}\n")

    if args.mode == "prepare":
        run_prepare(args)
    elif args.mode == "train":
        run_training(args, device)
    elif args.mode == "eval":
        run_evaluation(args, device)
    elif args.mode == "inspect":
        run_inspect(args, device)
    else:
        run_prediction(args, device)


if __name__ == "__main__":
    main()
