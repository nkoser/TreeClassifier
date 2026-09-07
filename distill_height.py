"""Stage 1: distil measured parallax into a single-image network.

Parallax needs several frames and is therefore unavailable in later operation.
Its knowledge can be transferred into weights, though -- that is exactly how
monocular depth models are built: trained on stereo, applied to one image.

Setup:
  backbone   DINOv3 ViT-B/16 from the DINOvTree checkpoint, frozen. Meta's
             original weights are not needed; the checkpoint contains the
             complete fine-tuned backbone.
  head       A small convolutional decoder on the patch tokens, ~2 M parameters.
  target     The measured parallax map from build_parallax.py.

The absolute scale of the parallax is unknown and varies per frame with the
baseline. Training is therefore **scale-invariant**: prediction and target are
standardised per crop before the L1 distance is formed. What is learned is thus
the shape of the relief, not its magnitude -- exactly what treetop finding and
saddle prominence need.

The split is by **folder**, not by frame: frames of the same flight are too
similar, and a random split would flatter the result.

Example:
    python distill_height.py --mode train
    python distill_height.py --mode predict --out-cache .../predicted_cache
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_probe import hillshade, normalize
from infer_species import (
    IMAGE_SUFFIXES,
    QUEBEC_TREES_EXCLUDE,
    REPO_ROOT,
    build_dinovtree,
    load_class_names,
    resolve_device,
)

PATCH = 16


class HeightHead(nn.Module):
    """Convolutional decoder from patch tokens to a dense height map."""

    def __init__(self, in_dim: int = 768, width: int = 256) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_dim, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(width, width // 2, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(width // 2, width // 4, 3, padding=1), nn.GELU(),
            nn.Conv2d(width // 4, 1, 1),
        )

    def forward(self, tokens: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        out = self.blocks(tokens)
        return F.interpolate(out, size=size, mode="bilinear", align_corners=False)


def standardize(x: torch.Tensor) -> torch.Tensor:
    """Per example to mean 0 and std 1 -- this makes the loss scale-invariant."""
    flat = x.flatten(1)
    mean = flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, keepdim=True).clamp_min(1e-6)
    return ((flat - mean) / std).view_as(x)


class CropDataset(torch.utils.data.Dataset):
    """Random crops from frames that have a measured parallax."""

    def __init__(self, samples: list[tuple[Path, Path]], crop: int, length: int, augment: bool) -> None:
        self.samples = samples
        self.crop = crop
        self.length = length
        self.augment = augment
        self.cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return self.length

    def _load(self, image_path: Path, target_path: Path):
        if image_path not in self.cache:
            image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
            target = np.load(target_path).astype(np.float32)
            self.cache[image_path] = (image, target)
        return self.cache[image_path]

    def __getitem__(self, index: int):
        rng = np.random.default_rng(index)
        image_path, target_path = self.samples[rng.integers(len(self.samples))]
        image, target = self._load(image_path, target_path)

        height, width = target.shape
        y = int(rng.integers(0, max(1, height - self.crop)))
        x = int(rng.integers(0, max(1, width - self.crop)))
        patch = image[y : y + self.crop, x : x + self.crop]
        label = target[y : y + self.crop, x : x + self.crop]

        if self.augment:
            k = int(rng.integers(4))
            patch, label = np.rot90(patch, k, (0, 1)), np.rot90(label, k, (0, 1))
            if rng.random() < 0.5:
                patch, label = patch[:, ::-1], label[:, ::-1]

        return (
            torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1)).astype(np.float32) / 255.0),
            torch.from_numpy(np.ascontiguousarray(label)[None]),
        )


def collect_samples(args) -> dict[str, list[tuple[Path, Path]]]:
    """Frames with a parallax map, grouped by folder."""
    grouped: dict[str, list[tuple[Path, Path]]] = {}
    for target_path in sorted(args.parallax_cache.glob("*.npy")):
        folder, _, stem = target_path.stem.partition("__")
        candidates = [p for p in (args.input / folder).glob(f"{stem}.*") if p.suffix.lower() in IMAGE_SUFFIXES]
        if candidates:
            grouped.setdefault(folder, []).append((candidates[0], target_path))
    return grouped


def build_backbone(args, device):
    class_names = load_class_names(args.categories, QUEBEC_TREES_EXCLUDE)
    model = build_dinovtree(args.ckpt, n_classes=len(class_names), max_height=30.0, device=device)
    backbone = model.backbone
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    backbone.eval()
    return backbone


@torch.no_grad()
def tokens_of(backbone, batch: torch.Tensor) -> torch.Tensor:
    patch_tokens, _ = backbone(batch)
    return patch_tokens


def run_training(args, device) -> None:
    grouped = collect_samples(args)
    if not grouped:
        print("Keine Frames mit Parallaxenkarte gefunden.")
        return

    val_folders = [f for f in args.val_folders if f in grouped] or [sorted(grouped)[0]]
    train_folders = [f for f in sorted(grouped) if f not in val_folders]
    train_samples = [s for f in train_folders for s in grouped[f]]
    val_samples = [s for f in val_folders for s in grouped[f]]

    print(f"Train: {len(train_samples)} Frames aus {train_folders}")
    print(f"Val:   {len(val_samples)} Frames aus {val_folders}\n")

    backbone = build_backbone(args, device)
    head = HeightHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)

    loaders = {
        "train": torch.utils.data.DataLoader(
            CropDataset(train_samples, args.crop, args.steps_per_epoch * args.batch_size, augment=True),
            batch_size=args.batch_size, num_workers=args.workers, drop_last=True),
        "val": torch.utils.data.DataLoader(
            CropDataset(val_samples, args.crop, args.val_steps * args.batch_size, augment=False),
            batch_size=args.batch_size, num_workers=args.workers),
    }

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        losses = {}
        for phase, loader in loaders.items():
            head.train(phase == "train")
            total, count = 0.0, 0
            for images, targets in loader:
                images, targets = images.to(device), targets.to(device)
                tokens = tokens_of(backbone, images)

                with torch.set_grad_enabled(phase == "train"):
                    prediction = head(tokens, targets.shape[-2:])
                    loss = F.l1_loss(standardize(prediction), standardize(targets))
                    if phase == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()

                total += loss.item() * len(images)
                count += len(images)
            losses[phase] = total / max(1, count)

        marker = ""
        if losses["val"] < best:
            best = losses["val"]
            torch.save({"head": head.state_dict(), "val_loss": best}, args.checkpoint)
            marker = "  <- gespeichert"
        print(f"Epoche {epoch:3d}  train {losses['train']:.4f}  val {losses['val']:.4f}{marker}")

    print(f"\nBester Validierungsverlust: {best:.4f} -> {args.checkpoint}")


@torch.no_grad()
def predict_frame(backbone, head, image_rgb: np.ndarray, device, long_side: int) -> np.ndarray:
    """The whole image in one pass -- no tiling, so the map stays globally consistent."""
    height, width = image_rgb.shape[:2]
    scale = long_side / max(height, width)
    new_w = int(round(width * scale / PATCH)) * PATCH
    new_h = int(round(height * scale / PATCH)) * PATCH

    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    batch = torch.from_numpy(resized.transpose(2, 0, 1).astype(np.float32) / 255.0)[None].to(device)

    tokens = tokens_of(backbone, batch)
    prediction = head(tokens, (new_h, new_w))[0, 0].cpu().numpy()
    return cv2.resize(prediction, (width, height), interpolation=cv2.INTER_LINEAR)


def run_prediction(args, device) -> None:
    backbone = build_backbone(args, device)
    head = HeightHead().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    head.load_state_dict(state["head"])
    head.eval()
    print(f"Kopf geladen (val {state['val_loss']:.4f})")

    args.out_cache.mkdir(parents=True, exist_ok=True)
    args.preview.mkdir(parents=True, exist_ok=True)

    for folder in sorted(p for p in args.input.iterdir() if p.is_dir()):
        frames = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        preview_folder = args.preview / folder.name
        preview_folder.mkdir(parents=True, exist_ok=True)

        for frame_path in frames:
            image = cv2.imread(str(frame_path))
            if image is None:
                continue
            prediction = predict_frame(backbone, head, cv2.cvtColor(image, cv2.COLOR_BGR2RGB), device, args.long_side)
            np.save(args.out_cache / f"{folder.name}__{frame_path.stem}.npy", prediction.astype(np.float32))

            surface = normalize(prediction)
            cv2.imwrite(str(preview_folder / f"{frame_path.stem}_hillshade.jpg"),
                        (hillshade(surface) * 255).astype(np.uint8))
        print(f"  {folder.name}: {len(frames)} Frames")

    print(f"\nVorhergesagte Hoehenkarten: {args.out_cache}")


@torch.no_grad()
def run_evaluation(args, device) -> None:
    """How well does the prediction match the measurement -- and does it beat the alternatives?

    It is compared against two references: a constant prediction (what a model
    that learned nothing achieves) and Depth Anything (the off-the-shelf monocular
    model we want to replace). Only if the distilled head beats both has the
    distillation achieved anything.
    """
    from scipy.stats import spearmanr

    from segment_trees import build_pseudo_chm

    backbone = build_backbone(args, device)
    head = HeightHead().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    head.load_state_dict(state["head"])
    head.eval()

    grouped = collect_samples(args)
    val_folders = [f for f in args.val_folders if f in grouped] or [sorted(grouped)[0]]

    rows = []
    for folder in val_folders:
        for image_path, target_path in grouped[folder]:
            image_rgb = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
            measured = np.load(target_path).astype(np.float32)
            measured = measured - cv2.GaussianBlur(measured, (0, 0), 300.0)

            predicted = predict_frame(backbone, head, image_rgb, device, args.long_side)
            predicted = predicted - cv2.GaussianBlur(predicted, (0, 0), 300.0)

            depth_path = args.depth_cache / f"{folder}__{image_path.stem}.npy"
            monocular = (
                build_pseudo_chm(np.load(depth_path), 100.0, 3.0) if depth_path.exists() else None
            )

            # A sample is enough and keeps Spearman affordable.
            index = np.random.default_rng(0).choice(measured.size, size=20000, replace=False)
            flat_measured = measured.ravel()[index]
            rows.append({
                "folder": folder,
                "frame": image_path.stem,
                "destilliert": spearmanr(predicted.ravel()[index], flat_measured).statistic,
                "monokular": (
                    spearmanr(monocular.ravel()[index], flat_measured).statistic
                    if monocular is not None else float("nan")
                ),
            })
            print(f"  {folder}/{image_path.stem}: destilliert {rows[-1]['destilliert']:+.3f}, "
                  f"monokular {rows[-1]['monokular']:+.3f}")

    import pandas as pd

    frame = pd.DataFrame(rows)
    print("\n=== Rangkorrelation zur gemessenen Parallaxe (Validierungsordner) ===")
    print(frame.groupby("folder")[["destilliert", "monokular"]].mean().round(3).to_string())
    print(f"\nMittel destilliert: {frame['destilliert'].mean():+.3f}")
    print(f"Mittel monokular:   {frame['monokular'].mean():+.3f}")
    print("(0 = kein Zusammenhang, 1 = perfekt. Eine konstante Vorhersage ergibt 0.)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("train", "predict", "eval"), default="train")
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--parallax-cache", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/parallax_cache"))
    parser.add_argument("--depth-cache", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/depth_cache"))
    parser.add_argument("--out-cache", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/predicted_height_cache"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/checkpoints/height_head.pth"))
    parser.add_argument("--preview", type=Path, default=REPO_ROOT / "results_height_pred")
    parser.add_argument("--ckpt", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/checkpoints/dinovtreeb_quebectrees.pth"))
    parser.add_argument("--categories", type=Path, default=REPO_ROOT / "third_party" / "quebec_trees_categories.json")

    parser.add_argument("--val-folders", nargs="*", default=["dense", "mixed"])
    parser.add_argument("--crop", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps-per-epoch", type=int, default=64)
    parser.add_argument("--val-steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--long-side", type=int, default=1024, help="Image size at prediction time.")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | Modus: {args.mode}\n")

    if args.mode == "train":
        run_training(args, device)
    elif args.mode == "eval":
        run_evaluation(args, device)
    else:
        run_prediction(args, device)


if __name__ == "__main__":
    main()
