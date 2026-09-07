"""Query-based crown instances: EoMT and Mask2Former on BAMFORESTS.

Both architectures predict fixed queries, each with its own mask and its own
score -- no anchors, no NMS, no watershed post-processing. That structurally
removes the error Mask R-CNN failed on in Hain: there the anchors covered 32 to
512 px, the crowns reach up to 842 px, and anything above that could not even be
proposed by the RPN. A query has no size assumption.

  eomt          Encoder-only Mask Transformer (CVPR 2025) with a DINOv3 backbone.
                No pixel decoder, no transformer decoder -- the ViT itself carries
                the queries. It fits the project because DINOv3 is already here
                via the DINOvTree checkpoint anyway.
  mask2former   Masked-attention decoder on Swin. More mature and more widely
                proven, especially with densely packed instances.

Both run over the same data path (`bamforests.CrownCrops`), the same window
logic (`tiling.slide`) and the same metric (`metrics`). What is compared is
therefore the architecture and not the environment around it.

Both see the same ground footprint: a 1024 px crop of the tile, scaled to the
input size of the model. Same area, same crown size in metres -- only the pixel
count differs, and that belongs to the architecture.

    python crownseg/queryseg.py --arch eomt --mode train
    python crownseg/queryseg.py --arch mask2former --mode eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bamforests as bam  # noqa: E402
import metrics as met  # noqa: E402
from tiling import draw_overlay, slide, suppress, to_label_map  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = Path(f"/scratch/shared/{os.environ.get('USER', 'nik')}/data/treeclf/checkpoints")

# The `tue-mps/<task>_eomt_<...>` repos are the authors' original format and have
# no config.json. The versions converted to transformers live under the
# hyphenated form `eomt-dinov3-coco-instance-large-640`.
ARCHS = {
    "eomt": "tue-mps/eomt-dinov3-coco-instance-large-640",
    "mask2former": "facebook/mask2former-swin-base-coco-instance",
}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
# The depth is stretched to 0..255 per tile, so it is roughly uniform. Mean 0.5
# and std 0.29 put it in about the same value range as the ImageNet-normalised
# colour channels.
DEPTH_MEAN, DEPTH_STD = 0.5, 0.29


def stats(channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    if channels == 3:
        return IMAGENET_MEAN, IMAGENET_STD
    return (torch.cat([IMAGENET_MEAN, torch.tensor([[[DEPTH_MEAN]]])]),
            torch.cat([IMAGENET_STD, torch.tensor([[[DEPTH_STD]]])]))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def add_depth_channel(model) -> bool:
    """Widen the first convolution from 3 to 4 input channels.

    The new channel starts with zero weights. The model therefore behaves exactly
    like the RGB model at the first step and has to earn the benefit of the depth
    -- initialising it with the mean of the colour weights instead would make the
    network see the tile structure twice at once, and the comparison against the
    RGB run would measure that jump as well.
    """
    import torch.nn as nn

    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            weight = module.weight.data
            expanded = torch.zeros(weight.shape[0], 4, *weight.shape[2:], dtype=weight.dtype)
            expanded[:, :3] = weight
            module.in_channels = 4
            module.weight = nn.Parameter(expanded)
            return True
    return False


class GatedDepthEmbed(torch.nn.Module):
    """A separate input convolution for the depth, added via a learned weight.

    Measured, the fourth input channel with zero weights had no effect (+0.005,
    below the spread), even though the depth alone reaches 0.514 -- the network
    *may* ignore it, and as long as RGB carries the task alone there is no reason
    to use it.

    Here the depth therefore gets its own convolution, copied from the RGB
    weights. The pretrained filters are edge and texture detectors, which apply as
    sensibly to a height map as to an image. The weight `gate` starts at 0.5: the
    depth is in the token image from the first iteration on, and the network has
    to actively turn it down rather than never letting it in.

    The learned value is at the same time the measurement -- if it stays near
    zero, the network discarded the depth even when it was forced upon it.
    """

    def __init__(self, base: torch.nn.Conv2d, start: float) -> None:
        super().__init__()
        import copy

        self.rgb = base
        self.depth = copy.deepcopy(base)
        self.gate = torch.nn.Parameter(torch.tensor(float(start)))

    @property
    def weight(self) -> torch.Tensor:
        """The surrounding code reads the dtype of the input convolution through this."""
        return self.rgb.weight

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        colour = self.rgb(pixels[:, :3])
        if pixels.shape[1] < 4:
            return colour
        height = self.depth(pixels[:, 3:4].expand(-1, 3, -1, -1))
        return colour + self.gate * height


def add_gated_depth(model, start: float) -> bool:
    """Replace the first 3-channel convolution with the gated variant."""
    import torch.nn as nn

    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Conv2d) and child.in_channels == 3:
                setattr(module, child_name, GatedDepthEmbed(child, start))
                print(f"Fusion an {name}.{child_name}, Startgewicht {start}", flush=True)
                return True
    return False


def build_model(arch: str, checkpoint: str | None = None, depth: bool = False,
                fusion: str = "kanal", gate_start: float = 0.5):
    """Rebuild a pretrained head for a single class.

    The COCO head predicts 80 classes, here there is only `tree`. The class layer
    therefore does not fit and is reinitialised (`ignore_mismatched_sizes`);
    backbone, pixel decoder and mask head stay pretrained -- that is where the
    transferable knowledge sits.
    """
    from transformers import AutoModelForUniversalSegmentation

    model = AutoModelForUniversalSegmentation.from_pretrained(
        checkpoint or ARCHS[arch],
        id2label={0: "tree"},
        label2id={"tree": 0},
        ignore_mismatched_sizes=True,
    )
    if depth:
        added = add_gated_depth(model, gate_start) if fusion == "gate" else add_depth_channel(model)
        if not added:
            raise RuntimeError("Keine 3-Kanal-Faltung gefunden -- Tiefe nicht eingebaut.")
    return model


def normalize(image_uint8: np.ndarray, size: int) -> torch.Tensor:
    resized = cv2.resize(image_uint8, (size, size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1).copy()).float().div_(255.0)
    mean, std = stats(tensor.shape[0])
    return (tensor - mean) / std


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


class QueryCrops(torch.utils.data.Dataset):
    """Crops from `CrownCrops` in the format of the universal segmentation models."""

    def __init__(self, base: bam.CrownCrops, size: int) -> None:
        self.base, self.size = base, size

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, target = self.base[index]
        pixels = F.interpolate(image[None], size=(self.size, self.size),
                               mode="bilinear", align_corners=False)[0]
        mean, std = stats(pixels.shape[0])
        pixels = (pixels - mean) / std

        masks = target["masks"]
        if len(masks):
            masks = F.interpolate(masks[None].float(), size=(self.size, self.size),
                                  mode="nearest")[0]
            keep = masks.flatten(1).sum(1) > 0  # crowns that vanished when downscaling
            masks = masks[keep]
        else:
            masks = torch.zeros((0, self.size, self.size))
        return pixels, masks, torch.zeros(len(masks), dtype=torch.int64)


def collate(batch):
    pixels, masks, classes = zip(*batch)
    return torch.stack(pixels), list(masks), list(classes)


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #


@torch.no_grad()
def predict_window(model, window_rgb: np.ndarray, device, args) -> list[met.Instance]:
    """Queries scoring above the threshold, as instances at window scale."""
    height, width = window_rgb.shape[:2]
    pixels = normalize(window_rgb, args.input_size)[None].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        output = model(pixel_values=pixels)

    # The last class is "no object"; class 0 is the crown.
    scores = output.class_queries_logits[0].float().softmax(-1)[:, 0]
    keep = scores >= args.score_thresh
    if not keep.any():
        return []

    logits = output.masks_queries_logits[0][keep].float()
    logits = F.interpolate(logits[None], size=(height, width), mode="bilinear", align_corners=False)[0]
    masks = (logits.sigmoid() > 0.5).cpu().numpy()

    instances = [met.instance_from_mask(m, float(s))
                 for m, s in zip(masks, scores[keep].cpu().numpy())]
    return [i for i in instances if i is not None and i.area >= args.min_area]


def predict_tiles(model, image_rgb: np.ndarray, device, args) -> list[met.Instance]:
    return slide(image_rgb, args.eval_tile, args.overlap,
                 lambda window: predict_window(model, window, device, args))


def rescale(instance: met.Instance, factor: float) -> met.Instance | None:
    """Bring one instance from a scaled image back into original coordinates."""
    x0, y0, x1, y1 = (int(round(v * factor)) for v in instance.box)
    width, height = max(1, x1 - x0), max(1, y1 - y0)
    mask = cv2.resize(instance.mask.astype(np.uint8), (width, height),
                      interpolation=cv2.INTER_NEAREST).astype(bool)
    if not mask.any():
        return None
    return met.Instance((x0, y0, x0 + width, y0 + height), mask, instance.score)


def predict_multiscale(model, image_bgr: np.ndarray, device, args,
                       scales: list[float]) -> list[met.Instance]:
    """The same capture at several resolutions, with the results merged.

    The image scale of our own frames is only estimated, and a trained model looks
    for crowns at the size it learned. Several scales side by side make the
    prediction independent of that estimate -- the same reason `segment_sam3.py`
    runs over several tile levels. Merging is by confidence, and strongly
    overlapping masks from neighbouring scales drop out in the process.
    """
    height, width = image_bgr.shape[:2]
    collected: list[met.Instance] = []
    for scale in scales:
        work = cv2.resize(image_bgr, (int(round(width * scale)), int(round(height * scale))),
                          interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        found = predict_tiles(model, cv2.cvtColor(work, cv2.COLOR_BGR2RGB), device, args)
        collected.extend(i for i in (rescale(f, 1.0 / scale) for f in found) if i is not None)
    return suppress(collected, args.merge_iou) if len(scales) > 1 else collected


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #


def roots(args) -> list[Path]:
    return args.prepared if isinstance(args.prepared, list) else [args.prepared]


def depth_dir_for(args, split: str, root: Path | None = None) -> Path | None:
    if not (args.depth or args.depth_only):
        return None
    directory = (root or roots(args)[0]) / f"{split}_tiefe_{args.depth_model}"
    if not directory.exists():
        raise FileNotFoundError(f"Tiefencache fehlt: {directory} -- erst depthcache.py laufen lassen.")
    return directory


def load_image(directory: Path, stem: str, args) -> np.ndarray:
    """A tile as RGB or RGB+depth, depending on the mode."""
    image = cv2.cvtColor(cv2.imread(str(directory / f"{stem}.jpg")), cv2.COLOR_BGR2RGB)
    depth_dir = depth_dir_for(args, directory.name, directory.parent)
    if depth_dir is None:
        return image
    depth = cv2.imread(str(depth_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    return np.dstack([depth] * 3) if args.depth_only else np.dstack([image, depth])


def validate_instances(model, args, device) -> float:
    """Instance F1 on whole validation tiles -- the quantity selection is made on."""
    model.eval()
    rows = []
    for directory in [root / "val" for root in roots(args)]:
        index = json.loads((directory / "annotations.json").read_text())
        stems = sorted(index)[:: max(1, len(index) // max(1, args.val_f1_tiles))][: args.val_f1_tiles]
        rows.extend(_validate_tiles(model, args, device, directory, index, stems))
    model.train()
    return met.accumulate(rows)["f1"]


def _validate_tiles(model, args, device, directory, index, stems) -> list[dict]:
    rows = []
    for stem in stems:
        _, rings = bam.load_tile(directory, stem, index)
        image = load_image(directory, stem, args)
        masks, _ = bam.masks_from_rings(rings, *image.shape[:2], args.min_area, 0.0)
        truth = [i for i in (met.instance_from_mask(m) for m in masks) if i is not None]
        rows.append(met.evaluate(predict_tiles(model, image, device, args), truth, args.iou_thresh))
    return rows


def run_training(args, device) -> None:
    model = build_model(args.arch, args.checkpoint_from, args.depth,
                        args.fusion, args.gate_start).to(device)

    sources = roots(args)
    loaders = {}
    for split, steps, augment in (("train", args.steps_per_epoch, True), ("val", args.val_steps, False)):
        # Equal weight per source, not by tile count. BAMFORESTS has 1438 training
        # tiles, Quebec 543 -- weighted by size, the dataset with the small crowns
        # would hardly appear, and that is exactly the one that is missing.
        # The orthomosaic source counts too, otherwise the loader yields more steps
        # than the learning-rate schedule plans for.
        n_parts = len(sources) + (1 if args.cog_root and augment else 0)
        per_source = max(1, (steps * args.batch_size) // n_parts)
        parts = [QueryCrops(bam.CrownCrops(
            root, split, args.crop, per_source, augment=augment,
            scale_jitter=tuple(args.scale_jitter) if augment else (1.0, 1.0),
            depth_dir=depth_dir_for(args, split, root), depth_only=args.depth_only), args.input_size)
            for root in sources]
        # Crops with a freely chosen field of view, straight from the orthomosaic.
        # The pre-cut tiles can widen the scale only to about 5.4 cm/px; here up to
        # 8.7 cm/px is possible, limited by the 200 queries of the model and not by
        # the data.
        if args.cog_root and augment:
            from quebec_cog import CogCrops

            parts.append(QueryCrops(CogCrops(
                args.cog_root, args.cog_zones, args.cog_date, args.input_size,
                per_source, gsd_range=tuple(args.cog_gsd), augment=True,
                max_instances=args.cog_max_instances), args.input_size))

        dataset = parts[0] if len(parts) == 1 else torch.utils.data.ConcatDataset(parts)
        loaders[split] = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, collate_fn=collate, shuffle=len(parts) > 1,
            num_workers=args.workers, drop_last=augment, persistent_workers=args.workers > 0)

    quelle = "nur Tiefe" if args.depth_only else "RGB + Tiefe" if args.depth else "RGB"
    print(f"{args.arch}: {ARCHS[args.arch]} | Eingabe: {quelle} | "
          f"Quellen: {', '.join(r.parent.name + '/' + r.name for r in sources)}"
          f"{' + Orthomosaik ' + str(tuple(args.cog_gsd)) + ' cm/px' if args.cog_root else ''}\n"
          f"Ausschnitt {args.crop} px Boden -> {args.input_size} px Eingabe | "
          f"Massstab {args.scale_jitter[0]:.2f}-{args.scale_jitter[1]:.2f}\n", flush=True)

    # The pretrained backbone needs a smaller step size than the newly initialised
    # class head, otherwise its knowledge is overwritten in the first epoch.
    backbone, rest = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (backbone if ("encoder" in name or "backbone" in name) else rest).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": args.lr * args.backbone_lr_factor}, {"params": rest, "lr": args.lr}],
        weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[args.lr * args.backbone_lr_factor, args.lr],
        total_steps=args.epochs * args.steps_per_epoch, pct_start=0.1)

    best = -1.0
    last_checkpoint = args.checkpoint.with_name(args.checkpoint.stem + "_letzte.pth")
    for epoch in range(1, args.epochs + 1):
        losses = {}
        for phase, loader in loaders.items():
            model.train()
            total, count = 0.0, 0
            for pixels, masks, classes in loader:
                pixels = pixels.to(device)
                masks = [m.to(device) for m in masks]
                classes = [c.to(device) for c in classes]
                with torch.set_grad_enabled(phase == "train"), \
                        torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    loss = model(pixel_values=pixels, mask_labels=masks, class_labels=classes).loss
                if phase == "train":
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(backbone + rest, 1.0)
                    optimizer.step()
                    scheduler.step()
                total += float(loss.detach()) * len(pixels)
                count += len(pixels)
            losses[phase] = total / max(1, count)

        f1 = validate_instances(model, args, device)
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        state = {"model": model.state_dict(), "val_loss": losses["val"], "val_f1": f1,
                 "epoch": epoch, "arch": args.arch, "args": vars(args)}
        torch.save(state, last_checkpoint)
        marker = ""
        if f1 > best:
            best, marker = f1, "  <- gespeichert"
            torch.save(state, args.checkpoint)
        gates = [f"{p.item():+.3f}" for n, p in model.named_parameters() if n.endswith(".gate")]
        gate_text = f"  Tiefengewicht {' '.join(gates)}" if gates else ""
        print(f"Epoche {epoch:3d}  train {losses['train']:.4f}  val {losses['val']:.4f}  "
              f"F1 {f1:.3f}{gate_text}{marker}", flush=True)

    print(f"\nBeste Instanz-F1: {best:.3f} -> {args.checkpoint}")


def load_trained(args, device):
    model = build_model(args.arch, depth=args.depth,
                        fusion=args.fusion, gate_start=args.gate_start).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"{args.arch} geladen (Epoche {state.get('epoch', '?')}, val F1 {state.get('val_f1', float('nan')):.3f})")
    return model


def run_fusion(args, device) -> None:
    """Run two models side by side and unify their instances.

    Measured, RGB and depth have complementary profiles: on Hain RGB is more
    precise (0.578 against 0.474), while depth finds more (recall 0.562 against
    0.532). The attempt to combine both early -- depth as a fourth input channel --
    brought nothing, because the channel starts with zero weights and the model can
    simply ignore it as long as RGB carries the task alone.

    Here the two run completely separately, and only the finished instances are
    merged by confidence. The same mechanism as with multi-scale: strongly
    overlapping masks count as duplicates, the rest stays.
    """
    import collections

    import pandas as pd

    rgb_args = argparse.Namespace(**{**vars(args), "depth_only": False, "depth": False})
    depth_args = argparse.Namespace(**{**vars(args), "depth_only": True, "depth": False})
    rgb_args.checkpoint = args.checkpoint
    depth_args.checkpoint = args.fuse_with

    rgb_model = load_trained(rgb_args, device)
    depth_model = load_trained(depth_args, device)

    rows = []
    for split in args.splits:
        directory = roots(args)[0] / split
        index = json.loads((directory / "annotations.json").read_text())
        stems = sorted(index)
        if args.eval_tiles:
            stems = stems[:: max(1, len(stems) // args.eval_tiles)][: args.eval_tiles]

        per_area = collections.defaultdict(list)
        for stem in stems:
            _, rings = bam.load_tile(directory, stem, index)
            image_rgb = load_image(directory, stem, rgb_args)
            image_depth = load_image(directory, stem, depth_args)
            masks, _ = bam.masks_from_rings(rings, *image_rgb.shape[:2], args.min_area, 0.0)
            truth = [i for i in (met.instance_from_mask(m) for m in masks) if i is not None]

            found = (predict_tiles(rgb_model, image_rgb, device, rgb_args)
                     + predict_tiles(depth_model, image_depth, device, depth_args))
            per_area[stem.split("_")[0]].append(
                met.evaluate(suppress(found, args.merge_iou), truth, args.iou_thresh))

        for area, tiles in sorted(per_area.items()):
            rows.append({"arch": f"{args.arch}+tiefe", "split": split, "gebiet": area,
                         **met.accumulate(tiles)})
            print(f"  {split}/{area}: {rows[-1]['f1']:.3f} F1", flush=True)

    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "eval_fusion.csv", index=False)
    print(f"\n=== RGB + Tiefe spaet vereinigt, IoU >= {args.iou_thresh} ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def run_evaluation(args, device) -> None:
    import collections

    import pandas as pd

    model = load_trained(args, device)
    rows = []
    for split in args.splits:
        directory = roots(args)[0] / split
        index = json.loads((directory / "annotations.json").read_text())
        stems = sorted(index)
        if args.eval_tiles:
            stems = stems[:: max(1, len(stems) // args.eval_tiles)][: args.eval_tiles]

        per_area = collections.defaultdict(list)
        for stem in stems:
            _, rings = bam.load_tile(directory, stem, index)
            image = load_image(directory, stem, args)
            masks, _ = bam.masks_from_rings(rings, *image.shape[:2], args.min_area, 0.0)
            truth = [i for i in (met.instance_from_mask(m) for m in masks) if i is not None]
            predicted = predict_tiles(model, image, device, args)
            per_area[stem.split("_")[0]].append(met.evaluate(predicted, truth, args.iou_thresh))

        for area, tiles in sorted(per_area.items()):
            rows.append({"arch": args.arch, "split": split, "gebiet": area, **met.accumulate(tiles)})
            print(f"  {split}/{area}: {rows[-1]['f1']:.3f} F1", flush=True)

    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / f"eval_{args.arch}.csv", index=False)
    print(f"\n=== {args.arch}, IoU >= {args.iou_thresh}, Score >= {args.score_thresh} ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


# Ground sampling of BAMFORESTS. Foreign imagery has to be brought to it so that
# the crowns arrive at the size that was learned.
BAM_GSD_CM = 1.70


def frame_scale(args, folder: str) -> float:
    """Scale factor from flight altitude and field of view.

    A 1920 px frame from 100 m at 73.7 degrees has about 7.8 cm/px, BAMFORESTS
    1.70 cm/px -- a factor of 4.6. Without this correction the model sees crowns of
    56 px where it learned 258 px, and finds nothing.
    """
    altitude = args.altitudes.get(folder, args.altitude)
    gsd_cm = 100 * altitude * 2 * np.tan(np.radians(args.hfov_deg) / 2) / args.frame_width
    return float(gsd_cm / BAM_GSD_CM)


def run_prediction(args, device) -> None:
    model = load_trained(args, device)
    args.out.mkdir(parents=True, exist_ok=True)
    folders = sorted(p for p in args.frames_dir.iterdir() if p.is_dir()) or [args.frames_dir]

    for folder in folders:
        frames = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        out_folder = args.out / folder.name
        out_folder.mkdir(parents=True, exist_ok=True)
        for frame_path in frames:
            image_bgr = cv2.imread(str(frame_path))
            if image_bgr is None:
                continue
            height, width = image_bgr.shape[:2]
            base = (args.scales.get(folder.name)
                    or (args.predict_scale if args.predict_scale > 0 else frame_scale(args, folder.name)))
            scales = [base * step for step in args.scale_steps]
            instances = predict_multiscale(model, image_bgr, device, args, scales)
            labels = to_label_map(instances, height, width)
            caption = (f"{int(labels.max())} Kronen ({args.arch}, x{base:.2f}"
                       f"{'' if len(scales) == 1 else ' x' + '/'.join(f'{s:.2f}' for s in args.scale_steps)})")
            cv2.imwrite(str(out_folder / f"{frame_path.stem}_labels.png"), labels)
            cv2.imwrite(str(out_folder / f"{frame_path.stem}_{args.arch}.jpg"),
                        draw_overlay(image_bgr, labels, caption), [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"  {folder.name}/{frame_path.name}: {caption}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=list(ARCHS), default="eomt")
    parser.add_argument("--mode", choices=("train", "eval", "predict", "fuse"), default="train")
    parser.add_argument("--fuse-with", type=Path, default=None,
                        help="Second checkpoint (depth) for --mode fuse.")
    parser.add_argument("--prepared", type=Path, nargs="+", default=[bam.BAMFORESTS / "crownseg"],
                        help="One or more prepared datasets; several get mixed.")
    parser.add_argument("--splits-per-source", nargs="*", default=[],
                        help="Differing split names per source, e.g. test for Quebec instead of test1.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_queryseg")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-from", default=None, help="A different starting checkpoint.")

    parser.add_argument("--crop", type=int, default=1024, help="Ground footprint in tile pixels.")
    parser.add_argument("--input-size", type=int, default=640, help="Input size of the model.")
    parser.add_argument("--scale-jitter", type=float, nargs=2, default=(0.6, 1.8))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--val-steps", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-factor", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-area", type=int, default=400)
    parser.add_argument("--depth", action="store_true",
                        help="Use the depth as a fourth input channel.")
    parser.add_argument("--depth-only", action="store_true",
                        help="The height map only, copied three times -- measures its "
                             "information content without colour.")
    parser.add_argument("--depth-model", default="depthpro", choices=("depthpro", "dav2"))
    parser.add_argument("--fusion", choices=("kanal", "gate"), default="kanal",
                        help="kanal: fourth input channel (measured to have no effect). "
                             "gate: a separate convolution for the depth with a learned weight.")
    parser.add_argument("--gate-start", type=float, default=0.5)

    parser.add_argument("--splits", nargs="*", default=["test1", "test2"])
    parser.add_argument("--eval-tile", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=768)
    parser.add_argument("--eval-tiles", type=int, default=40, help="0 = every tile.")
    parser.add_argument("--val-f1-tiles", type=int, default=10)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--iou-thresh", type=float, default=0.5)

    parser.add_argument("--frames-dir", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--predict-scale", type=float, default=0.0,
                        help="Fixed factor; 0 = derive it from altitude and field of view.")
    parser.add_argument("--altitude", type=float, default=100.0)
    parser.add_argument("--altitudes", nargs="*", default=[], help="FOLDER=ALTITUDE, e.g. pines=35")
    parser.add_argument("--scales", nargs="*", default=[],
                        help="FOLDER=FACTOR, measured with scale_probe.py. Overrides --altitudes.")
    parser.add_argument("--cog-root", type=Path, default=None,
                        help="Orthomosaic root for crops with a variable field of view.")
    parser.add_argument("--cog-zones", nargs="*", default=["zone1"])
    parser.add_argument("--cog-date", default="2021-09-02")
    parser.add_argument("--cog-gsd", type=float, nargs=2, default=(2.7, 8.7),
                        help="Scale range in cm per input pixel.")
    parser.add_argument("--cog-max-instances", type=int, default=150,
                        help="Discard crops with more crowns -- the model has 200 queries.")
    parser.add_argument("--scale-steps", type=float, nargs="*", default=[0.7, 1.0, 1.4],
                        help="Multiples of the base scale that get merged. A single value "
                             "turns multi-scale off.")
    parser.add_argument("--merge-iou", type=float, default=0.4,
                        help="From this overlap on, a mask counts as a duplicate.")
    parser.add_argument("--hfov-deg", type=float, default=73.7)
    parser.add_argument("--frame-width", type=int, default=1920)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()
    args.altitudes = {p.split("=")[0]: float(p.split("=")[1]) for p in args.altitudes}
    args.scales = {p.split("=")[0]: float(p.split("=")[1]) for p in args.scales}
    if args.checkpoint is None:
        suffix = (f"_nurtiefe_{args.depth_model}" if args.depth_only
                  else f"_tiefe{'gate' if args.fusion == 'gate' else ''}_{args.depth_model}"
                  if args.depth else "")
        args.checkpoint = CHECKPOINTS / f"crownseg_{args.arch}{suffix}.pth"
    return args


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device} | {args.arch} | Modus: {args.mode}\n", flush=True)
    {"train": run_training, "eval": run_evaluation,
     "predict": run_prediction, "fuse": run_fusion}[args.mode](args, device)


if __name__ == "__main__":
    main()
