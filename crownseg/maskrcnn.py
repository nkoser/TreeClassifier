"""Crown instances with Mask R-CNN on BAMFORESTS.

Why this design and not the head from `crownnet.py`: there the instances only
arise afterwards by watershed from three maps, so the network never predicts an
instance, only where an interior ends. With interlocking crowns that is exactly
where it breaks down -- the measured F1 values were 0.04 to 0.12 (IoU 0.5). Mask
R-CNN predicts every crown individually, with a confidence, and instances may
overlap. Crowns do overlap in reality, and a label image cannot represent that.

The mask head resolves at 56x56 (`--mask-pool 28` instead of the usual 14),
because at a mean edge length of 258 px the crowns are considerably larger than
COCO objects; at 28x28 every mask pixel would be 9 image pixels wide.

    python crownseg/maskrcnn.py --mode train
    python crownseg/maskrcnn.py --mode eval  --splits test1 test2
    python crownseg/maskrcnn.py --mode predict --frames-dir /cold/Mahfuz/chosen_frames
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bamforests as bam  # noqa: E402
import metrics as met  # noqa: E402
from tiling import draw_overlay, slide, to_label_map  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = Path(f"/scratch/shared/{os.environ.get('USER', 'nik')}/data/treeclf/checkpoints")

# Ground sampling of BAMFORESTS. Determines the factor by which foreign imagery
# has to be scaled so that the crowns arrive at the size that was learned.
BAM_GSD_CM = 1.70


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def build_model(mask_pool: int, detections: int, anchor_scale: float = 2.0, pretrained: bool = True):
    """Mask R-CNN, adapted to the size distribution of the crowns.

    The default anchors cover 32 to 512 px -- meant for COCO objects. The crowns
    here have a median of 281 px (Stadtwald) to 392 px (Hain), and the p95 reaches
    842 px. With the default anchors the RPN cannot propose anything above 512 px,
    no matter how long you train; those very trees were broken into pieces in
    Hain instead. `anchor_scale` shifts the ladder to 64 to 1024 px. Anchor sizes
    are not learned weights and the number of anchors per location stays three --
    the head fits unchanged.
    """
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    from torchvision.ops import MultiScaleRoIAlign

    model = maskrcnn_resnet50_fpn_v2(
        weights="DEFAULT" if pretrained else None,
        box_detections_per_img=detections,
    )
    if anchor_scale != 1.0:
        sizes = tuple((int(round(size * anchor_scale)),) for size in (32, 64, 128, 256, 512))
        model.rpn.anchor_generator = AnchorGenerator(sizes, ((0.5, 1.0, 2.0),) * len(sizes))
    in_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_box, 2)
    in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, 2)
    if mask_pool != 14:
        model.roi_heads.mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"], output_size=mask_pool, sampling_ratio=2)
    return model


def fix_input_size(model, size: int) -> None:
    """No rescaling inside the model -- the scale is set outside."""
    model.transform.min_size = (size,)
    model.transform.max_size = size


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #


def predict_tiles(model, image_rgb: np.ndarray, device, args) -> list[met.Instance]:
    """The shared window logic from tiling.py with the Mask R-CNN window step."""
    return slide(image_rgb, args.eval_tile, args.overlap,
                 lambda window: predict_window(model, window, device, args.score_thresh))


@torch.no_grad()
def predict_window(model, image_rgb: np.ndarray, device, score_thresh: float) -> list[met.Instance]:
    batch = torch.from_numpy(image_rgb.transpose(2, 0, 1).copy()).float().div_(255.0).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        output = model([batch])[0]
    keep = output["scores"] >= score_thresh
    masks = (output["masks"][keep, 0].float() > 0.5).cpu().numpy()
    scores = output["scores"][keep].float().cpu().numpy()
    instances = [met.instance_from_mask(m, float(s)) for m, s in zip(masks, scores)]
    return [i for i in instances if i is not None]


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #


def validate_instances(model, args, device) -> float:
    """Instance F1 on a few whole validation tiles.

    With Mask R-CNN the validation loss is unfit for model selection -- it
    routinely keeps rising while accuracy is still improving, because it averages
    over RPN samples rather than over instances. So what is measured is what
    matters: crowns hit at IoU 0.5.
    """
    directory = args.prepared / "val"
    index = json.loads((directory / "annotations.json").read_text())
    stems = sorted(index)[:: max(1, len(index) // max(1, args.val_f1_tiles))][: args.val_f1_tiles]

    model.eval()
    fix_input_size(model, args.eval_tile)
    rows = []
    for stem in stems:
        image, rings = bam.load_tile(directory, stem, index)
        masks, _ = bam.masks_from_rings(rings, *image.shape[:2], args.min_area, 0.0)
        truth = [i for i in (met.instance_from_mask(m) for m in masks) if i is not None]
        predicted = predict_tiles(model, image, device, args)
        rows.append(met.evaluate(predicted, truth, args.iou_thresh))

    fix_input_size(model, args.crop)
    model.train()
    return met.accumulate(rows)["f1"]


def run_training(args, device) -> None:
    model = build_model(args.mask_pool, args.detections, args.anchor_scale).to(device)
    fix_input_size(model, args.crop)

    loaders = {
        "train": torch.utils.data.DataLoader(
            bam.CrownCrops(args.prepared, "train", args.crop, args.steps_per_epoch * args.batch_size,
                           augment=True, scale_jitter=tuple(args.scale_jitter)),
            batch_size=args.batch_size, num_workers=args.workers, collate_fn=bam.collate,
            drop_last=True, persistent_workers=args.workers > 0),
        "val": torch.utils.data.DataLoader(
            bam.CrownCrops(args.prepared, "val", args.crop, args.val_steps * args.batch_size, augment=False),
            batch_size=args.batch_size, num_workers=args.workers, collate_fn=bam.collate,
            persistent_workers=args.workers > 0),
    }
    print(f"Train: {len(loaders['train'].dataset.stems)} Kacheln | "
          f"Val: {len(loaders['val'].dataset.stems)} | Ausschnitt {args.crop} px | "
          f"Massstab {args.scale_jitter[0]:.2f}-{args.scale_jitter[1]:.2f}\n", flush=True)

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * args.steps_per_epoch, pct_start=0.1)

    best = -1.0
    last_checkpoint = args.checkpoint.with_name(args.checkpoint.stem + "_letzte.pth")
    for epoch in range(1, args.epochs + 1):
        losses = {}
        for phase, loader in loaders.items():
            # Mask R-CNN only returns losses in training mode; for validation the
            # mode therefore stays on, only the gradient does not.
            model.train()
            total, count = 0.0, 0
            for images, targets in loader:
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                with torch.set_grad_enabled(phase == "train"):
                    loss = sum(model(images, targets).values())
                    if phase == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
                        optimizer.step()
                        scheduler.step()
                total += float(loss.detach()) * len(images)
                count += len(images)
            losses[phase] = total / max(1, count)

        f1 = validate_instances(model, args, device)

        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        state = {"model": model.state_dict(), "val_loss": losses["val"], "val_f1": f1,
                 "epoch": epoch, "args": vars(args)}
        torch.save(state, last_checkpoint)
        marker = ""
        if f1 > best:
            best = f1
            torch.save(state, args.checkpoint)
            marker = "  <- gespeichert"
        print(f"Epoche {epoch:3d}  train {losses['train']:.4f}  val {losses['val']:.4f}  "
              f"F1 {f1:.3f}{marker}", flush=True)

    print(f"\nBeste Instanz-F1: {best:.3f} -> {args.checkpoint}")
    print(f"Letzter Stand: {last_checkpoint}")


def load_trained(args, device):
    model = build_model(args.mask_pool, args.detections, args.anchor_scale, pretrained=False).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"Modell geladen (val {state['val_loss']:.4f})")
    return model


def run_evaluation(args, device) -> None:
    import collections

    import pandas as pd

    model = load_trained(args, device)
    fix_input_size(model, args.eval_tile)
    rows = []

    for split in args.splits:
        directory = args.prepared / split
        index = json.loads((directory / "annotations.json").read_text())
        stems = sorted(index)
        if args.eval_tiles:
            # Sample evenly across the split, not the first N -- otherwise the
            # alphabetically later area drops out entirely.
            stems = stems[:: max(1, len(stems) // args.eval_tiles)][: args.eval_tiles]

        per_area = collections.defaultdict(list)
        for stem in stems:
            image, rings = bam.load_tile(directory, stem, index)
            masks, _ = bam.masks_from_rings(rings, *image.shape[:2], args.min_area, 0.0)
            truth = [i for i in (met.instance_from_mask(m) for m in masks) if i is not None]
            predicted = predict_tiles(model, image, device, args)
            per_area[stem.split("_")[0]].append(met.evaluate(predicted, truth, args.iou_thresh))

        for area, tiles in sorted(per_area.items()):
            rows.append({"split": split, "gebiet": area, **met.accumulate(tiles)})
            print(f"  {split}/{area}: {rows[-1]['f1']:.3f} F1", flush=True)

    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "eval.csv", index=False)
    print(f"\n=== Instanzgenauigkeit bei IoU >= {args.iou_thresh} (Score >= {args.score_thresh}) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n{args.out / 'eval.csv'}")


def frame_scale(args, folder: str) -> float:
    """Scale factor so that foreign frames hit the BAMFORESTS crown size."""
    altitude = args.altitudes.get(folder, args.altitude)
    gsd_cm = 100 * altitude * 2 * np.tan(np.radians(args.hfov_deg) / 2) / args.frame_width
    return float(gsd_cm / BAM_GSD_CM)


def run_prediction(args, device) -> None:
    model = load_trained(args, device)
    fix_input_size(model, args.eval_tile)
    args.out.mkdir(parents=True, exist_ok=True)

    folders = sorted(p for p in args.frames_dir.iterdir() if p.is_dir()) or [args.frames_dir]
    for folder in folders:
        frames = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        out_folder = args.out / folder.name
        out_folder.mkdir(parents=True, exist_ok=True)
        scale = (args.scales.get(folder.name)
                     or (args.predict_scale if args.predict_scale > 0 else frame_scale(args, folder.name)))

        for frame_path in frames:
            image_bgr = cv2.imread(str(frame_path))
            if image_bgr is None:
                continue
            height, width = image_bgr.shape[:2]
            work = cv2.resize(image_bgr, (int(round(width * scale)), int(round(height * scale))),
                              interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
            instances = predict_tiles(model, cv2.cvtColor(work, cv2.COLOR_BGR2RGB), device, args)
            labels = to_label_map(instances, *work.shape[:2])
            labels = cv2.resize(labels, (width, height), interpolation=cv2.INTER_NEAREST)

            caption = f"{int(labels.max())} Kronen (x{scale:.2f})"
            cv2.imwrite(str(out_folder / f"{frame_path.stem}_labels.png"), labels)
            cv2.imwrite(str(out_folder / f"{frame_path.stem}_maskrcnn.jpg"),
                        draw_overlay(image_bgr, labels, caption), [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"  {folder.name}/{frame_path.name}: {caption}", flush=True)

    print(f"\nInstanzen: {args.out}")


def run_inspect(args, device) -> None:
    """Prediction and truth side by side on a few test tiles."""
    model = load_trained(args, device)
    fix_input_size(model, args.eval_tile)
    args.out.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        directory = args.prepared / split
        index = json.loads((directory / "annotations.json").read_text())
        for stem in sorted(index)[: args.inspect_tiles]:
            image, rings = bam.load_tile(directory, stem, index)
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            masks, _ = bam.masks_from_rings(rings, *image.shape[:2], args.min_area, 0.0)
            truth = [i for i in (met.instance_from_mask(m) for m in masks) if i is not None]
            predicted = predict_tiles(model, image, device, args)

            left = draw_overlay(bgr, to_label_map(truth, *image.shape[:2]), f"GT: {len(truth)} Kronen")
            right = draw_overlay(bgr, to_label_map(predicted, *image.shape[:2]), f"Mask R-CNN: {len(predicted)}")
            cv2.imwrite(str(args.out / f"{split}_{stem}_vergleich.jpg"),
                        np.hstack([left, right]), [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"  {split}/{stem}: GT {len(truth)} | Vorhersage {len(predicted)}", flush=True)


# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("train", "eval", "predict", "inspect"), default="train")
    parser.add_argument("--prepared", type=Path, default=bam.BAMFORESTS / "crownseg")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_crownseg")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINTS / "crownseg_maskrcnn.pth")

    parser.add_argument("--crop", type=int, default=1024, help="Edge length of the training crops.")
    parser.add_argument("--scale-jitter", type=float, nargs=2, default=(0.6, 1.8),
                        help="Scale range during training; <1 simulates coarser ground sampling.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--val-steps", type=int, default=40)
    parser.add_argument("--val-f1-tiles", type=int, default=12, help="Whole tiles for the per-epoch F1.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--anchor-scale", type=float, default=2.0,
                        help="Stretches the anchor ladder; 2.0 = 64 to 1024 px.")
    parser.add_argument("--mask-pool", type=int, default=28, help="RoI grid of the mask head; the output is twice as large.")
    parser.add_argument("--detections", type=int, default=300, help="Cap on instances per window.")
    parser.add_argument("--min-area", type=int, default=400)

    parser.add_argument("--splits", nargs="*", default=["test1", "test2"])
    parser.add_argument("--eval-tile", type=int, default=2048, help="Window size at inference.")
    parser.add_argument("--overlap", type=int, default=768,
                        help="Has to be larger than the largest expected crown.")
    parser.add_argument("--eval-tiles", type=int, default=60, help="0 = every tile.")
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--inspect-tiles", type=int, default=4)

    parser.add_argument("--frames-dir", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--predict-scale", type=float, default=0.0,
                        help="Fixed factor; 0 = derive it from altitude and field of view.")
    parser.add_argument("--altitude", type=float, default=100.0)
    parser.add_argument("--altitudes", nargs="*", default=[], help="FOLDER=ALTITUDE, e.g. pines=35")
    parser.add_argument("--scales", nargs="*", default=[],
                        help="FOLDER=FACTOR, measured with scale_probe.py. Overrides --altitudes.")
    parser.add_argument("--hfov-deg", type=float, default=73.7)
    parser.add_argument("--frame-width", type=int, default=1920)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()
    args.altitudes = {p.split("=")[0]: float(p.split("=")[1]) for p in args.altitudes}
    args.scales = {p.split("=")[0]: float(p.split("=")[1]) for p in args.scales}
    return args


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device} | Modus: {args.mode}\n", flush=True)
    {"train": run_training, "eval": run_evaluation,
     "predict": run_prediction, "inspect": run_inspect}[args.mode](args, device)


if __name__ == "__main__":
    main()
