"""What could a fine-tuned SAM 3 achieve at best?

SAM 3 with a text prompt reaches F1 0.312 on Hain, the trained EoMT 0.554. The
difference is not boundary quality -- with a mean IoU of 0.780, SAM 3 has the
best in the field -- but *which* objects it takes for crowns. That is exactly
what fine-tuning would correct.

Whether it is worth it hangs on a question that can be answered without any
training: **are the right crowns contained in SAM 3 raw proposals at all?**
Fine-tuning can improve the selection, but it cannot invent a mask that was
never proposed.

So the ceiling is measured: for every true crown, whether any of the unfiltered
SAM 3 masks hits it at IoU >= 0.5. That is the recall a perfect selector on
these proposals would reach -- fine-tuning cannot go beyond it.

    python crownseg/sam3_ceiling.py --split test1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bamforests as bam  # noqa: E402
import metrics as met  # noqa: E402
from segment_sam3 import SAM3_MODEL, segment_tile  # noqa: E402


def main() -> None:
    from transformers import Sam3Model, Sam3Processor

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, default=bam.BAMFORESTS / "subset")
    parser.add_argument("--prepared", type=Path, default=bam.BAMFORESTS / "crownseg")
    parser.add_argument("--split", default="test1")
    parser.add_argument("--prompt", default="tree")
    parser.add_argument("--threshold", type=float, default=0.02,
                        help="Very low: what counts here is what gets proposed, not what survives.")
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=400)
    parser.add_argument("--tiles", type=int, default=20)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    processor = Sam3Processor.from_pretrained(SAM3_MODEL)
    model = Sam3Model.from_pretrained(SAM3_MODEL).to(device).eval()
    print(f"Device: {device} | SAM3 '{args.prompt}' | Schwelle {args.threshold}\n", flush=True)

    index = json.loads((args.prepared / args.split / "annotations.json").read_text())
    frames = sorted((args.images / args.split).glob("*.jpg"))[: args.tiles]

    total_gt = covered = total_raw = 0
    for path in frames:
        stem = path.stem
        if stem not in index:
            continue
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        size = image.shape[0]
        scale = size / 2048.0
        floor = int(args.min_area * scale**2)

        rings = [np.asarray(p, np.float32).reshape(-1, 2) * scale for p in index[stem]]
        truth_masks, _ = bam.masks_from_rings(rings, size, size, floor, 0.0)
        truth = [i for i in (met.instance_from_mask(m) for m in truth_masks) if i is not None]

        masks, scores = segment_tile(model, processor, image, args.prompt, args.threshold, device)
        proposals = [i for i in (met.instance_from_mask(np.asarray(m, bool)) for m in masks)
                     if i is not None and i.area >= floor]

        # Every true crown against all proposals -- no assignment, no competition.
        # The only question is whether the proposal exists.
        hit = sum(1 for gt in truth if any(met.iou(p, gt) >= args.iou_thresh for p in proposals))
        total_gt += len(truth)
        covered += hit
        total_raw += len(proposals)
        print(f"  {stem}: {len(proposals):4d} Vorschlaege | {hit:3d}/{len(truth):3d} Kronen erreichbar",
              flush=True)

    print(f"\n=== Obergrenze fuer ein feingetuntes SAM 3 auf {args.split} ===")
    print(f"  Rohvorschlaege gesamt        {total_raw}")
    print(f"  GT-Kronen                    {total_gt}")
    print(f"  davon von einem Vorschlag getroffen  {covered} "
          f"({covered / max(1, total_gt):.1%})")
    print(f"\n  Mehr als {covered / max(1, total_gt):.3f} Trefferquote ist mit Auswahl "
          f"auf diesen Vorschlaegen nicht zu holen.")
    print(f"  Zum Vergleich: EoMT trainiert erreicht auf Hain 0.532, SAM 3 gefiltert 0.258.")


if __name__ == "__main__":
    main()
