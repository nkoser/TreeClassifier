"""Ablation: which SAM variant delineates tree crowns best?

Compares several SAM checkpoints under identical filter and overlap rules, so
that the differences really come from the model. Measured per frame: number of
raw masks, number of accepted crowns, area coverage, median diameter,
compactness and runtime.

SAM 3 (facebook/sam3, sam3.1) is access-restricted on HuggingFace and returns
HTTP 401 without a token. Once access is granted the checkpoint can simply be
added to MODELS -- conceptually SAM 3 is the best fit, because a text prompt
("tree") segments on the term directly instead of on a point grid.

Example:
    python ablate_sam.py --models sam_vit_huge sam2.1_large
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from infer_species import REPO_ROOT, resolve_device
from segment_sam import draw, select_crowns

MODELS = {
    "sam_vit_base": "facebook/sam-vit-base",
    "sam_vit_large": "facebook/sam-vit-large",
    "sam_vit_huge": "facebook/sam-vit-huge",
    "sam2_large": "facebook/sam2-hiera-large",
    "sam2.1_large": "facebook/sam2.1-hiera-large",
    "sam2.1_base": "facebook/sam2.1-hiera-base-plus",
    # "sam3": "facebook/sam3",  # gated -- access required
}

DEFAULT_FRAMES = [
    "100/frame_000537.jpg",
    "80m/frame_000297.jpg",
    "dense/frame_000073.jpg",
    "pines/frame_000006.jpg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--frames", nargs="*", default=DEFAULT_FRAMES)
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_ablation")

    # Identical to the defaults in segment_sam.py, so that the comparison is fair.
    parser.add_argument("--crown-px", type=float, default=100.0)
    parser.add_argument("--min-area-factor", type=float, default=0.12)
    parser.add_argument("--max-area-factor", type=float, default=5.0)
    parser.add_argument("--min-compactness", type=float, default=0.25)
    parser.add_argument("--min-solidity", type=float, default=0.65)
    parser.add_argument("--max-overlap", type=float, default=0.30)
    parser.add_argument("--points-per-crop", type=int, default=48)
    parser.add_argument("--crop-layers", type=int, default=2)
    parser.add_argument("--points-per-batch", type=int, default=256)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.70)
    parser.add_argument("--stability-score-thresh", type=float, default=0.85)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}\n")

    images = {}
    for relative in args.frames:
        path = args.input / relative
        if not path.exists():
            print(f"fehlt: {path}")
            continue
        bgr = cv2.imread(str(path))
        images[relative] = (bgr, Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))

    rows = []
    for name in args.models:
        model_id = MODELS[name]
        print(f"=== {name} ({model_id}) ===")

        try:
            from transformers import pipeline

            generator = pipeline(
                "mask-generation",
                model=model_id,
                device=0 if str(device).startswith("cuda") else -1,
                points_per_crop=args.points_per_crop,
                crop_n_layers=args.crop_layers,
                points_per_batch=args.points_per_batch,
                pred_iou_thresh=args.pred_iou_thresh,
                stability_score_thresh=args.stability_score_thresh,
            )
        except Exception as error:  # gated, missing architecture, OOM while loading
            print(f"  uebersprungen: {type(error).__name__}: {str(error)[:160]}")
            continue

        for relative, (bgr, pil) in images.items():
            try:
                start = time.time()
                with torch.no_grad():
                    output = generator(pil)
                elapsed = time.time() - start
            except Exception as error:
                print(f"  {relative}: FEHLER {type(error).__name__}: {str(error)[:120]}")
                continue

            masks = [np.asarray(m, dtype=bool) for m in output["masks"]]
            scores = np.asarray(output["scores"], dtype=np.float32)
            crowns, kept = select_crowns(masks, scores, args)
            covered = float(np.any(np.stack(kept), axis=0).mean()) if kept else 0.0

            out_folder = args.out / name
            out_folder.mkdir(parents=True, exist_ok=True)
            caption = (
                f"{name} | {len(masks)} Masken -> {len(crowns)} Kronen | "
                f"Abdeckung {covered:.0%} | {elapsed:.1f}s"
            )
            cv2.imwrite(
                str(out_folder / f"{relative.replace('/', '__').rsplit('.', 1)[0]}.jpg"),
                draw(bgr, kept, crowns, caption),
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )

            rows.append(
                {
                    "modell": name,
                    "frame": relative,
                    "rohmasken": len(masks),
                    "kronen": len(crowns),
                    "abdeckung": round(covered, 3),
                    "durchmesser_px": round(crowns["durchmesser_px"].median(), 1) if len(crowns) else np.nan,
                    "kompaktheit": round(crowns["kompaktheit"].median(), 3) if len(crowns) else np.nan,
                    "sekunden": round(elapsed, 1),
                }
            )
            print(f"  {relative}: {caption}")

        del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        print("Keine Ergebnisse.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "ablation.csv", index=False)

    print("\n=== Kronen je Modell und Frame ===")
    print(df.pivot(index="modell", columns="frame", values="kronen").to_string())
    print("\n=== Flaechenabdeckung ===")
    print(df.pivot(index="modell", columns="frame", values="abdeckung").to_string())
    print("\n=== Mittelwerte ueber alle Frames ===")
    print(
        df.groupby("modell")
        .agg(
            rohmasken=("rohmasken", "mean"),
            kronen=("kronen", "mean"),
            abdeckung=("abdeckung", "mean"),
            durchmesser_px=("durchmesser_px", "mean"),
            kompaktheit=("kompaktheit", "mean"),
            sekunden=("sekunden", "mean"),
        )
        .round(2)
        .sort_values("abdeckung", ascending=False)
        .to_string()
    )


if __name__ == "__main__":
    main()
