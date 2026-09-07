"""Estimate the true image scale of our own frames from the model response.

The flight altitudes of most folders are unknown, and the scale decides
everything: BAMFORESTS has 1.70 cm/px, and a model trained on it looks for
crowns of around 258 px. If the scale is off, it finds nothing or breaks every
crown apart.

Mask R-CNN suits the role of measuring instrument here precisely because it has
a fixed size assumption: its anchor ladder covers 64 to 1024 px, and at the
wrong scale its confidence collapses. A query-based model is poorly suited --
it delivers something at any scale.

Measurement runs per folder over several frames and several scales: how many
confident instances come out, and how high is their mean confidence. The maximum
of that indicates the scale at which the crowns have the size that was learned;
from it the ground sampling and the flight altitude follow backwards.

Decisive for comparability: what is compared is a fixed **ground** footprint,
not a fixed pixel window. A 1024 px window in a halved image shows twice as much
forest as in the original -- comparing that way measures the number of visible
trees and not how well the scale fits. So the same image region is always taken
here and only its resolution is changed: the same trees, at different sizes.

    python crownseg/scale_probe.py --scales 0.5 0.75 1.0 1.5 2.0 3.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import maskrcnn as mr  # noqa: E402

BAM_GSD_CM = 1.70


def probe_window(model, window_rgb: np.ndarray, device, tile: int) -> tuple[int, float]:
    mr.fix_input_size(model, tile)
    batch = torch.from_numpy(window_rgb.transpose(2, 0, 1).copy()).float().div_(255.0).to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        output = model([batch])[0]
    scores = output["scores"].float().cpu().numpy()
    sure = scores[scores >= 0.5]
    return len(sure), float(sure.mean()) if len(sure) else 0.0


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames-dir", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--checkpoint", type=Path,
                        default=mr.CHECKPOINTS / "crownseg_maskrcnn.pth")
    parser.add_argument("--scales", type=float, nargs="*",
                        default=[0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 4.5])
    parser.add_argument("--ground", type=int, default=1024,
                        help="Edge length of the ground footprint in original pixels.")
    parser.add_argument("--frames-per-folder", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("results_crownseg/massstab.csv"))
    parser.add_argument("--hfov-deg", type=float, default=73.7)
    parser.add_argument("--frame-width", type=int, default=1920)
    parser.add_argument("--mask-pool", type=int, default=28)
    parser.add_argument("--detections", type=int, default=300)
    parser.add_argument("--anchor-scale", type=float, default=2.0)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    model = mr.load_trained(args, device)

    rows = []
    folders = sorted(p for p in args.frames_dir.iterdir() if p.is_dir())
    for folder in folders:
        frames = sorted(p for p in folder.iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[: args.frames_per_folder]
        for scale in args.scales:
            counts, confidences = [], []
            for frame_path in frames:
                image = cv2.imread(str(frame_path))
                if image is None:
                    continue
                # Always the same ground footprint from the original image ...
                ground = min(args.ground, image.shape[0], image.shape[1])
                cy, cx = image.shape[0] // 2, image.shape[1] // 2
                patch = image[cy - ground // 2 : cy + ground // 2, cx - ground // 2 : cx + ground // 2]
                # ... only its resolution changes.
                size = max(64, int(round(ground * scale)))
                work = cv2.resize(patch, (size, size),
                                  interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
                count, confidence = probe_window(model, cv2.cvtColor(work, cv2.COLOR_BGR2RGB), device, size)
                counts.append(count)
                confidences.append(confidence)
            if counts:
                rows.append({"ordner": folder.name, "massstab": scale,
                             "kronen": float(np.mean(counts)),
                             "konfidenz": float(np.mean(confidences))})
                print(f"  {folder.name:8s} x{scale:<5.2f} {rows[-1]['kronen']:6.1f} Kronen  "
                      f"Konfidenz {rows[-1]['konfidenz']:.3f}", flush=True)

    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    print("\n=== Bester Massstab je Ordner ===")
    print(f"{'Ordner':10s} {'Massstab':>9s} {'Kronen':>8s} {'Konfidenz':>10s} "
          f"{'GSD cm/px':>10s} {'Flughoehe m':>12s}")
    for name, group in table.groupby("ordner"):
        # Crown count times confidence: many confident instances beat a few very
        # confident ones, and both beat many uncertain ones.
        best = group.loc[(group["kronen"] * group["konfidenz"]).idxmax()]
        gsd = BAM_GSD_CM / best["massstab"]
        altitude = gsd / 100 * args.frame_width / (2 * np.tan(np.radians(args.hfov_deg) / 2))
        print(f"{name:10s} {best['massstab']:9.2f} {best['kronen']:8.1f} "
              f"{best['konfidenz']:10.3f} {gsd:10.2f} {altitude:12.0f}")
    print(f"\n{args.out}")


if __name__ == "__main__":
    main()
