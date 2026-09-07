"""Tiefenkarten fuer alle BAMFORESTS-Kacheln vorrechnen.

Fuer die Tiefe als vierten Eingabekanal muss sie zu jedem Trainingsausschnitt
verfuegbar sein -- 2456 Kacheln, im Training zufaellig ausgeschnitten. Bei jedem
Zugriff Depth Pro laufen zu lassen waere um Groessenordnungen teurer als das
eigentliche Training, also einmal vorrechnen und als PNG ablegen.

Gespeichert wird die **je Kachel normierte** Tiefe als uint8, nicht der Rohwert.
Monokulare Tiefe hat ohnehin keine verlaessliche absolute Skala -- die Modelle
sind auf Bodenperspektiven trainiert, nicht auf Nadir aus 80 m. Was traegt, ist
die relative Struktur innerhalb der Kachel, und die bleibt erhalten. Nebenbei
kostet uint8 ein Viertel von float16 (10 GB statt 40 GB fuer den ganzen Satz).

    python crownseg/depthcache.py --splits train val test1 test2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bamforests as bam  # noqa: E402

DEPTH_MODELS = {
    "depthpro": "apple/DepthPro-hf",
    "dav2": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
}


@torch.no_grad()
def depth_of(model, processor, image_rgb: np.ndarray, device) -> np.ndarray:
    inputs = processor(images=image_rgb, return_tensors="pt").to(device)
    outputs = model(**inputs)
    depth = processor.post_process_depth_estimation(
        outputs, target_sizes=[image_rgb.shape[:2]])[0]["predicted_depth"]
    return depth.float().cpu().numpy()


def to_uint8(depth: np.ndarray) -> np.ndarray:
    """Invertiert (naeher = hoeher) und auf 0..255 gespreizt.

    Die Spreizung laeuft ueber das 1.- bis 99.-Perzentil statt ueber Min und Max:
    ein einzelner Ausreisser -- eine Luecke bis zum Boden, ein Reflex -- wuerde
    sonst den gesamten nutzbaren Wertebereich zusammendruecken.
    """
    surface = -depth.astype(np.float32)
    low, high = np.percentile(surface, [1, 99])
    if high <= low:
        return np.zeros(surface.shape, np.uint8)
    return np.clip((surface - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)


def main() -> None:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", type=Path, default=bam.BAMFORESTS / "crownseg")
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test1", "test2"])
    parser.add_argument("--depth-model", default="depthpro", choices=list(DEPTH_MODELS))
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    model_id = DEPTH_MODELS[args.depth_model]
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
    print(f"Device: {device} | {model_id}", flush=True)

    for split in args.splits:
        directory = args.prepared / split
        out_dir = directory.parent / f"{split}_tiefe_{args.depth_model}"
        out_dir.mkdir(parents=True, exist_ok=True)
        tiles = sorted(directory.glob("*.jpg"))

        written = skipped = 0
        for tile in tiles:
            target = out_dir / f"{tile.stem}.png"
            if target.exists():
                skipped += 1
                continue
            image = cv2.cvtColor(cv2.imread(str(tile)), cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(target), to_uint8(depth_of(model, processor, image, device)))
            written += 1
            if written % 100 == 0:
                print(f"  {split}: {written}/{len(tiles) - skipped}", flush=True)

        print(f"{split:6s} {written} gerechnet, {skipped} vorhanden -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
