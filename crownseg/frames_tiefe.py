"""Das Tiefe-Modell auf die eigenen Frames -- mit purer und feinabgestimmter Tiefe.

`queryseg.py --mode predict` liest das RGB. Das mit `--depth-only` trainierte
Modell will aber die Tiefenkarte als Bild (dreimal uebereinandergelegt, so wie
im Training). Dieses Skript reicht sie durch und zeichnet das Ergebnis auf das
*Originalbild*, damit es neben den RGB-Laeufen liegt.

Beide Tiefenquellen aus `depthft/` laufen durch:

  pur              Depth Pro unveraendert -- dieselbe Quelle wie im Training
                   auf BAMFORESTS.
  feinabgestimmt   auf FORTRESS nachtrainiert, metrisch richtig.

Erwartung: kein Unterschied. Die Karten werden je Frame normiert, der Massstab
faellt also weg -- und genau den repariert das Feinabstimmen. Bleibt die
Kantenschaerfe, die AbsRel nicht misst.

    python crownseg/frames_tiefe.py --variante pur feinabgestimmt
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

from queryseg import build_model, predict_multiscale  # noqa: E402
from tiling import draw_overlay, to_label_map  # noqa: E402


def als_bild(karte: np.ndarray) -> np.ndarray:
    """Tiefenkarte je Frame normiert und dreikanalig -- wie im Training."""
    gueltig = np.isfinite(karte)
    werte = karte[gueltig]
    lo, hi = np.percentile(werte, [1, 99])
    norm = np.clip((karte - lo) / max(1e-6, hi - lo), 0, 1)
    norm[~gueltig] = 0
    return np.dstack([(norm * 255).astype(np.uint8)] * 3)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames-dir", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    p.add_argument("--karten", type=Path,
                   default=Path(__file__).resolve().parent.parent / "results_depthft_frames" / "hoehenkarten")
    p.add_argument("--variante", nargs="*", default=["pur", "feinabgestimmt"])
    p.add_argument("--checkpoint", type=Path,
                   default=Path("/scratch/shared/nik/data/treeclf/checkpoints/crownseg_eomt_nurtiefe_depthpro.pth"))
    p.add_argument("--scales", nargs="*",
                   default=["dense=1.2", "dense1=1.2", "mixed=1.0", "pines=1.2",
                            "100=1.0", "80m=1.0", "mixed1=1.0", "urban=1.5"])
    p.add_argument("--scale-steps", type=float, nargs="*", default=[0.7, 1.0, 1.4])
    p.add_argument("--score-thresh", type=float, default=0.25)
    p.add_argument("--min-area", type=int, default=400)
    p.add_argument("--eval-tile", type=int, default=1024)
    p.add_argument("--overlap", type=int, default=768)
    p.add_argument("--input-size", type=int, default=640)
    p.add_argument("--merge-iou", type=float, default=0.4,
                   help="Schwelle beim Zusammenfuehren der Massstaebe.")
    p.add_argument("--arch", default="eomt")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    args.scales = {x.split("=")[0]: float(x.split("=")[1]) for x in args.scales}
    args.predict_scale, args.max_overlap = 0.0, 0.30

    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    modell = build_model("eomt").to(device)
    modell.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model"])
    modell.eval()
    wurzel = Path(__file__).resolve().parent.parent

    for variante in args.variante:
        segmente = wurzel / f"results_frames_nurtiefe_{variante}"
        ansichten = wurzel / f"results_views_nurtiefe_{variante}"
        zaehlung = {}
        for ordner in sorted(x for x in args.frames_dir.iterdir() if x.is_dir()):
            (segmente / ordner.name).mkdir(parents=True, exist_ok=True)
            (ansichten / ordner.name).mkdir(parents=True, exist_ok=True)
            for pfad in sorted(x for x in ordner.iterdir() if x.suffix.lower() in (".jpg", ".jpeg", ".png")):
                karte_pfad = args.karten / f"{ordner.name}_{pfad.stem}_{variante}.npy"
                bild = cv2.imread(str(pfad))
                if not karte_pfad.exists() or bild is None:
                    continue
                karte = np.load(karte_pfad).astype(np.float32)
                if karte.shape[:2] != bild.shape[:2]:
                    karte = cv2.resize(karte, bild.shape[1::-1], interpolation=cv2.INTER_LINEAR)

                basis = args.scales.get(ordner.name, 1.0)
                instanzen = predict_multiscale(modell, als_bild(karte), device, args,
                                               [basis * s for s in args.scale_steps])
                labels = to_label_map(instanzen, *bild.shape[:2])
                text = f"{int(labels.max())} Kronen (nur Tiefe, {variante}, x{basis:.2f})"
                cv2.imwrite(str(segmente / ordner.name / f"{pfad.stem}_labels.png"), labels)
                cv2.imwrite(str(ansichten / ordner.name / f"{pfad.stem}_nurtiefe.jpg"),
                            draw_overlay(bild, labels, text), [cv2.IMWRITE_JPEG_QUALITY, 92])
                zaehlung.setdefault(ordner.name, []).append(int(labels.max()))
                print(f"  {ordner.name}/{pfad.stem}: {text}", flush=True)

        print(f"\n=== {variante} -> {ansichten.name} ===")
        for ordner, werte in sorted(zaehlung.items()):
            print(f"  {ordner:8s} {sum(werte):5d} Kronen in {len(werte)} Frames "
                  f"(Median {int(np.median(werte))})")
        print(flush=True)


if __name__ == "__main__":
    main()
