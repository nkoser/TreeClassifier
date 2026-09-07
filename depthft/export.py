"""Den feinabgestimmten Checkpoint versandfertig machen.

Herauskommt ein Ordner, den jemand ohne dieses Repository benutzen kann:
Gewichte im Hugging-Face-Format, der passende Bildprozessor, das schlanke
Anwendungsmodul und eine Modellkarte, in der die eine Sache steht, an der sonst
alles scheitert -- dass der Bildwinkel vorgegeben und nicht geschaetzt gehoert.

    python depthft/export.py --ft /scratch/shared/nik/runs/depthft/bestes --tar
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

MODELCARD = """# Depth Pro, fine-tuned on FORTRESS (UAV nadir, metric height)

Base model: `apple/DepthPro-hf`
Fine-tuned on: FORTRESS (Schiefer, Frey & Kattenborn 2022, CC BY 4.0) --
47 UAV sites in the southern Black Forest with orthomosaic and normalised
height model (nDSM).

## What it is for

Off-the-shelf Depth Pro is trained on ground-level perspectives. On nadir images
from a drone the structure is often right but the **scale** is not -- crowns sit
too high, tree heights come out too small. This checkpoint corrects that for
exactly this capture situation: view straight down, flight altitude between about
25 and 120 m, forest.

## Loading and applying

```python
import numpy as np, torch
from transformers import AutoImageProcessor, DepthProForDepthEstimation

model = DepthProForDepthEstimation.from_pretrained("PATH").eval()
processor = AutoImageProcessor.from_pretrained("PATH")

bild = ...                       # RGB uint8, HxWx3
inputs = processor(images=bild, return_tensors="pt")
with torch.no_grad():
    ausgabe = model(**inputs)
```

### Important: supply the field of view, do not let it be estimated

> **This is the setting everything depends on.** An error in the field of view
> enters every depth linearly and cannot be seen in the depth map -- it looks
> right and is not. In the originating project the assumed value (73.7 degrees)
> was off by a factor of 1.7; back-calculation from frames with a known flight
> altitude gave around 48 degrees. Anyone applying this model to a different
> camera should take the value from EXIF or calibrate it.

Depth Pro predicts canonical inverse depth. Metres only arise via the camera:

    d = (f_px / image width) / D_raw

The built-in field-of-view head estimates `f_px` as well -- but it is trained on
ground-level perspectives and is regularly wrong on nadir captures. An error
there enters every depth **linearly**. With a known camera, compute it yourself:

```python
f_px = 0.5 * bildbreite / np.tan(np.radians(HFOV_GRAD) / 2)

# The Hugging Face processor has no parameter for a known f_px and would use the
# unchanged field-of-view head here. So do it directly:
D = torch.nn.functional.interpolate(
    ausgabe.predicted_depth[:, None], size=bild.shape[:2],
    mode="bilinear", align_corners=False)[0, 0]
tiefe_m = (f_px / bildbreite) / D.clamp_min(1e-6)
hoehe_ueber_boden = flughoehe_m - tiefe_m
```

`inferenz.py` in this folder does exactly that; `beispiel.py` shows it on an
image. Training used a supplied `f_px`, so this is also the route by which the
numbers below come about.

Without a known flight altitude the **crown height** remains readable, because it
is a difference and needs no reference point:

    crown height = 95th percentile of the depth - 2nd percentile of the depth

## How it was trained

Virtual nadir frames are computed from the orthomosaic and nDSM: a camera at
height `H` above the stand, depth `d = H - nDSM`, ground sampling `H / f_px`.
Flight altitude, field of view and position are drawn at random per crop. The
loss is {verlustbeschreibung} plus multi-scale gradient matching. The output
stays canonical inverse depth. The checkpoint is loadable with the normal
Hugging Face classes; the metric conversion needs the known field of view as
shown above.

The split is by site. The test-split sites named below were never seen in
training.

{einstellungen}

## Quality

{metriken}

## Limits

- **Flight altitude 25 to 120 m, nadir view, forest.** Outside that nothing is
  guaranteed. Oblique captures did not occur in training.
- **Above 120 m it extrapolates.** The FORTRESS sites are 130 m wide; a capture
  from 100 m with a wide field of view covers more than a site provides. The
  80 m case is fully covered.
- **The truth comes from orthomosaics**, not from real single captures. An ortho
  shows every tree from exactly above, a photograph shows crown flanks towards
  the image edge. For the height of a tree that hardly matters, for the exact
  position of its edge somewhat more.
- **The deepest point in the image is not the ground.** In a closed canopy it
  lies at 0.92 of the flight altitude at the median. Anyone computing the height
  above ground from an assumed flight altitude should factor that in.
- **Without a known flight altitude** the crown height stays readable (a
  difference needs no reference point), the absolute height above ground does
  not.

## Provenance of the data

FORTRESS, Schiefer, Frey & Kattenborn 2022, CC BY 4.0. Anyone publishing results
from this model should cite the dataset.
"""

BEISPIEL = '''"""Kleinstes lauffaehiges Beispiel fuer den feinabgestimmten Checkpoint."""

import sys

import cv2
import numpy as np
import torch

import inferenz

BILD = sys.argv[1] if len(sys.argv) > 1 else "frame.jpg"
HFOV_GRAD = float(sys.argv[2]) if len(sys.argv) > 2 else 73.7
FLUGHOEHE_M = float(sys.argv[3]) if len(sys.argv) > 3 else 80.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = inferenz.lade(".", device)
bild = cv2.cvtColor(cv2.imread(BILD), cv2.COLOR_BGR2RGB)

tiefe_m, k = inferenz.tiefe(model, bild, k=inferenz.k_von_fov(HFOV_GRAD), device=device)
hoehe = inferenz.hoehe_ueber_boden(tiefe_m, FLUGHOEHE_M)

print(f"Tiefe    Median {np.median(tiefe_m):6.1f} m   (Boden erwartet bei {FLUGHOEHE_M:.0f} m)")
print(f"         p95    {np.percentile(tiefe_m, 95):6.1f} m  <- geschaetzte Flughoehe")
print(f"Kronenhoehe     {np.percentile(tiefe_m, 95) - np.percentile(tiefe_m, 2):6.1f} m")
print(f"Hoehe    p95    {np.percentile(hoehe, 95):6.1f} m")

cv2.imwrite("hoehe.png", cv2.applyColorMap(
    (np.clip(hoehe / max(np.percentile(hoehe, 99), 1e-6), 0, 1) * 255).astype(np.uint8),
    cv2.COLORMAP_VIRIDIS))
print("-> hoehe.png")
'''


def tabelle(werte: dict[str, dict[str, float]], spalten: list[str]) -> str:
    kopf = "| Variant | " + " | ".join(spalten) + " |"
    trenn = "|---" * (len(spalten) + 1) + "|"
    zeilen = [f"| `{name}` | " + " | ".join(f"{eintrag.get(s, float('nan')):.3f}" for s in spalten) + " |"
              for name, eintrag in werte.items()]
    return "\n".join([kopf, trenn, *zeilen])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ft", type=Path, default=Path("/scratch/shared/nik/runs/depthft/bestes"))
    parser.add_argument("--out", type=Path, default=Path("/scratch/shared/nik/runs/depthft/versand"))
    parser.add_argument("--metriken", type=Path,
                        default=Path("/home/nik/workspace/TreeClassifier/results_depthft/zusammenfassung_test.json"))
    parser.add_argument("--basis", default="apple/DepthPro-hf", help="Woher der Bildprozessor kommt.")
    parser.add_argument("--name", default="depthpro-fortress-nadir")
    parser.add_argument("--tar", action="store_true", help="Zusaetzlich ein tar.gz danebenlegen.")
    parser.add_argument("--pruefen", action="store_true", default=True)
    parser.add_argument("--no-pruefen", dest="pruefen", action="store_false")
    args = parser.parse_args()

    if not args.ft.exists():
        raise SystemExit(f"Checkpoint fehlt: {args.ft}")
    ziel = args.out / args.name
    ziel.mkdir(parents=True, exist_ok=True)

    for datei in sorted(args.ft.iterdir()):
        if datei.is_file():
            shutil.copy2(datei, ziel / datei.name)
    print(f"Gewichte kopiert: {sum(1 for _ in ziel.iterdir())} Dateien", flush=True)

    from transformers import AutoImageProcessor
    AutoImageProcessor.from_pretrained(args.basis).save_pretrained(ziel)

    hier = Path(__file__).resolve().parent
    shutil.copy2(hier / "inferenz.py", ziel / "inferenz.py")
    (ziel / "beispiel.py").write_text(BEISPIEL)

    einstellungen, gebiete = "", ""
    verlustbeschreibung = "not documented in the checkpoint metadata"
    depthft = ziel / "depthft.json"
    if depthft.exists():
        info = json.loads(depthft.read_text())
        verlustbeschreibung = (
            "Huber on the metric height"
            if info.get("loss") == "huber_hoehe"
            else "L1 on the log depth (older training state)"
        )
        einstellungen = (
            "| Setting | Value |\n|---|---|\n"
            f"| trained parts | `{info.get('trainable')}` |\n"
            f"| flight altitude | {info.get('hoehe_min')} to {info.get('hoehe_max')} m |\n"
            f"| field of view | {info.get('fov_min')} to {info.get('fov_max')} degrees |\n"
            f"| crop | {info.get('crop_px')} px |\n"
            f"| best epoch | {info.get('epoche')} |\n"
        )
        gebiete = ", ".join(info.get("train_gebiete", []))
        if gebiete:
            einstellungen += f"\nTraining sites: {gebiete}\n"

    metriken = "_Not yet evaluated -- `depthft/evaluate.py` supplies the numbers._"
    if args.metriken.exists():
        daten = json.loads(args.metriken.read_text())
        metriken = (f"On {daten['ausschnitte']} crops of the test sites "
                    f"({', '.join(daten['gebiete'])}), which did not occur in training:\n\n"
                    + tabelle(daten["varianten"], ["absrel", "mae_m", "rmse_m", "bias_m", "delta125"])
                    + "\n\n`mae_m` is at the same time the error of the height above ground -- the "
                      "flight altitude cancels out in the difference.")

    (ziel / "README.md").write_text(MODELCARD.format(
        einstellungen=einstellungen, metriken=metriken,
        verlustbeschreibung=verlustbeschreibung,
    ))

    if args.pruefen:
        import torch
        from transformers import DepthProForDepthEstimation
        model = DepthProForDepthEstimation.from_pretrained(ziel, dtype=torch.float32)
        n = sum(p.numel() for p in model.parameters())
        hat_fov = model.fov_model is not None
        print(f"Geprueft: laedt sauber, {n/1e6:.0f} M Parameter, "
              f"Bildwinkelkopf {'enthalten' if hat_fov else 'fehlt'}", flush=True)
        del model

    groesse = sum(f.stat().st_size for f in ziel.rglob("*") if f.is_file()) / 1e9
    print(f"\n{ziel}  ({groesse:.2f} GB)")
    for datei in sorted(ziel.iterdir()):
        print(f"  {datei.name}")

    if args.tar:
        archiv = args.out / f"{args.name}.tar.gz"
        subprocess.run(["tar", "-czf", str(archiv), "-C", str(args.out), args.name], check=True)
        print(f"\n{archiv}  ({archiv.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
