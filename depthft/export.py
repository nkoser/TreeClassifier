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

MODELCARD = """# Depth Pro, feinabgestimmt auf FORTRESS (UAV-Nadir, metrische Hoehe)

Ausgangsmodell: `apple/DepthPro-hf`
Feinabgestimmt auf: FORTRESS (Schiefer, Frey & Kattenborn 2022, CC BY 4.0) --
47 UAV-Gebiete im Suedschwarzwald mit Orthomosaik und normalisiertem
Hoehenmodell (nDSM).

## Wofuer

Depth Pro von der Stange ist auf Bodenperspektiven trainiert. Bei Nadirbildern
aus einer Drohne stimmt die Struktur oft, die **Skala** aber nicht -- Kronen
sitzen zu hoch, Baumhoehen kommen zu klein heraus. Dieser Checkpoint korrigiert
das fuer genau diesen Aufnahmefall: Blick senkrecht nach unten, Flughoehe
zwischen etwa 25 und 120 m, Wald.

## Laden und anwenden

```python
import numpy as np, torch
from transformers import AutoImageProcessor, DepthProForDepthEstimation

model = DepthProForDepthEstimation.from_pretrained("PFAD").eval()
processor = AutoImageProcessor.from_pretrained("PFAD")

bild = ...                       # RGB uint8, HxWx3
inputs = processor(images=bild, return_tensors="pt")
with torch.no_grad():
    ausgabe = model(**inputs)
```

### Wichtig: den Bildwinkel vorgeben, nicht schaetzen lassen

> **Das ist die Angabe, an der alles haengt.** Ein Fehler im Bildwinkel geht
> linear in jede Tiefe ein und ist der Tiefenkarte nicht anzusehen -- sie sieht
> richtig aus und ist es nicht. Im Ursprungsprojekt war der angenommene Wert
> (73.7 Grad) um Faktor 1.7 daneben; die Rueckrechnung aus Frames mit bekannter
> Flughoehe ergab rund 48 Grad. Wer dieses Modell auf eine andere Kamera
> anwendet, sollte den Wert aus EXIF nehmen oder kalibrieren.

Depth Pro sagt kanonische inverse Tiefe voraus. Meter entstehen erst durch die
Kamera:

    d = (f_px / Bildbreite) / D_roh

Der eingebaute Bildwinkelkopf schaetzt `f_px` mit -- er ist aber auf
Bodenperspektiven trainiert und liegt bei Nadiraufnahmen regelmaessig daneben.
Ein Fehler dort geht **linear** in jede Tiefe ein. Bei bekannter Kamera also
selbst rechnen:

```python
f_px = 0.5 * bildbreite / np.tan(np.radians(HFOV_GRAD) / 2)

# Der Hugging-Face-Prozessor hat keinen Parameter fuer ein bekanntes f_px und
# wuerde hier den unveraenderten Bildwinkelkopf benutzen. Deshalb direkt:
D = torch.nn.functional.interpolate(
    ausgabe.predicted_depth[:, None], size=bild.shape[:2],
    mode="bilinear", align_corners=False)[0, 0]
tiefe_m = (f_px / bildbreite) / D.clamp_min(1e-6)
hoehe_ueber_boden = flughoehe_m - tiefe_m
```

`inferenz.py` in diesem Ordner macht genau das; `beispiel.py` zeigt es an einem
Bild. Trainiert wurde mit vorgegebenem `f_px`, deshalb ist dies auch der Weg,
auf dem die unten stehenden Zahlen zustande kommen.

Ohne bekannte Flughoehe bleibt die **Kronenhoehe** ablesbar, denn sie ist eine
Differenz und braucht keinen Bezugspunkt:

    Kronenhoehe = 95. Perzentil der Tiefe - 2. Perzentil der Tiefe

## Wie trainiert wurde

Aus Orthomosaik und nDSM werden virtuelle Nadirframes gerechnet: eine Kamera in
Hoehe `H` ueber dem Bestand, Tiefe `d = H - nDSM`, Bodenaufloesung `H / f_px`.
Flughoehe, Bildwinkel und Position werden je Ausschnitt gewuerfelt. Der Verlust
ist {verlustbeschreibung} plus mehrskalige Gradientenanpassung. Die
Ausgabe bleibt kanonische inverse Tiefe. Der Checkpoint ist mit den normalen
Hugging-Face-Klassen ladbar; die metrische Nachrechnung benoetigt wie oben
gezeigt den bekannten Bildwinkel.

Aufgeteilt wurde nach Gebieten. Die unten genannten Gebiete des Testsplits waren
im Training nie zu sehen.

{einstellungen}

## Guete

{metriken}

## Grenzen

- **Flughoehe 25 bis 120 m, Nadirblick, Wald.** Ausserhalb davon ist nichts
  zugesichert. Schraege Aufnahmen kamen im Training nicht vor.
- **Ueber 120 m wird extrapoliert.** Die FORTRESS-Gebiete sind 130 m breit; eine
  Aufnahme aus 100 m mit weitem Bildwinkel deckt mehr ab, als ein Gebiet
  hergibt. Der Fall 80 m ist voll abgedeckt.
- **Die Wahrheit stammt aus Orthomosaiken**, nicht aus echten Einzelaufnahmen.
  Ein Ortho zeigt jeden Baum von genau oben, ein Foto zeigt Kronenflanken zum
  Bildrand hin. Fuer die Hoehe eines Baumes spielt das kaum eine Rolle, fuer die
  genaue Lage seiner Kante etwas mehr.
- **Die tiefste Stelle im Bild ist nicht der Boden.** Im geschlossenen
  Kronendach liegt sie im Median bei 0.92 der Flughoehe. Wer die Hoehe ueber
  Boden aus einer angenommenen Flughoehe rechnet, sollte das einkalkulieren.
- **Ohne bekannte Flughoehe** bleibt die Kronenhoehe ablesbar (eine Differenz
  braucht keinen Bezugspunkt), die absolute Hoehe ueber Boden nicht.

## Herkunft der Daten

FORTRESS, Schiefer, Frey & Kattenborn 2022, CC BY 4.0. Wer Ergebnisse dieses
Modells veroeffentlicht, sollte den Datensatz zitieren.
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
    kopf = "| Variante | " + " | ".join(spalten) + " |"
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
    verlustbeschreibung = "nicht in den Checkpoint-Metadaten dokumentiert"
    depthft = ziel / "depthft.json"
    if depthft.exists():
        info = json.loads(depthft.read_text())
        verlustbeschreibung = (
            "Huber auf der metrischen Hoehe"
            if info.get("loss") == "huber_hoehe"
            else "L1 auf der Log-Tiefe (alter Trainingsstand)"
        )
        einstellungen = (
            "| Einstellung | Wert |\n|---|---|\n"
            f"| trainierte Teile | `{info.get('trainable')}` |\n"
            f"| Flughoehe | {info.get('hoehe_min')} bis {info.get('hoehe_max')} m |\n"
            f"| Bildwinkel | {info.get('fov_min')} bis {info.get('fov_max')} Grad |\n"
            f"| Ausschnitt | {info.get('crop_px')} px |\n"
            f"| beste Epoche | {info.get('epoche')} |\n"
        )
        gebiete = ", ".join(info.get("train_gebiete", []))
        if gebiete:
            einstellungen += f"\nTrainingsgebiete: {gebiete}\n"

    metriken = "_Noch nicht ausgewertet -- `depthft/evaluate.py` liefert die Zahlen._"
    if args.metriken.exists():
        daten = json.loads(args.metriken.read_text())
        metriken = (f"Auf {daten['ausschnitte']} Ausschnitten der Testgebiete "
                    f"({', '.join(daten['gebiete'])}), die im Training nicht vorkamen:\n\n"
                    + tabelle(daten["varianten"], ["absrel", "mae_m", "rmse_m", "bias_m", "delta125"])
                    + "\n\n`mae_m` ist zugleich der Fehler der Hoehe ueber Boden -- die Flughoehe "
                      "kuerzt sich in der Differenz heraus.")

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
