"""Artklassifikator fuer mitteleuropaeische Baeume.

Der veroeffentlichte DINOvTree-Kopf kennt 14 kanadische Klassen. Auf Quebec
erreicht er 90.1 %, auf Nik's Bestaenden liefert er Unsinn -- ein Kiefernbestand
wird zu Gelb-Birke, weil *Pinus sylvestris* im Label-Satz fehlt und das Modell
antworten muss. Das Problem ist also nicht der Backbone, sondern die Klassen, auf
die er zeigt.

Hier bleibt der Backbone unveraendert und eingefroren; neu ist nur der Kopf. Die
Trainingsdaten kommen aus `fortress.py`: 9373 Kronenausschnitte, beschriftet
durch Verschneidung unserer Segmentierung mit den Artpolygonen von FORTRESS.

Zwei Entscheidungen bestimmen, ob die Zahl am Ende etwas wert ist:

  Aufteilung nach Gebiet, nicht zufaellig. Ausschnitte desselben Gebiets teilen
  Beleuchtung, Aufnahmetag und teils denselben Baum -- eine zufaellige Aufteilung
  wuerde die Guete deutlich schoenrechnen. Kattenborn et al. haben genau das fuer
  raeumlich autokorrelierte Waldddaten gezeigt.

  Seltene Klassen getrennt ausweisen. Fichte hat 4669 Beispiele, Esche 8. Eine
  Gesamtgenauigkeit waere hier vor allem ein Mass dafuer, wie oft Fichte richtig
  erkannt wird.

    python crownseg/train_head.py --min-per-class 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cluster_crowns import extract_features  # noqa: E402
from infer_species import (  # noqa: E402
    QUEBEC_TREES_EXCLUDE, REPO_ROOT, build_dinovtree, load_class_names,
    resolve_device, to_model_input,
)


def merkmale(crops: np.ndarray, model, device, batch_size: int) -> np.ndarray:
    """Backbone-Merkmale je Ausschnitt -- einmal gerechnet, danach wiederverwendet."""
    features, _ = extract_features(model, crops, batch_size, device, "backbone")
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("/scratch/shared/nik/data/fortress/kronen"))
    parser.add_argument("--ckpt", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/checkpoints/dinovtreeb_quebectrees.pth"))
    parser.add_argument("--categories", type=Path,
                        default=REPO_ROOT / "third_party" / "quebec_trees_categories.json")
    parser.add_argument("--out", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/checkpoints/kopf_fortress.pth"))
    parser.add_argument("--min-per-class", type=int, default=100,
                        help="Klassen darunter fliegen raus -- zu wenig zum Lernen und zum Messen.")
    parser.add_argument("--test-sites", type=float, default=0.25, help="Anteil der Gebiete fuer den Test.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)   # ohne festen Startwert schwankt die
    # ausgewogene Genauigkeit zwischen Laeufen um rund 2 Punkte.
    tabelle = pd.read_csv(args.data / "kronen.csv")
    haeufig = tabelle["art"].value_counts()
    behalten = haeufig[haeufig >= args.min_per_class].index.tolist()
    tabelle = tabelle[tabelle["art"].isin(behalten)].reset_index(drop=True)
    klassen = sorted(behalten)
    zu_id = {k: i for i, k in enumerate(klassen)}
    print(f"{len(tabelle)} Ausschnitte | {len(klassen)} Klassen: {', '.join(klassen)}")
    print(f"verworfen: {', '.join(f'{k} ({v})' for k, v in haeufig.items() if k not in behalten)}\n", flush=True)

    # Aufteilung nach Gebiet -- nicht nach Ausschnitt.
    gebiete = sorted(tabelle["gebiet"].unique())
    rng = np.random.default_rng(0)
    test_gebiete = set(rng.permutation(gebiete)[: max(1, int(len(gebiete) * args.test_sites))])
    ist_test = tabelle["gebiet"].isin(test_gebiete).to_numpy()
    print(f"{len(gebiete)} Gebiete, davon {len(test_gebiete)} fuer den Test: "
          f"{', '.join(sorted(test_gebiete))}")
    print(f"Training {(~ist_test).sum()} | Test {ist_test.sum()}\n", flush=True)

    backbone = build_dinovtree(args.ckpt, n_classes=len(load_class_names(args.categories, QUEBEC_TREES_EXCLUDE)),
                               max_height=30.0, device=device)
    backbone.eval()

    crops = np.stack([to_model_input(cv2.cvtColor(cv2.imread(str(args.data / "crops" / f)), cv2.COLOR_BGR2RGB))
                      for f in tabelle["datei"]])
    print(f"Merkmale rechnen fuer {len(crops)} Ausschnitte ...", flush=True)
    features = merkmale(crops, backbone, device, args.feature_batch)
    ziel = tabelle["art"].map(zu_id).to_numpy()
    print(f"Merkmalsdimension {features.shape[1]}\n", flush=True)

    X = torch.from_numpy(features.copy()).float()
    # Standardisierung aus dem Trainingsteil -- wandert mit in den Checkpoint,
    # sonst bekommt der Kopf beim Anwenden Eingaben auf anderer Skala.
    mittel, streuung = X[~ist_test].mean(0), X[~ist_test].std(0).clamp_min(1e-6)
    X = (X - mittel) / streuung
    y = torch.from_numpy(ziel).long()
    Xtr, ytr = X[~ist_test].to(device), y[~ist_test].to(device)
    Xte, yte = X[ist_test].to(device), y[ist_test].to(device)

    kopf = nn.Sequential(nn.Linear(X.shape[1], args.hidden), nn.GELU(),
                         nn.Dropout(0.3), nn.Linear(args.hidden, len(klassen))).to(device)
    # Klassengewichte: Fichte hat 4669 Beispiele, Douglasie 157. Ohne Ausgleich
    # lernt der Kopf vor allem, Fichte zu sagen.
    anzahl = np.bincount(ziel[~ist_test], minlength=len(klassen))
    gewicht = torch.tensor(len(anzahl) / np.maximum(1, anzahl) / (1 / np.maximum(1, anzahl)).sum(),
                           dtype=torch.float32, device=device)
    verlust = nn.CrossEntropyLoss(weight=gewicht)
    optimierer = torch.optim.AdamW(kopf.parameters(), lr=args.lr, weight_decay=1e-2)
    plan = torch.optim.lr_scheduler.CosineAnnealingLR(optimierer, T_max=args.epochs)

    bestes, bester_stand = -1.0, None
    for epoche in range(1, args.epochs + 1):
        kopf.train()
        perm = torch.randperm(len(Xtr), device=device)
        for start in range(0, len(perm), args.batch_size):
            index = perm[start : start + args.batch_size]
            optimierer.zero_grad(set_to_none=True)
            verlust(kopf(Xtr[index]), ytr[index]).backward()
            optimierer.step()
        plan.step()

        kopf.eval()
        with torch.no_grad():
            vorhersage = kopf(Xte).argmax(1)
        # Ausgewogene Genauigkeit: Mittel der Trefferquoten je Klasse.
        quoten = [(vorhersage[yte == k] == k).float().mean().item()
                  for k in range(len(klassen)) if (yte == k).any()]
        ausgewogen = float(np.mean(quoten))
        if ausgewogen > bestes:
            bestes, bester_stand = ausgewogen, {k: v.cpu().clone() for k, v in kopf.state_dict().items()}
        if epoche % 10 == 0 or epoche == 1:
            roh = (vorhersage == yte).float().mean().item()
            print(f"Epoche {epoche:3d}  Genauigkeit {roh:.3f}  ausgewogen {ausgewogen:.3f}", flush=True)

    kopf.load_state_dict(bester_stand)
    kopf.eval()
    with torch.no_grad():
        vorhersage = kopf(Xte).argmax(1).cpu().numpy()
    wahr = yte.cpu().numpy()

    print(f"\n=== Test auf {len(test_gebiete)} zurueckgehaltenen Gebieten ===")
    print(f"Genauigkeit {np.mean(vorhersage == wahr):.1%} | ausgewogen {bestes:.1%} | "
          f"haeufigste Klasse {pd.Series(wahr).value_counts(normalize=True).iloc[0]:.1%}\n")
    print(f"{'Klasse':24s} {'n':>5} {'Trefferquote':>13} {'meist vorhergesagt':>22}")
    for k, name in enumerate(klassen):
        maske = wahr == k
        if not maske.any():
            continue
        top = pd.Series([klassen[v] for v in vorhersage[maske]]).value_counts()
        print(f"{name:24s} {maske.sum():5d} {np.mean(vorhersage[maske] == k):12.1%} "
              f"{top.index[0][:20]:>22s}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kopf": bester_stand, "klassen": klassen, "ausgewogen": bestes,
                "mittel": mittel, "streuung": streuung,
                "test_gebiete": sorted(test_gebiete), "args": vars(args)}, args.out)
    print(f"\n{args.out}")


if __name__ == "__main__":
    main()
