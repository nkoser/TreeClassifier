"""Pur gegen feinabgestimmt, so dargestellt, dass man es lesen kann.

Eine gemeinsame Farbskala fuer beide Modelle klingt fair, macht die Abbildung
aber unbrauchbar: pures Depth Pro liegt um Faktor 50 daneben, also ist seine
Kachel durchgehend gesaettigt und zeigt nichts. Umgekehrt verschweigt eine je
eigene Skala genau den Fehler, um den es geht.

Deshalb beides nebeneinander, plus ein Schnitt quer durchs Bild:

  Zeile 1   Bild, Wahrheit und beide Vorhersagen -- jede mit **eigener** Skala.
            Hier ist zu sehen, ob die *Struktur* stimmt: sitzen Kronen dort, wo
            Kronen sind?
  Zeile 2   Dieselben Karten auf **gemeinsamer** Skala, an der Wahrheit
            ausgerichtet. Hier ist zu sehen, ob der *Massstab* stimmt.
  Zeile 3   Ein waagerechter Schnitt durch die Bildmitte, alle Kurven in einem
            Diagramm. Das ist die ehrlichste Ansicht: die Kurve von pur liegt
            flach bei einem Meter, waehrend die Wahrheit ueber 30 m schwingt.

Auf den FORTRESS-Testgebieten laeuft die Wahrheit mit. Auf unseren eigenen
Frames gibt es keine -- dort bleiben Bild und die beiden Vorhersagen.

    python depthft/vergleichsbild.py --quelle fortress --n 4
    python depthft/vergleichsbild.py --quelle frames --hfov-deg 48.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inferenz  # noqa: E402
from bilder import (  # noqa: E402
    balkendiagramm, beschriften, einfache_abbildung, farbskala, hoehenbild,
)

BILDENDUNGEN = {".jpg", ".jpeg", ".png"}


def kachel(hoehe: np.ndarray, unten: float, oben: float, titel: str, zweite: str) -> np.ndarray:
    return beschriften(hoehenbild(hoehe, oben, unten), titel, zweite)


def schnitt_diagramm(kurven: list[tuple[str, np.ndarray, tuple[int, int, int]]],
                     breite: int, hoehe: int, y_titel: str) -> np.ndarray:
    """Einfaches Liniendiagramm, ohne Zusatzbibliothek gezeichnet."""
    bild = np.full((hoehe, breite, 3), 22, np.uint8)
    rand_l, rand_u, rand_r, rand_o = 70, 44, 18, 40
    flaeche = (breite - rand_l - rand_r, hoehe - rand_u - rand_o)
    alle = np.concatenate([k for _, k, _ in kurven])
    if not np.isfinite(alle).any():
        return bild
    lo, hi = float(np.nanmin(alle)), float(np.nanmax(alle))
    spanne = max(hi - lo, 1e-6)
    lo, hi = lo - 0.05 * spanne, hi + 0.05 * spanne

    def punkt(i: int, n: int, wert: float) -> tuple[int, int]:
        x = rand_l + int(i * (flaeche[0] - 1) / max(n - 1, 1))
        y = rand_o + int((1.0 - (wert - lo) / (hi - lo)) * (flaeche[1] - 1))
        return x, y

    for anteil in np.linspace(0, 1, 6):
        wert = lo + anteil * (hi - lo)
        y = rand_o + int((1.0 - anteil) * (flaeche[1] - 1))
        cv2.line(bild, (rand_l, y), (breite - rand_r, y), (52, 52, 52), 1)
        cv2.putText(bild, f"{wert:6.1f}", (6, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(bild, y_titel, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    for spalte, (name, kurve, farbe) in enumerate(kurven):
        n = len(kurve)
        # Luecken (keine Wahrheit vorhanden) bleiben Luecken, statt als Linie
        # auf null durchzulaufen.
        vorher = None
        for i, w in enumerate(kurve):
            if not np.isfinite(w):
                vorher = None
                continue
            jetzt = punkt(i, n, float(w))
            if vorher is not None:
                cv2.line(bild, vorher, jetzt, farbe, 2, cv2.LINE_AA)
            vorher = jetzt
        x = rand_l + 12 + spalte * max(150, flaeche[0] // max(len(kurven), 1))
        cv2.line(bild, (x, hoehe - 16), (x + 26, hoehe - 16), farbe, 3, cv2.LINE_AA)
        cv2.putText(bild, name, (x + 32, hoehe - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    farbe, 1, cv2.LINE_AA)
    return bild


def eine_abbildung(rgb: np.ndarray, karten: list[tuple[str, np.ndarray, tuple[int, int, int]]],
                   kopf: str, bezug: np.ndarray | None) -> np.ndarray:
    """Zwei Kachelzeilen und ein Schnitt, untereinander."""
    h, w = rgb.shape[:2]
    ziel_h = 420
    faktor = ziel_h / h
    klein = lambda a: cv2.resize(a, (int(w * faktor), ziel_h), interpolation=cv2.INTER_AREA)  # noqa: E731

    # Gemeinsame Skala: an der Wahrheit ausgerichtet, sonst am ersten Eintrag.
    grundlage = bezug if bezug is not None else karten[0][1]
    g_unten, g_oben = 0.0, float(max(np.nanpercentile(grundlage, 99), 5.0))

    eigene, gemeinsam = [beschriften(klein(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)), kopf)], \
                        [beschriften(klein(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)), "dasselbe Bild")]
    for name, karte, _ in karten:
        unten = float(np.nanpercentile(karte, 1))
        oben = float(np.nanpercentile(karte, 99))
        if oben - unten < 1e-3:
            oben = unten + 1.0
        eigene.append(kachel(klein(karte), unten, oben, name, f"eigene Skala {unten:.1f}-{oben:.1f} m"))
        gemeinsam.append(kachel(klein(karte), g_unten, g_oben, name,
                                f"gemeinsame Skala 0-{g_oben:.0f} m"))

    zeile1 = np.hstack(eigene + [farbskala(60, ziel_h, 1.0)[:, :0]])   # ohne Keil, Skalen verschieden
    zeile2 = np.hstack(gemeinsam + [farbskala(60, ziel_h, g_oben)])
    breite = max(zeile1.shape[1], zeile2.shape[1])
    for i, z in enumerate([zeile1, zeile2]):
        if z.shape[1] < breite:
            pad = np.zeros((z.shape[0], breite - z.shape[1], 3), np.uint8)
            (zeile1, zeile2)[i]
            if i == 0:
                zeile1 = np.hstack([z, pad])
            else:
                zeile2 = np.hstack([z, pad])

    mitte = h // 2
    kurven = [(name, karte[mitte, ::4], farbe) for name, karte, farbe in karten]
    diagramm = schnitt_diagramm(kurven, breite, 300, "Hoehe ueber Boden (m), Schnitt durch die Bildmitte")
    return np.vstack([zeile1, zeile2, diagramm])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quelle", default="fortress", choices=("fortress", "frames"))
    parser.add_argument("--data", type=Path, default=Path("/scratch/shared/nik/data/fortress/depthft"))
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--ft", type=Path, default=Path("/scratch/shared/nik/runs/depthft/bestes"))
    parser.add_argument("--pur", default="apple/DepthPro-hf")
    parser.add_argument("--out", type=Path,
                        default=Path("/home/nik/workspace/TreeClassifier/results_depthft/vergleich"))
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--split", default="test")
    parser.add_argument("--hfov-deg", type=float, default=48.0, help="Nur fuer --quelle frames.")
    parser.add_argument("--altitudes", nargs="*", metavar="ORDNER=HOEHE",
                        default=["dense=51", "dense1=69", "mixed=92", "mixed1=103",
                                 "pines=60", "urban=120"])
    parser.add_argument("--stil", default="einfach", choices=("einfach", "ausfuehrlich"),
                        help="einfach: jede Karte eigene Skala plus Balken. "
                             "ausfuehrlich: zusaetzlich gemeinsame Skala und Schnitt.")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    GRUEN, ROT, BLAU = (120, 220, 120), (110, 110, 245), (245, 190, 110)

    faelle = []
    if args.quelle == "fortress":
        from dataset import NadirFrames
        # Ein Ausschnitt je Gebiet: die Sampleliste ist nach Gebieten gruppiert,
        # mehrere je Gebiet lieferten sonst n-mal denselben Bestand.
        daten = NadirFrames(args.data, args.split, crop_px=1536, pro_gebiet=1,
                            augment=False, cache=1, seed=args.seed, fov_min=60.0, fov_max=85.0)
        for i in range(min(args.n, len(daten))):
            p = daten[i]
            faelle.append({
                "rgb": p["bild"].numpy().transpose(1, 2, 0),
                "k": float(p["k"]), "H": float(p["flughoehe"]),
                "wahrheit": np.where(p["maske"].numpy(), p["hoehe"].numpy(), np.nan),
                "name": f"{p['site']}", "kopf": f"{p['site']}, {float(p['flughoehe']):.0f} m Flughoehe",
            })
    else:
        vorgaben = {e.split("=")[0]: float(e.split("=")[1]) for e in args.altitudes}
        k = inferenz.k_von_fov(args.hfov_deg)
        ordner = sorted(o for o in args.input.iterdir() if o.is_dir())
        for o in ordner[: args.n] if args.n else ordner:
            treffer = sorted(f for f in o.iterdir() if f.suffix.lower() in BILDENDUNGEN)
            if not treffer:
                continue
            import re
            m = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*m?\s*", o.name, re.IGNORECASE)
            H = float(m.group(1)) if m else vorgaben.get(o.name, 100.0)
            faelle.append({
                "rgb": cv2.cvtColor(cv2.imread(str(treffer[0])), cv2.COLOR_BGR2RGB),
                "k": k, "H": H, "wahrheit": None, "name": o.name,
                "kopf": f"{o.name}, {H:.0f} m Flughoehe, HFOV {args.hfov_deg:.0f} Grad",
            })
    print(f"{len(faelle)} Faelle, Quelle {args.quelle}", flush=True)

    for name, quelle in (("pur", args.pur), ("feinabgestimmt", str(args.ft))):
        if name == "feinabgestimmt" and not args.ft.exists():
            continue
        model = inferenz.lade(quelle, device, fov_head=False)
        for fall in faelle:
            D, _ = inferenz.roh(model, fall["rgb"], device=device)
            fall[name] = fall["H"] - fall["k"] / D
        del model
        torch.cuda.empty_cache()

    for nummer, fall in enumerate(faelle):
        karten = []
        if fall["wahrheit"] is not None:
            karten.append(("Wahrheit (nDSM)", fall["wahrheit"], GRUEN))
        karten.append(("pur Depth Pro", fall["pur"], ROT))
        if "feinabgestimmt" in fall:
            karten.append(("feinabgestimmt", fall["feinabgestimmt"], BLAU))
        if args.stil == "einfach":
            abbildung = einfache_abbildung(cv2.cvtColor(fall["rgb"], cv2.COLOR_RGB2BGR),
                                           karten, fall["kopf"])
        else:
            abbildung = eine_abbildung(fall["rgb"], karten, fall["kopf"],
                                       fall["wahrheit"] if fall["wahrheit"] is not None else None)
        ziel = args.out / f"{args.quelle}_{args.stil}_{nummer:02d}_{fall['name']}.jpg"
        cv2.imwrite(str(ziel), abbildung, [cv2.IMWRITE_JPEG_QUALITY, 90])
        werte = " | ".join(f"{n}: p95 {np.nanpercentile(k, 95):5.1f} m" for n, k, _ in karten)
        print(f"  {ziel.name}  {werte}", flush=True)

    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
