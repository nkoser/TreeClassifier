"""Pures gegen feinabgestimmtes Depth Pro -- auf FORTRESS-Gebieten mit Wahrheit.

Die Testgebiete waren im Training nie zu sehen (der Split steht in
`index.json` und wird von `prepare.py` gesetzt). Aus ihnen werden dieselben
virtuellen Nadirframes geschnitten wie im Training, nur ohne Augmentierung und
mit festem Wurf -- alle Varianten sehen exakt dieselben Ausschnitte.

Verglichen werden vier Varianten, und die dritte ist die aufschlussreichste:

  pur + Bildwinkelkopf   Depth Pro so, wie man es von der Stange nimmt.
  pur + bekannte Kamera  Derselbe Lauf, aber `k` vorgegeben statt geschaetzt.
                         Trennt Fehler im Bildwinkel von Fehlern in der Tiefe.
  pur + Skalenangleich   Zusaetzlich global so skaliert, dass der Median exakt
                         stimmt. **Kein anwendbares Verfahren, sondern ein
                         Orakel**: der Faktor kommt aus der Wahrheit. Die Zeile
                         ist die Obergrenze dessen, was eine reine
                         Skalenkorrektur je erreichen koennte -- und misst
                         damit, wie gut die *relative* Struktur ist.
  pur + Hoehenanker      Skaliert, bis die tiefste Stelle im Bild (95.
                         Perzentil) der bekannten Flughoehe entspricht. Das ist
                         anwendbar, denn eine Drohne kennt ihre Hoehe aus
                         Barometer und GPS. Es setzt aber voraus, dass im Bild
                         ueberhaupt Boden zu sehen ist -- im geschlossenen
                         Kronendach ist die tiefste sichtbare Stelle nicht der
                         Boden, und der Anker verrutscht.
  feinabgestimmt         Das Ergebnis dieses Projekts: die Skala kommt aus dem
                         Bild selbst, ohne Anker und ohne Orakel.
  feinabgestimmt + Anker Beides zusammen.

Mit `--videolook` laufen dieselben Ausschnitte durch Weichzeichnung, Rauschen
und JPEG-Artefakte. Das ist kein Schoenheitsfehler, sondern die eigentliche
Frage: unsere Frames sind einzelne Videobilder, das Orthomosaik ist ein aus
vielen Aufnahmen gerechnetes, gestochen scharfes Produkt. Der Detailgrad ist bei
einer Nadiraufnahme im Wald aber der einzige Hinweis auf die Flughoehe -- sieht
das Modell mehr Details, schliesst es auf feinere Bodenaufloesung und damit auf
geringere Hoehe. Beide Zahlen nebeneinander zeigen, wie stark das durchschlaegt.

    python depthft/evaluate.py --ft /scratch/shared/nik/runs/depthft/bestes
    python depthft/evaluate.py --videolook          # in der Schaerfe unserer Frames
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inferenz  # noqa: E402
from bilder import beschriften, hoehenbild  # noqa: E402
from dataset import NadirFrames  # noqa: E402

HOEHENKLASSEN = [(0, 40), (40, 70), (70, 200)]


def messen(d_pred: np.ndarray, d_gt: np.ndarray, maske: np.ndarray) -> dict[str, float]:
    p, g = d_pred[maske], d_gt[maske]
    if p.size == 0:
        return {}
    verhaeltnis = np.maximum(p / g, g / p)
    return {
        "absrel": float(np.mean(np.abs(p - g) / g)),
        "mae_m": float(np.mean(np.abs(p - g))),
        "rmse_m": float(np.sqrt(np.mean((p - g) ** 2))),
        "bias_m": float(np.mean(p - g)),
        "delta125": float(np.mean(verhaeltnis < 1.25)),
        "skalenfehler": float(np.median(p) / np.median(g)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("/scratch/shared/nik/data/fortress/depthft"))
    parser.add_argument("--ft", type=Path, default=Path("/scratch/shared/nik/runs/depthft/bestes"),
                        help="Feinabgestimmter Checkpoint.")
    parser.add_argument("--pur", default="apple/DepthPro-hf")
    parser.add_argument("--out", type=Path, default=Path("/home/nik/workspace/TreeClassifier/results_depthft"))
    parser.add_argument("--split", default="test", choices=("test", "val", "train"))
    parser.add_argument("--pro-gebiet", type=int, default=40)
    parser.add_argument("--crop-px", type=int, default=1536)
    parser.add_argument("--seitenverhaeltnis", type=float, default=16 / 9)
    parser.add_argument("--hoehe-min", type=float, default=25.0)
    parser.add_argument("--hoehe-max", type=float, default=120.0)
    parser.add_argument("--abstand-min", type=float, default=20.0)
    parser.add_argument("--fov-min", type=float, default=60.0,
                        help="Zur Auswertung enger als im Training: um unsere Kamera herum.")
    parser.add_argument("--fov-max", type=float, default=85.0)
    parser.add_argument("--strahl-tiefe", action="store_true",
                        help="Muss zum Training passen -- steht in bestes/depthft.json.")
    parser.add_argument("--videolook", action="store_true",
                        help="Ausschnitte auf die Schaerfe unserer Videoframes bringen.")
    parser.add_argument("--beispiele", type=int, default=6)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "beispiele").mkdir(exist_ok=True)

    daten = NadirFrames(args.data, args.split, crop_px=args.crop_px,
                        seitenverhaeltnis=args.seitenverhaeltnis,
                        hoehe_min=args.hoehe_min, hoehe_max=args.hoehe_max,
                        abstand_min=args.abstand_min,
                        fov_min=args.fov_min, fov_max=args.fov_max,
                        pro_gebiet=args.pro_gebiet, augment=False, spiegeln=False,
                        domaene=args.videolook, cache=2, seed=args.seed,
                        strahl_tiefe=args.strahl_tiefe)
    print(f"{len(daten)} Ausschnitte aus {len(daten.sites)} Gebieten: {', '.join(daten.sites)}", flush=True)

    # Ausschnitte einmal erzeugen und behalten -- so sehen alle Varianten
    # garantiert dieselben Bilder, auch wenn sich am Datensatz etwas aendert.
    proben = [daten[i] for i in range(len(daten))]
    zeigen = set(np.linspace(0, len(proben) - 1, min(args.beispiele, len(proben))).astype(int).tolist())

    zeilen: list[dict] = []
    karten: dict[int, dict[str, np.ndarray]] = {i: {} for i in zeigen}

    for name, quelle, mit_fov in (("pur", args.pur, True), ("feinabgestimmt", str(args.ft), False)):
        if name == "feinabgestimmt" and not args.ft.exists():
            print(f"\n{args.ft} fehlt -- nur das pure Modell wird ausgewertet.", flush=True)
            continue
        print(f"\n--- {name}: {quelle} ---", flush=True)
        model = inferenz.lade(quelle, device, fov_head=mit_fov)

        for i, probe in enumerate(proben):
            bild = probe["bild"].numpy().transpose(1, 2, 0)
            d_gt = probe["tiefe"].numpy()
            maske = probe["maske"].numpy()
            k_wahr, H = float(probe["k"]), float(probe["flughoehe"])

            D, fov = inferenz.roh(model, bild, device=device)
            varianten = {f"{name}_kamera": k_wahr / D}
            if fov is not None:
                varianten[f"{name}_fovkopf"] = inferenz.k_von_fov(fov) / D
            d = varianten[f"{name}_kamera"]
            # Anker aus der bekannten Flughoehe -- ohne Wahrheit, also anwendbar.
            boden = float(np.percentile(d[maske], 95))
            varianten[f"{name}_hoehenanker"] = d * (H / max(boden, 1e-6))
            if name == "pur":
                # Orakel: kennt den wahren Median. Nur als Obergrenze zu lesen.
                faktor = float(np.median(d_gt[maske]) / max(np.median(d[maske]), 1e-6))
                varianten["pur_skalenangleich"] = d * faktor

            for variante, d_pred in varianten.items():
                werte = messen(d_pred, d_gt, maske)
                if not werte:
                    continue
                werte.update(variante=variante, ausschnitt=i, gebiet=probe["site"],
                             flughoehe_m=H, fov_grad=inferenz.fov_von_k(k_wahr),
                             gsd_cm=float(probe["gsd"]) * 100,
                             fov_geschaetzt=fov if fov is not None else np.nan,
                             hoehe_p95_gt=float(np.percentile(probe["hoehe"].numpy()[maske], 95)))
                zeilen.append(werte)

            if i in zeigen:
                karten[i]["rgb"] = bild
                karten[i]["gt"] = probe["hoehe"].numpy()
                karten[i][f"{name}_h"] = H - varianten[f"{name}_kamera"]
                karten[i]["info"] = f"{probe['site']} | {H:.0f} m | GSD {float(probe['gsd'])*100:.1f} cm"
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(proben)}", flush=True)

        del model
        torch.cuda.empty_cache()

    tabelle = pd.DataFrame(zeilen)
    kennung = f"{args.split}{'_videolook' if args.videolook else ''}"
    tabelle.to_csv(args.out / f"metriken_{kennung}.csv", index=False)

    spalten = ["absrel", "mae_m", "rmse_m", "bias_m", "delta125", "skalenfehler"]
    # Ausschnitte desselben Gebiets sind stark korreliert. Deshalb ist das
    # Gebiet, nicht der Ausschnitt, die statistische Einheit: erst innerhalb
    # eines Gebiets mitteln, dann alle Gebiete gleich gewichten.
    je_gebiet = tabelle.groupby(["variante", "gebiet"])[spalten].mean()
    zusammen = je_gebiet.groupby("variante").mean().sort_values("absrel")
    streuung = je_gebiet.groupby("variante").std()
    je_gebiet.to_csv(args.out / f"metriken_{kennung}_je_gebiet.csv")
    print(f"\n=== {kennung}, {len(proben)} Ausschnitte ===")
    print(zusammen.to_string(float_format=lambda v: f"{v:8.3f}"))

    tabelle["hoehenklasse"] = pd.cut(tabelle["flughoehe_m"], [g[0] for g in HOEHENKLASSEN] + [HOEHENKLASSEN[-1][1]],
                                     labels=[f"{a}-{b} m" for a, b in HOEHENKLASSEN])
    nach_hoehe = tabelle.groupby(["variante", "hoehenklasse"], observed=True)[["absrel", "mae_m"]].mean()
    print("\n--- nach Flughoehe ---")
    print(nach_hoehe.to_string(float_format=lambda v: f"{v:8.3f}"))
    nach_hoehe.to_csv(args.out / f"metriken_{kennung}_nach_hoehe.csv")

    if "fov_geschaetzt" in tabelle and tabelle["fov_geschaetzt"].notna().any():
        teil = tabelle[tabelle["variante"] == "pur_kamera"]
        print(f"\nBildwinkel: wahr {teil['fov_grad'].mean():.1f} Grad, "
              f"vom Kopf geschaetzt {teil['fov_geschaetzt'].mean():.1f} Grad")

    for i, eintrag in karten.items():
        if "gt" not in eintrag:
            continue
        obergrenze = float(np.percentile(eintrag["gt"], 99)) or 1.0
        kacheln = [beschriften(cv2.cvtColor(eintrag["rgb"], cv2.COLOR_RGB2BGR), eintrag["info"]),
                   beschriften(hoehenbild(eintrag["gt"], obergrenze), "Wahrheit (nDSM)")]
        for name, titel in (("pur_h", "pur"), ("feinabgestimmt_h", "feinabgestimmt")):
            if name in eintrag:
                kacheln.append(beschriften(hoehenbild(eintrag[name], obergrenze),
                                           f"{titel} (0-{obergrenze:.0f} m)"))
        cv2.imwrite(str(args.out / "beispiele" / f"{kennung}_{i:03d}.jpg"), np.hstack(kacheln),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

    (args.out / f"zusammenfassung_{kennung}.json").write_text(
        json.dumps({"ausschnitte": len(proben), "gebiete": daten.sites,
                    "statistische_einheit": "gebiet",
                    "varianten": zusammen.to_dict(orient="index"),
                    "streuung_zwischen_gebieten": streuung.to_dict(orient="index")}, indent=2))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
