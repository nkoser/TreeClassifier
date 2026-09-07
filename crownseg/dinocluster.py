"""Dichte DINOv3-Merkmale: taugen die Flaechenmerkmale zum Clustern?

Bisher lief alles ueber Ausschnitte: eine Krone wird ausgeschnitten, der
Backbone liefert *einen* Vektor, der wird geclustert oder klassifiziert. DINOv3
liefert aber je 16x16-Bildfeld einen eigenen Vektor, und diese Feldmerkmale
tragen bekanntlich eine emergente Segmentierung -- Objekte trennen sich in den
Merkmalen, ohne dass jemand Masken gezeigt hat (LOST, TokenCut, STEGO).

Zwei Fragen, die davon abhaengen, und die getrennt gemessen gehoeren:

  Flaeche -> Art   Trennen die Feldmerkmale Baumarten, ohne Labels? Gemessen
                   auf FORTRESS gegen die Artpolygone. Mitgemessen wird
                   dieselbe Groesse auf reiner RGB-Farbe -- wenn Farbe schon
                   fast so gut trennt, hat der Backbone wenig beigetragen, und
                   der Verdacht "sortiert Herbstfarben statt Arten" steht.

  Flaeche -> Baum  Trennen sie *einzelne* Kronen? Gemessen auf BAMFORESTS gegen
                   die Kronenpolygone: Zusammenhangskomponenten der Cluster als
                   Instanzen, F1 bei IoU 0.5. Meine Erwartung ist hier niedrig
                   -- benachbarte Fichten sind semantisch identisch, und genau
                   das ist der schwere Teil der Kronenabgrenzung.

    python crownseg/dinocluster.py --modus art --sites CFB014 CFB019
    python crownseg/dinocluster.py --modus instanz
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

import metrics as met  # noqa: E402
from infer_species import (  # noqa: E402
    QUEBEC_TREES_EXCLUDE, REPO_ROOT, build_dinovtree, load_class_names, resolve_device,
)

BAM_GSD_M = 0.0170


@torch.no_grad()
def feldmerkmale(backbone, bild_rgb: np.ndarray, device) -> np.ndarray:
    """Feldmerkmale eines Bildes als (h, w, d).

    Das Bild geht in voller Kachelgroesse hinein, nicht auf 512 skaliert: die
    Ortsaufloesung der Felder ist hier das Ergebnis, nicht ein Nebenprodukt.
    """
    x = torch.from_numpy(np.transpose(bild_rgb.astype(np.float32) / 255.0, (2, 0, 1)))
    felder, _ = backbone(x[None].to(device))
    if felder.dim() == 4:                      # (B, D, h, w)
        return felder[0].permute(1, 2, 0).float().cpu().numpy()
    seite = int(round(felder.shape[1] ** 0.5))  # (B, N, D)
    return felder[0].reshape(seite, seite, -1).float().cpu().numpy()


def kmeans(punkte: np.ndarray, k: int, seed: int = 0, runden: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """k-Means auf L2-normierten Vektoren -- Kosinusabstand, wie bei DINO ueblich."""
    x = punkte / np.linalg.norm(punkte, axis=1, keepdims=True).clip(1e-6)
    rng = np.random.default_rng(seed)
    zentren = x[rng.choice(len(x), k, replace=False)]
    for _ in range(runden):
        zuordnung = np.argmax(x @ zentren.T, axis=1)
        for j in range(k):
            treffer = x[zuordnung == j]
            if len(treffer):
                zentren[j] = treffer.mean(0) / np.linalg.norm(treffer.mean(0)).clip(1e-6)
    return np.argmax(x @ zentren.T, axis=1), zentren


def nmi_und_reinheit(cluster: np.ndarray, wahrheit: np.ndarray) -> tuple[float, float]:
    """Normierte Transinformation und Reinheit zwischen zwei Zuordnungen."""
    k, c = cluster.max() + 1, wahrheit.max() + 1
    tafel = np.zeros((k, c))
    np.add.at(tafel, (cluster, wahrheit), 1)
    n = tafel.sum()
    p_kc, p_k, p_c = tafel / n, tafel.sum(1) / n, tafel.sum(0) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        anteil = np.where(tafel > 0, p_kc * np.log(p_kc / np.outer(p_k, p_c)), 0.0)
        h_k = -np.sum(np.where(p_k > 0, p_k * np.log(p_k), 0.0))
        h_c = -np.sum(np.where(p_c > 0, p_c * np.log(p_c), 0.0))
    nmi = float(anteil.sum() / max(1e-9, (h_k * h_c) ** 0.5))
    return nmi, float(tafel.max(1).sum() / n)


def komponenten(karte: np.ndarray, kachel: int, min_flaeche: int) -> list[met.Instance]:
    """Zusammenhangskomponenten je Clusterwert als Instanzen."""
    gross = cv2.resize(karte.astype(np.int32), (kachel, kachel), interpolation=cv2.INTER_NEAREST)
    heraus = []
    for wert in np.unique(gross):
        anzahl, teile = cv2.connectedComponents((gross == wert).astype(np.uint8))
        for i in range(1, anzahl):
            maske = teile == i
            if maske.sum() < min_flaeche:
                continue
            instanz = met.instance_from_mask(maske, 1.0)
            if instanz is not None:
                heraus.append(instanz)
    return heraus


PALETTE = np.array([
    (228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163), (255, 127, 0),
    (255, 255, 51), (166, 86, 40), (247, 129, 191), (153, 153, 153), (26, 188, 156),
    (241, 196, 15), (142, 68, 173), (52, 73, 94), (231, 76, 60), (46, 204, 113),
    (52, 152, 219), (155, 89, 182), (241, 90, 34), (127, 140, 141), (39, 174, 96),
], np.uint8)


def einfaerben(karte: np.ndarray, form: tuple[int, int]) -> np.ndarray:
    """Clusterkarte als BGR-Bild in Zielgroesse."""
    farbig = PALETTE[karte % len(PALETTE)][:, :, ::-1]
    return cv2.resize(farbig, form[::-1], interpolation=cv2.INTER_NEAREST)


def nebeneinander(teile: list[tuple[str, np.ndarray]]) -> np.ndarray:
    beschriftet = []
    for titel, bild in teile:
        canvas = bild.copy()
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(canvas, titel, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        beschriftet.append(canvas)
    return np.hstack(beschriftet)


def modus_bild(args, backbone, device) -> None:
    """Clusterkarten zum Anschauen -- auf den eigenen Frames.

    Die Zahlen sagen, dass die Feldmerkmale Arten trennen und Kronen nicht.
    Hier ist zu sehen, was das bedeutet: zusammenhaengende Flaechen gleicher
    Farbe folgen Bestandsgrenzen, nicht Baumgrenzen.
    """
    args.out.mkdir(parents=True, exist_ok=True)
    frames = []
    for ordner in sorted(args.frames.iterdir()):
        if not ordner.is_dir():
            continue
        bilder = [x for x in sorted(ordner.iterdir()) if x.suffix.lower() in (".jpg", ".png")]
        frames.extend(bilder[: args.pro_ordner])
    for pfad in frames:
        bild = cv2.imread(str(pfad))
        hoehe, breite = bild.shape[:2]
        # Auf ein Vielfaches der Feldgroesse bringen, Seitenverhaeltnis behalten.
        neu_b = args.kachel
        neu_h = int(round(hoehe * neu_b / breite / 16)) * 16
        klein = cv2.resize(bild, (neu_b, neu_h), interpolation=cv2.INTER_AREA)
        f = feldmerkmale(backbone, cv2.cvtColor(klein, cv2.COLOR_BGR2RGB), device)

        teile = [(f"{pfad.parent.name}/{pfad.stem}", bild)]
        for k in args.k:
            karte, _ = kmeans(f.reshape(-1, f.shape[-1]), k)
            karte = karte.reshape(f.shape[0], f.shape[1])
            farbig = einfaerben(karte, (hoehe, breite))
            teile.append((f"DINOv3, {k} Cluster", cv2.addWeighted(bild, 0.45, farbig, 0.55, 0)))
        ziel = args.out / f"{pfad.parent.name}_{pfad.stem}_dinov3.jpg"
        cv2.imwrite(str(ziel), nebeneinander(teile), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  {ziel.name}", flush=True)
    print(f"\n-> {args.out}")


def lade_backbone(args, device):
    modell = build_dinovtree(args.ckpt,
                             n_classes=len(load_class_names(args.categories, QUEBEC_TREES_EXCLUDE)),
                             max_height=30.0, device=device)
    modell.eval()
    return modell.backbone


def modus_art(args, backbone, device) -> None:
    """Feldmerkmale gegen die Artpolygone von FORTRESS."""
    import rasterio
    from rasterio.windows import Window

    from fortress import read_dbf, read_shp

    shapes = args.fortress / "10.35097-538/data/dataset/shapefiles/shapefile"
    orthos = args.fortress / "orthomosaic/orthomosaic"
    sites = args.sites or sorted(p.stem.replace("_ortho", "") for p in orthos.glob("*_ortho.tif"))[:12]
    arten = sorted({a for s in sites if (shapes / f"poly_{s}.dbf").exists()
                    for a in read_dbf(shapes / f"poly_{s}.dbf")})
    zu_id = {a: i + 1 for i, a in enumerate(arten)}

    merkmale, farben, orte, wahr = [], [], [], []
    for site in sites:
        ortho, shp = orthos / f"{site}_ortho.tif", shapes / f"poly_{site}.shp"
        if not ortho.exists() or not shp.exists():
            continue
        with rasterio.open(ortho) as src:
            faktor = abs(src.transform.a) / BAM_GSD_M
            inverse = ~src.transform
            polygone = [(np.stack(inverse * (r[:, 0], r[:, 1]), 1).astype(np.float32), zu_id[a])
                        for teile, a in zip(read_shp(shp), read_dbf(shp.with_suffix(".dbf")))
                        for r in teile]
            quelle = int(round(args.kachel / faktor))
            schritt = int(round(args.kachel * 1.5 / faktor))
            genommen = 0
            for y0 in range(0, max(1, src.height - quelle), schritt):
                for x0 in range(0, max(1, src.width - quelle), schritt):
                    if genommen >= args.pro_gebiet:
                        break
                    patch = np.ascontiguousarray(np.transpose(
                        src.read((1, 2, 3), window=Window(x0, y0, quelle, quelle),
                                 out_shape=(3, args.kachel, args.kachel)), (1, 2, 0)))
                    if (patch.max(axis=2) == 0).mean() > 0.10:
                        continue
                    semantik = np.zeros((args.kachel, args.kachel), np.uint8)
                    skal = args.kachel / quelle
                    for ring, klasse in polygone:
                        p = np.round((ring - (x0, y0)) * skal).astype(np.int32)
                        if (p[:, 0].max() < 0 or p[:, 1].max() < 0
                                or p[:, 0].min() >= args.kachel or p[:, 1].min() >= args.kachel):
                            continue
                        cv2.fillPoly(semantik, [p], int(klasse))
                    if (semantik > 0).mean() < 0.20:
                        continue

                    f = feldmerkmale(backbone, patch, device)
                    seite = f.shape[0]
                    klein = cv2.resize(semantik, (seite, seite), interpolation=cv2.INTER_NEAREST)
                    rgb_klein = cv2.resize(patch, (seite, seite), interpolation=cv2.INTER_AREA)
                    gueltig = klein > 0
                    yy, xx = np.mgrid[0:seite, 0:seite]
                    merkmale.append(f[gueltig])
                    farben.append(rgb_klein[gueltig].astype(np.float32))
                    # Nur der Ort im Bild, je Kachel getrennt: wenn schon das
                    # einen guten NMI ergibt, misst die Zahl oben vor allem,
                    # dass Arten in Bestaenden zusammenstehen -- nicht, dass
                    # die Merkmale Arten kennen.
                    orte.append(np.stack([xx[gueltig], yy[gueltig],
                                          np.full(gueltig.sum(), genommen * 1000.0)], 1))
                    wahr.append(klein[gueltig])
                    genommen += 1
            print(f"  {site}: {genommen} Kacheln", flush=True)

    merkmale, farben = np.concatenate(merkmale), np.concatenate(farben)
    orte, wahr = np.concatenate(orte), np.concatenate(wahr)
    vorhanden = sorted(np.unique(wahr))
    wahr = np.searchsorted(vorhanden, wahr)
    umkehr = {v: k for k, v in zu_id.items()}
    print(f"\n{len(wahr)} beschriftete Bildfelder | {len(vorhanden)} Arten: "
          f"{', '.join(umkehr[v] for v in vorhanden)}\n")
    haeufigste = np.bincount(wahr).max() / len(wahr)
    print(f"{'k':>4} {'DINOv3 NMI':>11} {'Reinh':>7}   {'RGB NMI':>8} {'Reinh':>7}   "
          f"{'Ort NMI':>8} {'Reinh':>7}")
    for k in args.k:
        ergebnis = []
        for punkte in (merkmale, farben, orte):
            c, _ = kmeans(punkte, k)
            ergebnis.append(nmi_und_reinheit(c, wahr))
        print(f"{k:4d} {ergebnis[0][0]:11.3f} {ergebnis[0][1]:7.1%}   "
              f"{ergebnis[1][0]:8.3f} {ergebnis[1][1]:7.1%}   "
              f"{ergebnis[2][0]:8.3f} {ergebnis[2][1]:7.1%}", flush=True)
    print(f"\nhaeufigste Art allein: {haeufigste:.1%}")


def modus_instanz(args, backbone, device) -> None:
    """Feldmerkmale gegen die Kronenpolygone von BAMFORESTS."""
    import bamforests as bam

    index = json.loads((args.bamforests / "annotations.json").read_text())
    stems = sorted(index)
    auswahl = stems[:: max(1, len(stems) // args.n_kacheln)][: args.n_kacheln]
    skal = args.kachel / 2048.0
    print(f"{len(auswahl)} Kacheln aus {args.bamforests}\n", flush=True)

    # Merkmale einmal rechnen, dann alle k darauf.
    zwischen = []
    for stem in auswahl:
        bild = cv2.cvtColor(cv2.imread(str(args.bamforests / f"{stem}.jpg")), cv2.COLOR_BGR2RGB)
        bild = cv2.resize(bild, (args.kachel, args.kachel), interpolation=cv2.INTER_AREA)
        ringe = [np.asarray(r, np.float32).reshape(-1, 2) * skal for r in index[stem]]
        masken, _ = bam.masks_from_rings(ringe, args.kachel, args.kachel,
                                         int(args.min_flaeche * skal ** 2), 0.0)
        wahrheit = [i for i in (met.instance_from_mask(m) for m in masken) if i is not None]
        zwischen.append((feldmerkmale(backbone, bild, device), wahrheit))
    print(f"{'k':>4} {'Instanzen':>10} {'echte':>7} {'Treffer':>8} {'Praez':>7} {'Quote':>7} {'F1':>7} {'IoU':>6}")
    for k in args.k:
        zeilen = []
        for f, wahrheit in zwischen:
            karte, _ = kmeans(f.reshape(-1, f.shape[-1]), k)
            vorher = komponenten(karte.reshape(f.shape[0], f.shape[1]), args.kachel,
                                 int(args.min_flaeche * skal ** 2))
            zeilen.append(met.evaluate(vorher, wahrheit, 0.5))
        z = met.accumulate(zeilen)
        print(f"{k:4d} {z['kronen_pred']:10d} {z['kronen_gt']:7d} "
              f"{int(z['praezision'] * z['kronen_pred']):8d} {z['praezision']:7.3f} "
              f"{z['trefferquote']:7.3f} {z['f1']:7.3f} {z.get('mittlere_iou', 0):6.3f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modus", choices=("art", "instanz", "bild"), default="art")
    parser.add_argument("--frames", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_views_dinov3")
    parser.add_argument("--ckpt", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/checkpoints/dinovtreeb_quebectrees.pth"))
    parser.add_argument("--categories", type=Path,
                        default=REPO_ROOT / "third_party" / "quebec_trees_categories.json")
    parser.add_argument("--fortress", type=Path, default=Path("/scratch/shared/nik/data/fortress"))
    parser.add_argument("--bamforests", type=Path,
                        default=Path("/scratch/shared/nik/data/bamforests/crownseg/test1"))
    parser.add_argument("--sites", nargs="*", default=None)
    parser.add_argument("--kachel", type=int, default=1024)
    parser.add_argument("--pro-gebiet", type=int, default=4)
    parser.add_argument("--pro-ordner", type=int, default=1)
    parser.add_argument("--n-kacheln", type=int, default=12)
    parser.add_argument("--min-flaeche", type=int, default=1500)
    parser.add_argument("--k", type=int, nargs="*", default=[4, 8, 12, 20])
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()

    device = resolve_device(args.device)
    backbone = lade_backbone(args, device)
    print(f"Device: {device} | Kachel {args.kachel} px\n", flush=True)
    {"art": modus_art, "instanz": modus_instanz, "bild": modus_bild}[args.modus](args, backbone, device)


if __name__ == "__main__":
    main()
